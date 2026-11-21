"""
bess_functions.py - battery model, optimiser and settlement for SOLA4237 Assignment 2.

Used by NB05-NB08. Keep this file in the Drive folder SOLA4237-A2 (next to the data).

Contents
- BATTERY            : default battery parameters (13.5 kWh / 5 kW AC-coupled, sources in Notes Part B Q5/Q6)
- retail_prices()    : wholesale-passthrough import/export price from a spot price
- charge_derate()    : temperature derating of charging power
- solve_bess()       : one linear program (LP) for one planning window (scipy.optimize.linprog, HiGHS)
- window()           : next H hourly values of a series (for building plans)
- run_operation()    : rolling-horizon operation: plan every hour, execute every 5 minutes
- settle()           : annual bill, import/export, self-consumption, cycles, ...

Sign convention (Week 3): grid import positive, export negative; battery power positive = discharging.
Time convention (Notes Q4): every timestamp labels the END of its interval; the hour ending H holds the
5-min rows labelled H-55min ... H.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

# ----------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------
BATTERY = {
    'e_cap_kwh': 13.5,          # usable capacity (Powerwall 2 AC datasheet [R20])
    'p_max_kw': 5.0,            # continuous charge/discharge power [R20]
    'eta_ch': 0.949,            # sqrt(0.90 round-trip) [R20]
    'eta_dis': 0.949,
    'soc_min': 0.10,            # 10-95 % SoC window [R17]
    'soc_max': 0.95,
    'soc0': 0.50,               # initial SoC (modelling choice)
    'c_deg': 0.005,             # $/kWh discharged, operational/degradation cost [R18]
    'max_dis_kwh_day': 10.4,    # warranty throughput 37.8 MWh / 3650 days [R22] (cost modes only)
}

LOSS_FACTOR = 1.062             # Ergon East MLF x DLF 2023-24 [R26]
ENV_MKT_AUD_KWH = 0.01891       # LRET + SRES + other market costs (Ergon) [R26]
GST = 1.10
NET_EVENING, NET_DAY, NET_NIGHT = 0.16734, 0.03163, 0.04947   # ERTOUET1 NUOS $/kWh ex GST [R28]


def scaled_battery(e_cap_kwh, base=BATTERY):
    """
    What it does: battery parameters for another size (sizing sweep). Power = max(3 kW, 0.37 x capacity),
    warranty throughput scaled with capacity (same cycles per day).
    Input: e_cap_kwh (float). Output: dict like BATTERY.
    """
    b = dict(base)
    b['e_cap_kwh'] = float(e_cap_kwh)
    b['p_max_kw'] = max(3.0, round(0.37 * e_cap_kwh, 1))
    b['max_dis_kwh_day'] = base['max_dis_kwh_day'] * e_cap_kwh / base['e_cap_kwh']
    return b


def network_rate(hour):
    """ERTOUET1 network energy rate ($/kWh ex GST) by label hour: 16-21 evening, 9-16 day, else night."""
    hour = np.asarray(hour)
    return np.select([(hour >= 16) & (hour < 21), (hour >= 9) & (hour < 16)], [NET_EVENING, NET_DAY], default=NET_NIGHT)


def retail_prices(spot_aud_kwh, hour):
    """
    What it does: wholesale-passthrough prices from a spot price (Notes D.4.4).
    import = (spot x 1.062 + network ToU + 0.01891) x 1.10 ; export = spot
    Input: spot_aud_kwh (array, $/kWh), hour (array of label hours 0-23).
    Output: (import_aud_kwh, export_aud_kwh) as numpy arrays.
    """
    spot = np.asarray(spot_aud_kwh, dtype=float)
    imp = (spot * LOSS_FACTOR + network_rate(hour) + ENV_MKT_AUD_KWH) * GST
    return imp, spot.copy()


def charge_derate(temp_c):
    """
    What it does: share of rated charging power allowed at ambient temperature temp_c.
    100 % at >= 10 C, linear to 0 % at 0 C (thresholds [R21, R23]; linear shape = modelling choice).
    Discharge is not derated (operating range -20 to 50 C [R20]).
    """
    return np.clip(np.asarray(temp_c, dtype=float) / 10.0, 0.0, 1.0)


# ----------------------------------------------------------------------------------------------
# One LP for one planning window
# ----------------------------------------------------------------------------------------------
def solve_bess(load_kw, pv_kw, c_imp, c_exp, k_ch, e_start_kwh, bat=BATTERY, dt_h=1.0,
               mode='cost', allow_curtail=False, max_dis_kwh=None):
    """
    What it does
        Solves one battery scheduling LP over T steps (Notes D.4.5) with scipy.optimize.linprog (HiGHS).
        Decision variables per step t: p_ch, p_dis, g_imp, g_exp, curt (kW) and E (kWh, SoC at the END of t).
    Objective
        mode='cost'     : min sum (c_imp*g_imp - c_exp*g_exp + c_deg*p_dis) * dt
        mode='selfcons' : min sum g_imp * dt  (no grid charging, no battery export)
    Constraints
        (1) pv - curt + g_imp + p_dis = load + p_ch + g_exp
        (2) E_t = E_{t-1} + eta_ch*p_ch*dt - p_dis*dt/eta_dis   (E_0 = e_start)
        (3) soc_min*Ecap <= E_t <= soc_max*Ecap
        (4) p_ch <= p_max*k_ch(t), p_dis <= p_max
        (5) curt <= pv only if allow_curtail and c_exp < 0, else curt = 0
        (6) optional: sum p_dis*dt <= max_dis_kwh  (warranty throughput)
    Input
        load_kw, pv_kw, c_imp, c_exp, k_ch : arrays of length T ($/kWh for prices, k_ch in [0,1])
        e_start_kwh : SoC at the start (kWh); bat : parameter dict; dt_h : step length (h)
    Output
        dict of numpy arrays: p_ch, p_dis, g_imp, g_exp, curt, E ; plus 'cost' (objective value, $) and 'status'.
    """
    load, pv = np.asarray(load_kw, float), np.asarray(pv_kw, float)
    c_imp, c_exp, k_ch = np.asarray(c_imp, float), np.asarray(c_exp, float), np.asarray(k_ch, float)
    T = len(load)
    n = 6 * T
    # variable blocks: [p_ch | p_dis | g_imp | g_exp | curt | E]
    PCH, PDIS, GIMP, GEXP, CURT, E = (np.arange(T) + k * T for k in range(6))

    c = np.zeros(n)
    if mode == 'cost':
        c[GIMP] = c_imp * dt_h
        c[GEXP] = -c_exp * dt_h
        c[PDIS] = bat['c_deg'] * dt_h
    elif mode == 'selfcons':
        c[GIMP] = dt_h
        c[GEXP] = 1e-6                      # tiny tie-breakers keep the solution clean
        c[PDIS] = 1e-6
    else:
        raise ValueError("mode must be 'cost' or 'selfcons'")

    A_eq = lil_matrix((2 * T, n))
    b_eq = np.zeros(2 * T)
    t = np.arange(T)
    # (1) power balance: -p_ch + p_dis + g_imp - g_exp - curt = load - pv
    A_eq[t, PCH] = -1; A_eq[t, PDIS] = 1; A_eq[t, GIMP] = 1; A_eq[t, GEXP] = -1; A_eq[t, CURT] = -1
    b_eq[:T] = load - pv
    # (2) SoC: E_t - E_{t-1} - eta_ch*dt*p_ch + dt/eta_dis*p_dis = 0  (E_{-1} = e_start)
    r = T + t
    A_eq[r, E] = 1
    A_eq[r[1:], E[:-1]] = -1
    A_eq[r, PCH] = -bat['eta_ch'] * dt_h
    A_eq[r, PDIS] = dt_h / bat['eta_dis']
    b_eq[T] = e_start_kwh

    bounds = np.zeros((n, 2))
    bounds[PCH, 1] = bat['p_max_kw'] * k_ch
    bounds[PDIS, 1] = bat['p_max_kw']
    # physical grid limits: import <= load + full charging, export <= PV + full discharging. They never bind when
    # import price > export price, but stop an unbounded buy-and-sell loop when a (forecast) spot price is so negative
    # that the import price falls below the export price (e.g. pre-dispatch at -$1,000/MWh).
    bounds[GIMP, 1] = load + bat['p_max_kw']
    bounds[GEXP, 1] = pv + bat['p_max_kw']
    bounds[CURT, 1] = np.where(allow_curtail & (c_exp < 0), pv, 0.0)
    bounds[E, 0] = bat['soc_min'] * bat['e_cap_kwh']
    bounds[E, 1] = bat['soc_max'] * bat['e_cap_kwh']
    if mode == 'selfcons':
        bounds[PCH, 1] = np.minimum(bounds[PCH, 1], np.clip(pv - load, 0, None))   # charge only from surplus
        bounds[PDIS, 1] = np.minimum(bounds[PDIS, 1], np.clip(load - pv, 0, None)) # never discharge to grid

    A_ub, b_ub = None, None
    if max_dis_kwh is not None and mode == 'cost':
        A_ub = lil_matrix((1, n)); A_ub[0, PDIS] = dt_h
        b_ub = np.array([max_dis_kwh])

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq.tocsr(), b_eq=b_eq, bounds=bounds, method='highs')
    if res.status != 0:
        raise RuntimeError(f'LP not solved: {res.message}')
    x = res.x
    return {'p_ch': x[PCH], 'p_dis': x[PDIS], 'g_imp': x[GIMP], 'g_exp': x[GEXP], 'curt': x[CURT], 'E': x[E],
            'cost': float(res.fun), 'status': res.status}


# ----------------------------------------------------------------------------------------------
# Rolling-horizon operation
# ----------------------------------------------------------------------------------------------
def window(series, t, horizon_h=48):
    """
    What it does: values of an hourly series (index = hour END) for the hours ending t+1h ... t+horizon_h.
    Missing values at the end of the data are filled with the value 24 h earlier (repeat the last day).
    Output: numpy array of length horizon_h.
    """
    idx = pd.date_range(t + pd.Timedelta(hours=1), periods=horizon_h, freq='h')
    v = series.reindex(idx)
    if v.isna().any():
        v = v.fillna(series.reindex(idx - pd.Timedelta(hours=24)).set_axis(idx))
        v = v.ffill().bfill()
    return v.to_numpy(dtype=float)


def run_operation(df5, imp_col, exp_col, bat=BATTERY, mode='cost', allow_curtail=False,
                  pv_plan=None, price_plan=None, replan_h=1, horizon_h=48, use_throughput_cap=True, verbose=False):
    """
    What it does
        Simulates one battery over the period in df5 (Notes D.4.6).
        Every replan_h hours (at hh:00): build a horizon_h-hour hourly plan with solve_bess and take, for the current
        hour, its grid target g_target = g_imp - g_exp and its battery direction (charge / discharge / idle).
        Every 5 minutes: battery power = load - pv - g_target, kept in the planned direction (a charging hour never
        discharges, a discharging hour never charges, an idle hour stays idle), clipped to power and SoC limits with
        the ACTUAL SoC; settle with ACTUAL prices. mode='selfcons' uses g_target = 0
        (the classic self-consumption rule: charge from surplus, discharge to cover deficit).
    Input
        df5        : 5-min DataFrame indexed by datetime (interval END) with load_kw, pv_actual_kw, temperature_c,
                     imp_col, exp_col (actual prices, $/kWh)
        imp_col, exp_col : names of the actual import/export price columns
        bat        : battery dict; mode : 'cost' or 'selfcons'; allow_curtail : True for wholesale
        pv_plan    : None (= perfect foresight: hourly mean of actual PV) or function t -> array(horizon_h) of PV (kW)
        price_plan : None (= perfect foresight: hourly mean of actual prices) or function t -> (imp array, exp array)
        replan_h   : hours between plans (1 = hourly, as in the brief; 24 = daily, faster for the sizing sweep)
    Output
        DataFrame (same index as df5): load_kw, pv_kw, p_bat_kw (+ = discharge), grid_kw (+ = import), import_kw,
        export_kw, curt_kw, soc_kwh, import_price, export_price, cost_aud.
    """
    dt5 = 5 / 60
    hourly = df5[['load_kw', 'pv_actual_kw', 'temperature_c', imp_col, exp_col]].resample('h', label='right', closed='right').mean()
    e_min, e_max = bat['soc_min'] * bat['e_cap_kwh'], bat['soc_max'] * bat['e_cap_kwh']
    soc = bat['soc0'] * bat['e_cap_kwh']
    max_dis = bat['max_dis_kwh_day'] * horizon_h / 24 if use_throughput_cap else None

    load = df5['load_kw'].to_numpy(float); pv = df5['pv_actual_kw'].to_numpy(float)
    k_act = charge_derate(df5['temperature_c'].to_numpy(float))
    imp_p = df5[imp_col].to_numpy(float); exp_p = df5[exp_col].to_numpy(float)
    n = len(df5)
    out = {k: np.zeros(n) for k in ['p_bat_kw', 'grid_kw', 'curt_kw', 'soc_kwh']}

    # plan times: hh:00 at the start of each hour of the period (the 5-min rows of hour t+1h are labelled t+5min ... t+60min)
    start = df5.index[0] - pd.Timedelta(minutes=5)
    hour_of_row = ((df5.index - pd.Timedelta(minutes=5)).floor('h'))     # plan time each row belongs to
    plan_times = pd.date_range(start.floor('h'), hour_of_row[-1], freq='h')
    row_pos = pd.Series(np.arange(n), index=df5.index).groupby(hour_of_row).agg(['first', 'last'])

    g_target, neg_price_flags = 0.0, np.zeros(1, bool)
    for i, t in enumerate(plan_times):
        if t not in row_pos.index:
            continue
        step_in_plan = i % replan_h
        if mode == 'cost' and step_in_plan == 0:
            ld = window(hourly['load_kw'], t, horizon_h)
            pvp = window(hourly['pv_actual_kw'], t, horizon_h) if pv_plan is None else pv_plan(t)
            if price_plan is None:
                ci, ce = window(hourly[imp_col], t, horizon_h), window(hourly[exp_col], t, horizon_h)
            else:
                ci, ce = price_plan(t)
            kc = charge_derate(window(hourly['temperature_c'], t, horizon_h))
            plan = solve_bess(ld, pvp, ci, ce, kc, soc, bat, dt_h=1.0, mode='cost',
                              allow_curtail=allow_curtail, max_dis_kwh=max_dis)
            plan_g = plan['g_imp'] - plan['g_exp']
            plan_dir = np.sign(np.round(plan['p_dis'] - plan['p_ch'], 3))   # +1 discharge, -1 charge, 0 idle
            neg_price_flags = ce < 0
        if mode == 'cost':
            g_target = plan_g[step_in_plan]
            direction = plan_dir[step_in_plan]
            curtail_ok = allow_curtail and neg_price_flags[step_in_plan]
        else:
            g_target, direction, curtail_ok = 0.0, None, False

        a, b = row_pos.loc[t, 'first'], row_pos.loc[t, 'last']
        for j in range(a, b + 1):
            p = load[j] - pv[j] - g_target                       # desired battery power (+ = discharge)
            if direction is not None:                            # follow the plan's direction for this hour:
                p = max(p, 0.0) if direction > 0 else (min(p, 0.0) if direction < 0 else 0.0)
            if p > 0:
                p = min(p, bat['p_max_kw'], (soc - e_min) * bat['eta_dis'] / dt5)
                soc -= p * dt5 / bat['eta_dis']
            else:
                p = max(p, -bat['p_max_kw'] * k_act[j], -(e_max - soc) / (bat['eta_ch'] * dt5))
                soc += -p * dt5 * bat['eta_ch']
            grid = load[j] - pv[j] - p
            curt = 0.0
            if curtail_ok and grid < 0:                          # planned negative price: do not export
                curt = min(pv[j], -grid)
                grid += curt
            out['p_bat_kw'][j], out['grid_kw'][j], out['curt_kw'][j], out['soc_kwh'][j] = p, grid, curt, soc
        if verbose and i % (24 * 30) == 0:
            print('  ', t)

    res = pd.DataFrame(out, index=df5.index)
    res.insert(0, 'load_kw', load); res.insert(1, 'pv_kw', pv)
    res['import_kw'] = res['grid_kw'].clip(lower=0)
    res['export_kw'] = (-res['grid_kw']).clip(lower=0)
    res['import_price'], res['export_price'] = imp_p, exp_p
    res['cost_aud'] = (res['import_kw'] * imp_p - res['export_kw'] * exp_p) * dt5
    return res


def no_battery(df5, imp_col, exp_col):
    """Same output format as run_operation for PV only (no battery) - scenario S1."""
    dt5 = 5 / 60
    res = pd.DataFrame(index=df5.index)
    res['load_kw'], res['pv_kw'] = df5['load_kw'], df5['pv_actual_kw']
    res['p_bat_kw'], res['curt_kw'], res['soc_kwh'] = 0.0, 0.0, 0.0
    res['grid_kw'] = res['load_kw'] - res['pv_kw']
    res['import_kw'] = res['grid_kw'].clip(lower=0)
    res['export_kw'] = (-res['grid_kw']).clip(lower=0)
    res['import_price'], res['export_price'] = df5[imp_col], df5[exp_col]
    res['cost_aud'] = (res['import_kw'] * res['import_price'] - res['export_kw'] * res['export_price']) * dt5
    return res


def settle(res, bat=BATTERY, fee_aud=0.0):
    """
    What it does: annual metrics from a run_operation / no_battery result (Notes D.4.8).
    Output: dict with bill, import/export energy, export revenue, self-consumption, self-sufficiency,
            equivalent full cycles, curtailed energy and the energy-balance check.
    """
    dt5 = 5 / 60
    pv_used = res['pv_kw'] - res['curt_kw']
    pv_export = np.minimum(res['export_kw'], pv_used)              # export that came from PV (rest = battery)
    pv_kwh = res['pv_kw'].sum() * dt5
    balance = (pv_used + res['import_kw'] + res['p_bat_kw'] - res['load_kw'] - res['export_kw']).abs().max()
    dis_kwh = res['p_bat_kw'].clip(lower=0).sum() * dt5
    return {
        'bill_aud': res['cost_aud'].sum() + fee_aud,
        'energy_cost_aud': res['cost_aud'].sum(),
        'import_kwh': res['import_kw'].sum() * dt5,
        'export_kwh': res['export_kw'].sum() * dt5,
        'export_revenue_aud': (res['export_kw'] * res['export_price']).sum() * dt5,
        'battery_export_kwh': (res['export_kw'] - pv_export).sum() * dt5,
        'curtailed_kwh': res['curt_kw'].sum() * dt5,
        'self_consumption': 1 - (pv_export.sum() * dt5 + res['curt_kw'].sum() * dt5) / pv_kwh,
        'self_sufficiency': 1 - res['import_kw'].sum() * dt5 / (res['load_kw'].sum() * dt5),
        'discharge_kwh': dis_kwh,
        'equivalent_full_cycles': dis_kwh / bat['e_cap_kwh'] if bat['e_cap_kwh'] > 0 else 0.0,
        'max_balance_error_kw': float(balance),
    }
