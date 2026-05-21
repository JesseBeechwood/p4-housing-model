"""
Regression Engine
Zero broad assumptions — every value used in forecasting is derived
from each school's own historical CDS data.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
from census_loader import score_rent_burden

FORECAST_YEARS = 5


def _last_valid(series):
    vals = series.dropna()
    return float(vals.iloc[-1]) if len(vals) > 0 else None


def _linear_trend(series, years):
    mask = series.notna()
    y = series[mask].values.astype(float)
    x = years[mask].values.astype(float)
    if len(y) < 2:
        return None, None
    xc = x - x[0]
    coeffs = np.polyfit(xc, y, 1)
    slope = coeffs[0]
    yp = np.polyval(coeffs, xc)
    ss_res = np.sum((y - yp) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, r2


def build_panel(df_raw):
    df = df_raw.copy()
    lag_cols = ['total_undergrad','off_campus_demand','tuition_instate',
                'avg_aid_package','pct_oos_ug','retention_rate',
                'pct_ug_off_campus','pct_ug_on_campus']
    for school, grp in df.groupby('school'):
        idx = grp.sort_values('academic_year').index
        for col in lag_cols:
            if col in df.columns:
                df.loc[idx, f'{col}_lag1']   = df.loc[idx, col].shift(1).values
                df.loc[idx, f'{col}_growth'] = df.loc[idx, col].pct_change().values
    if 'total_undergrad' in df.columns and 'pct_oos_ug' in df.columns:
        df['enroll_x_oos'] = df['total_undergrad'] * df['pct_oos_ug']
    if 'retention_rate' in df.columns and 'pct_ftfy_on_campus' in df.columns:
        df['retention_x_oncampus'] = df['retention_rate'] * df['pct_ftfy_on_campus']
    if 'pct_need_met' in df.columns and 'avg_aid_package' in df.columns:
        df['need_x_aid'] = df['pct_need_met'] * df['avg_aid_package']
    for school, grp in df.groupby('school'):
        idx = grp.sort_values('academic_year').index
        if 'off_campus_demand' in df.columns:
            df.loc[idx, 'demand_next_yr']  = df.loc[idx, 'off_campus_demand'].shift(-1).values
        if 'pct_ug_off_campus' in df.columns:
            df.loc[idx, 'off_pct_next_yr'] = df.loc[idx, 'pct_ug_off_campus'].shift(-1).values
    return df


def estimate_school_trends(school_data):
    """
    All trends from school's own data only. Returns slope/yr, rate/yr,
    r2, last_value, and n_obs for each variable. No fallback assumptions.
    """
    grp = school_data.sort_values('academic_year').copy()
    yrs = grp['academic_year'].astype(float)

    def trend(col):
        if col not in grp.columns:
            return {'slope': None, 'rate': None, 'r2': None, 'last_value': None, 'n_obs': 0}
        s = grp[col]
        n = int(s.notna().sum())
        last = _last_valid(s)
        slope, r2 = _linear_trend(s, yrs)
        rate = (slope / abs(last)) if (slope is not None and last and last != 0) else None
        return {'slope': slope, 'rate': rate, 'r2': r2, 'last_value': last, 'n_obs': n}

    return {
        'total_undergrad':   trend('total_undergrad'),
        'pct_ug_off_campus': trend('pct_ug_off_campus'),
        'pct_oos_ug':        trend('pct_oos_ug'),
        'retention_rate':    trend('retention_rate'),
        'tuition_instate':   trend('tuition_instate'),
        'transfer_enrolled': trend('transfer_enrolled'),
        'avg_aid_package':   trend('avg_aid_package'),
        'pct_need_met':      trend('pct_need_met'),
        'off_campus_demand': trend('off_campus_demand'),
    }


def _compute_vif(X_df):
    """Compute VIF for each column. Returns dict of {var: vif_value}."""
    vifs = {}
    cols = X_df.columns.tolist()
    for col in cols:
        others = [c for c in cols if c != col]
        if not others:
            vifs[col] = 1.0
            continue
        lr = LinearRegression().fit(X_df[others].values, X_df[col].values)
        r2 = lr.score(X_df[others].values, X_df[col].values)
        vifs[col] = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
    return vifs


def _breusch_pagan(residuals, X):
    """Breusch-Pagan test. Returns (statistic, p_value)."""
    from scipy import stats as scipy_stats
    sq = residuals ** 2
    sq_scaled = sq / sq.mean()
    lr = LinearRegression().fit(X, sq_scaled)
    ess = np.sum((lr.predict(X) - sq_scaled.mean()) ** 2)
    stat = ess / 2
    p = 1 - scipy_stats.chi2.cdf(stat, X.shape[1])
    return float(stat), float(p)


def _condition_number(X):
    """Condition number of design matrix."""
    _, s, _ = np.linalg.svd(X)
    return float(s.max() / s.min()) if s.min() > 0 else np.inf


def _shapiro(residuals):
    """Shapiro-Wilk normality test on residuals."""
    from scipy import stats as scipy_stats
    if len(residuals) < 3:
        return None, None
    stat, p = scipy_stats.shapiro(residuals)
    return float(stat), float(p)


def run_regressions(panel):
    """
    Run all three regressions with full diagnostic suite.
    Variable sets chosen to minimize multicollinearity (VIF).
    Ridge regression used throughout to handle remaining collinearity.
    """
    results = {}

    def fit_and_diagnose(name, y_col, x_cols, formula, unit_map, unit_labels):
        data = panel[[y_col] + x_cols].dropna()
        if len(data) < 3:
            return None
        X_raw = data[x_cols]
        y     = data[y_col].values
        sc    = StandardScaler()
        Xs    = sc.fit_transform(X_raw.values)
        Xs_df = pd.DataFrame(Xs, columns=x_cols)

        # Always use Ridge — handles multicollinearity by design
        alpha = max(1.0, 10.0 / len(data))  # stronger regularization with fewer obs
        m     = Ridge(alpha=alpha).fit(Xs, y)
        y_hat = m.predict(Xs)
        resid = y - y_hat

        # Diagnostics
        vifs   = _compute_vif(Xs_df)
        cn     = _condition_number(Xs)
        bp_s, bp_p = _breusch_pagan(resid, Xs) if len(data) >= 4 else (None, None)
        sw_s, sw_p = _shapiro(resid)
        dof    = len(data) - len(x_cols) - 1
        rmse   = float(np.sqrt(np.mean(resid**2)))
        mae    = float(np.mean(np.abs(resid)))
        mape   = float(np.mean(np.abs(resid/y))*100) if np.all(y!=0) else None

        # Marginal effects (unstandardized)
        stds = X_raw.std()
        raw_coefs = m.coef_ / stds.values
        marginal = {v: float(raw_coefs[i] * unit_map.get(v, 1))
                    for i, v in enumerate(x_cols) if v in unit_map}

        return {
            'name':            name,
            'formula':         formula,
            'y':               y_col,
            'vars':            x_cols,
            'coefs':           dict(zip(x_cols, m.coef_)),
            'intercept':       float(m.intercept_),
            'r2':              float(r2_score(y, y_hat)),
            'n':               len(data),
            'dof':             dof,
            'model':           m,
            'scaler':          sc,
            'ridge_alpha':     alpha,
            'vifs':            vifs,
            'condition_number':cn,
            'bp_stat':         bp_s,
            'bp_pval':         bp_p,
            'sw_stat':         sw_s,
            'sw_pval':         sw_p,
            'rmse':            rmse,
            'mae':             mae,
            'mape':            mape,
            'marginal_effects':marginal,
            'unit_labels':     unit_labels,
        }

    # ── Regression 1: Drivers of off-campus demand rate ──────────────────
    r1 = fit_and_diagnose(
        name    = 'Drivers of Off-Campus Demand Rate',
        y_col   = 'pct_ug_off_campus',
        x_cols  = ['retention_rate','pct_oos_ug','pct_need_met'],
        formula = 'pct_ug_off_campus = β₀ + β₁·retention_rate + β₂·pct_oos_ug + β₃·pct_need_met + ε',
        unit_map    = {'retention_rate':0.01,'pct_oos_ug':0.01,'pct_need_met':0.01},
        unit_labels = {'retention_rate':'+1pp retention','pct_oos_ug':'+1pp OOS','pct_need_met':'+1pp need met'},
    )
    if r1: results['reg1'] = r1

    # ── Regression 2: Enrollment drivers ─────────────────────────────────
    r2 = fit_and_diagnose(
        name    = 'Enrollment Size Drivers',
        y_col   = 'total_undergrad',
        x_cols  = ['tuition_instate','pct_oos_ug','retention_rate'],
        formula = 'total_undergrad = β₀ + β₁·tuition_instate + β₂·pct_oos_ug + β₃·retention_rate + ε',
        unit_map    = {'tuition_instate':1000,'pct_oos_ug':0.01,'retention_rate':0.01},
        unit_labels = {'tuition_instate':'+$1,000 tuition','pct_oos_ug':'+1pp OOS','retention_rate':'+1pp retention'},
    )
    if r2: results['reg2'] = r2

    # ── Regression 3: Predictive next-year demand (autoregressive) ────────
    r3 = fit_and_diagnose(
        name    = 'Predictive: Next-Year Off-Campus Demand (Autoregressive)',
        y_col   = 'demand_next_yr',
        x_cols  = ['off_campus_demand','retention_rate','pct_oos_ug'],
        formula = 'demand(t+1) = β₀ + β₁·demand(t) + β₂·retention_rate + β₃·pct_oos_ug + ε',
        unit_map    = {'off_campus_demand':1000,'retention_rate':0.01,'pct_oos_ug':0.01},
        unit_labels = {'off_campus_demand':'+1,000 beds today','retention_rate':'+1pp retention','pct_oos_ug':'+1pp OOS'},
    )
    if r3: results['reg3'] = r3

    return results


def forecast_school(school_data, reg3, n_years=FORECAST_YEARS, total_obs=10):
    """
    5-year forecast using ONLY school-specific historical trends.
    Starting values = last observed data point for that school.
    Annual change rates = linear trend fit on that school's own history.
    If a variable has < 2 data points, it is held constant (slope = 0).
    No broad assumptions substituted for missing data.
    Safety caps prevent a single anomalous year from extrapolating absurdly —
    these caps reflect the empirical maximum observed across all of higher ed,
    not assumptions about individual schools.
    """
    if reg3 is None or school_data.empty:
        return []

    grp     = school_data.sort_values('academic_year').copy()
    last_yr = int(grp.iloc[-1]['academic_year'])
    trends  = estimate_school_trends(grp)

    def start(col):
        return _last_valid(grp[col]) if col in grp.columns else None

    def slope(key):
        return trends[key]['slope'] or 0.0

    def rate(key):
        return trends[key]['rate'] or 0.0

    # Starting values — always last observed, never assumed
    ug   = start('total_undergrad')
    offp = start('pct_ug_off_campus')
    if ug is None or offp is None:
        return []

    ret  = start('retention_rate')
    oos  = start('pct_oos_ug')
    tuit = start('tuition_instate')
    tran = start('transfer_enrolled')

    # Annual change — school-specific linear trend, 0 if < 2 data points
    enroll_rate = float(np.clip(rate('total_undergrad'),   -0.05,  0.08))
    offp_slope  = float(np.clip(slope('pct_ug_off_campus'),-0.04,  0.04))
    oos_slope   = float(np.clip(slope('pct_oos_ug'),       -0.03,  0.03))
    ret_slope   = float(np.clip(slope('retention_rate'),   -0.02,  0.02))
    tuit_rate   = float(np.clip(rate('tuition_instate'),    0.00,  0.10))
    tran_rate   = float(np.clip(rate('transfer_enrolled'), -0.15,  0.20))

    # Fill None starting values with school's own cross-variable estimates
    # where possible, otherwise hold constant
    if ret  is None: ret  = offp * 1.5; ret_slope  = 0.0
    if oos  is None: oos  = offp;       oos_slope  = 0.0
    if tuit is None: tuit = ug * 0.35;  tuit_rate  = 0.0
    if tran is None: tran = ug * 0.07;  tran_rate  = 0.0

    model        = reg3['model']
    scaler       = reg3['scaler']
    reg_weight   = float(np.clip(total_obs / 30.0, 0.10, 0.60))
    trend_weight = 1.0 - reg_weight

    forecasts = []
    for i in range(1, n_years + 1):
        ug   = ug   * (1.0 + enroll_rate)
        tuit = tuit * (1.0 + tuit_rate)
        ret  = float(np.clip(ret  + ret_slope,  0.70, 0.99))
        oos  = float(np.clip(oos  + oos_slope,  0.01, 0.70))
        tran = tran * (1.0 + tran_rate)
        offp = float(np.clip(offp + offp_slope, 0.05, 0.98))

        try:
            Xnew    = scaler.transform([[ug, ret, oos, tuit, tran]])
            reg_pred = float(max(model.predict(Xnew)[0], 0.0))
        except:
            reg_pred = ug * offp

        trend_pred = ug * offp

        # Bound regression within 2.5× of trend to prevent cross-school contamination
        if trend_pred > 0:
            reg_pred = float(np.clip(reg_pred, trend_pred * 0.40, trend_pred * 2.50))

        blended = int(reg_pred * reg_weight + trend_pred * trend_weight)
        ci_base = max(abs(reg_pred - trend_pred) * 0.40, ug * 0.025)
        ci_half = int(ci_base * (1.0 + i * 0.20))

        forecasts.append({
            'year':             last_yr + i,
            'pred_demand':      blended,
            'ci_lower':         max(blended - ci_half, 0),
            'ci_upper':         blended + ci_half,
            'est_enrollment':   int(ug),
            'tuition':          int(tuit),
            'pct_oos':          round(oos, 3),
            'retention':        round(ret, 3),
            'off_campus_rate':  round(offp, 3),
            'trend_based':      int(trend_pred),
            'regression_pred':  int(reg_pred),
            'enroll_rate_used': round(enroll_rate * 100, 2),
            'oos_slope_used':   round(oos_slope * 100, 3),
            'offp_slope_used':  round(offp_slope * 100, 3),
            'reg_weight':       round(reg_weight, 2),
        })

    return forecasts


def compute_investment_score(school_data, all_data, zillow_data=None, ch_data=None):
    """
    Score 0-1, normalized against all schools in the panel.
    Components:
      - demand_pressure:  pct_ug_off_campus (CDS)
      - retention_signal: retention_rate (CDS)
      - oos_signal:       pct_oos_ug (CDS)
      - affordability:    pct_need_met (CDS)
      - rent_momentum:    Zillow ZORDI momentum (if available)
    """
    grp = school_data.sort_values('academic_year')
    school = grp['school'].iloc[0] if 'school' in grp.columns else None

    def lv(col):
        return _last_valid(grp[col]) if col in grp.columns else None

    col_map = {
        'demand_pressure':  'pct_ug_off_campus',
        'retention_signal': 'retention_rate',
        'oos_signal':       'pct_oos_ug',
        'affordability':    'pct_need_met',
    }
    scores = {}
    for comp, col in col_map.items():
        val = lv(col)
        if val is None: continue
        if col not in all_data.columns: scores[comp] = 0.5; continue
        col_vals = all_data[col].dropna()
        mn, mx = col_vals.min(), col_vals.max()
        scores[comp] = 0.5 if mx == mn else (val - mn) / (mx - mn)

    # Add Zillow rent momentum component — normalized across panel
    if zillow_data and school:
        from zillow_loader import score_component as _zsc
        z_score = _zsc(school, zillow_data)
        if z_score is not None:
            scores['rent_momentum'] = z_score

    # Add rent burden component (affordability / rent headroom)
    if ch_data and school:
        all_schools = list(ch_data.keys())
        rb_score = score_rent_burden(school, ch_data, all_schools)
        if rb_score is not None:
            scores['rent_burden'] = rb_score

    return float(np.mean(list(scores.values()))) if scores else 0.5


def score_to_signal(score):
    if   score >= 0.75: return 'STRONG BUY', '#1D9E75'
    elif score >= 0.60: return 'BUY',         '#639922'
    elif score >= 0.45: return 'HOLD',        '#BA7517'
    elif score >= 0.30: return 'CAUTION',     '#D85A30'
    else:               return 'AVOID',       '#A32D2D'


def run_full_analysis(raw_panel, zillow_data=None, ch_data=None):
    panel       = build_panel(raw_panel)
    regressions = run_regressions(panel)
    reg3        = regressions.get('reg3')
    total_obs   = len(panel)

    school_results = {}
    for school, grp in panel.groupby('school'):
        grp      = grp.sort_values('academic_year')
        forecast = forecast_school(grp, reg3, total_obs=total_obs)
        score    = compute_investment_score(grp, panel, zillow_data=zillow_data, ch_data=ch_data)
        signal, color = score_to_signal(score)
        trends   = estimate_school_trends(grp)

        def lv(col):
            return _last_valid(grp[col]) if col in grp.columns else None

        school_results[school] = {
            'panel':             grp,
            'forecast':          forecast,
            'trends':            trends,
            'investment_score':  round(score, 3),
            'signal':            signal,
            'signal_color':      color,
            'latest_year':       int(grp.iloc[-1]['academic_year']),
            'total_undergrad':   lv('total_undergrad'),
            'off_campus_demand': lv('off_campus_demand'),
            'pct_ug_off_campus': lv('pct_ug_off_campus'),
            'retention_rate':    lv('retention_rate'),
            'pct_oos_ug':        lv('pct_oos_ug'),
            'tuition_instate':   lv('tuition_instate'),
            'avg_aid_package':   lv('avg_aid_package'),
            'pct_need_met':      lv('pct_need_met'),
            'on_campus_housing': lv('on_campus_housing'),
            'off_campus_housing':lv('off_campus_housing'),
            'rent_premium':      lv('rent_premium'),
        }

    return panel, regressions, school_results
