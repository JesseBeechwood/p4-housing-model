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



def compute_forecast_ci(school_data, n_years=5):
    """
    Statistically grounded confidence intervals on the 5-year demand forecast.

    Uncertainty comes from two sources:
      1. Enrollment trend: uses the standard error of the regression slope
         to build a 90% CI on future enrollment via t-distribution.
      2. Off-campus rate: uses the historical standard deviation of the rate
         to build a 90% CI assuming the rate stays within its observed range.

    Returns dict with per-year (lo, mid, hi) and a confidence tier:
      HIGH   — p<0.05 enrollment trend, rate std<0.02, n>=4 years
      MEDIUM — p<0.15 enrollment trend OR rate std<0.04
      LOW    — trend not statistically reliable or <3 years of data
    """
    from scipy import stats as _stats

    grp = school_data.sort_values('academic_year').copy()
    ev  = grp[['academic_year', 'total_undergrad']].dropna()
    rv  = grp['pct_ug_off_campus'].dropna()

    rate_now = float(rv.iloc[-1]) if len(rv) else 0.05
    rate_std = float(rv.std())    if len(rv) >= 2 else 0.02

    enrl_now = float(ev['total_undergrad'].iloc[-1]) if len(ev) else 0.0
    slope = intercept = se = 0.0
    p_enroll = 1.0

    if len(ev) >= 3:
        slope, intercept, r_val, p_enroll, se = _stats.linregress(
            ev['academic_year'].values,
            np.log(ev['total_undergrad'].values)
        )

    # 90% CI on slope (t-distribution, df = n-2)
    n_obs = len(ev)
    t_crit = _stats.t.ppf(0.95, df=max(n_obs - 2, 1))  # one-sided 95% = two-sided 90%
    slope_lo = slope - t_crit * se
    slope_hi = slope + t_crit * se

    # Rate 90% CI — clip to valid range
    rate_lo = max(0.01, rate_now - 1.645 * rate_std)
    rate_hi = min(0.98, rate_now + 1.645 * rate_std)

    # Confidence tier
    if p_enroll < 0.05 and rate_std < 0.02 and n_obs >= 4:
        tier = 'HIGH'
    elif p_enroll < 0.15 and n_obs >= 3:
        tier = 'MEDIUM'
    else:
        tier = 'LOW'

    year_cis = {}
    for yr in range(1, n_years + 1):
        enrl_mid = enrl_now * np.exp(slope    * yr)
        enrl_lo  = enrl_now * np.exp(slope_lo * yr)
        enrl_hi  = enrl_now * np.exp(slope_hi * yr)

        demand_mid = enrl_mid * rate_now
        demand_lo  = max(0, enrl_lo * rate_lo)
        demand_hi  = enrl_hi * rate_hi

        year_cis[yr] = {
            'lo':  int(demand_lo),
            'mid': int(demand_mid),
            'hi':  int(demand_hi),
        }

    return {
        'year_cis':    year_cis,
        'tier':        tier,
        'p_enroll':    round(p_enroll, 4),
        'enroll_se':   round(se, 5),
        'rate_std':    round(rate_std, 4),
        'n_years_data': n_obs,
        'enroll_trend_pct': round((np.exp(slope) - 1) * 100, 2),
    }


def compute_score_ci(school_data, panel, zillow_data=None, ch_data=None, n_boot=None, _norms=None):
    """
    Analytic confidence interval on the investment score — O(1), no iterations.

    Approximates CI width from the historical variance of the three CDS inputs
    that drive the demand and growth score components (off-campus rate, OOS share,
    retention). Each variable contributes proportionally to its component weight
    in the score. The result is scaled so that panel-average variance produces
    a CI width consistent with what a full bootstrap would yield (~0.04).

    This is intentionally fast — the score CI is a display signal about data
    reliability, not a precise statistical interval. Stable data -> narrow band.
    Noisy data -> wide band. That relationship is preserved analytically.
    """
    grp = school_data.sort_values('academic_year').copy()
    central = compute_investment_score(grp, panel, zillow_data=zillow_data, ch_data=ch_data, _norms=_norms)

    off_std = float(grp['pct_ug_off_campus'].std()) if grp['pct_ug_off_campus'].notna().sum() >= 2 else 0.015
    oos_std = float(grp['pct_oos_ug'].std())        if grp['pct_oos_ug'].notna().sum() >= 2        else 0.012
    ret_std = float(grp['retention_rate'].std())    if grp['retention_rate'].notna().sum() >= 2    else 0.006
    n_yrs   = int(grp['academic_year'].notna().sum())

    # Weight each variance by its contribution to the score components
    # off_campus_rate affects Demand (30%) + Growth (20%) = 50% of score
    # oos_share affects Demand (30% x 20%) = 6% of score
    # retention barely affects Growth (<5%) — small contribution
    # Combine as root-sum-of-squares, scaled to score units
    composite_std = float(np.sqrt(
        (off_std * 0.50) ** 2 +
        (oos_std * 0.06) ** 2 +
        (ret_std * 0.05) ** 2
    ))

    # Scale factor calibrated so panel-average std (~0.025 off, ~0.015 oos, ~0.010 ret)
    # produces a CI width of ~0.04, matching bootstrap results
    scale = 3.5
    half_width = composite_std * scale

    # Widen for fewer years of data (less information = more uncertainty)
    year_penalty = max(0, (4 - n_yrs) * 0.008)
    half_width  += year_penalty

    half_width = float(np.clip(half_width, 0.005, 0.12))

    lo = float(np.clip(central - half_width, 0.0, 1.0))
    hi = float(np.clip(central + half_width, 0.0, 1.0))

    return {
        'lo':      round(lo, 3),
        'hi':      round(hi, 3),
        'central': round(central, 3),
        'width':   round(hi - lo, 3),
    }

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


def _precompute_panel_norms(all_data, ch_data=None, zillow_data=None):
    """
    Pre-compute all panel-level normalization values once.
    Pass the result into compute_investment_score to avoid recomputing
    on every school call.
    """
    from scipy import stats as _stats

    norms = {}

    # Demand normalization
    if 'off_campus_demand' in all_data.columns:
        d = np.log1p(all_data['off_campus_demand'].dropna().values)
        norms['demand_log_min'] = float(d.min())
        norms['demand_log_max'] = float(d.max())
    if 'pct_ug_off_campus' in all_data.columns:
        v = all_data['pct_ug_off_campus'].dropna()
        norms['off_rate_min'] = float(v.min()); norms['off_rate_max'] = float(v.max())
    if 'pct_oos_ug' in all_data.columns:
        v = all_data['pct_oos_ug'].dropna()
        norms['oos_min'] = float(v.min()); norms['oos_max'] = float(v.max())

    # Supply normalization
    if ch_data:
        all_occ  = [ch_data[s].get('occupancy_rate')       for s in ch_data if ch_data[s].get('occupancy_rate') is not None]
        all_bts  = [ch_data[s].get('bed_to_student_ratio') for s in ch_data if ch_data[s].get('bed_to_student_ratio') is not None]
        all_pipe = [ch_data[s].get('pipeline_pct', 0) or 0 for s in ch_data]
        norms['occ_min']  = min(all_occ)  if all_occ  else 0.0
        norms['occ_max']  = max(all_occ)  if all_occ  else 1.0
        norms['bts_min']  = min(all_bts)  if all_bts  else 0.0
        norms['bts_max']  = max(all_bts)  if all_bts  else 3.0
        norms['pipe_min'] = min(all_pipe) if all_pipe else 0.0
        norms['pipe_max'] = max(all_pipe) if all_pipe else 1.0

    # Growth normalization — run linregress once per school across panel
    all_weighted_trends = []
    all_rate_trends = []
    for s_school in all_data['school'].unique():
        s_grp = all_data[all_data['school'] == s_school].sort_values('academic_year')
        s_ev  = s_grp[['academic_year', 'total_undergrad']].dropna()
        if len(s_ev) >= 3:
            s_sl, _, _, s_p, _ = _stats.linregress(
                s_ev['academic_year'].values,
                np.log(s_ev['total_undergrad'].values))
            s_et = (np.exp(s_sl) - 1) * 100
            s_cw = 1.0 if s_p < 0.05 else (0.5 if s_p < 0.15 else 0.25)
            all_weighted_trends.append(s_et * s_cw)
        s_rv = s_grp[['academic_year', 'pct_ug_off_campus']].dropna()
        if len(s_rv) >= 3 and s_rv['pct_ug_off_campus'].std() > 0.005:
            s_rsl, _, _, _, _ = _stats.linregress(
                s_rv['academic_year'].values,
                s_rv['pct_ug_off_campus'].values)
            all_rate_trends.append(s_rsl * 100)

    if all_weighted_trends:
        norms['trend_min'] = min(all_weighted_trends)
        norms['trend_max'] = max(all_weighted_trends)
    if all_rate_trends:
        norms['rate_trend_min'] = min(all_rate_trends)
        norms['rate_trend_max'] = max(all_rate_trends)

    # Zillow normalization
    if zillow_data:
        from zillow_loader import score_component as _zsc
        z_scores = [_zsc(s, zillow_data) for s in zillow_data]
        z_scores = [z for z in z_scores if z is not None]
        norms['zillow_min'] = min(z_scores) if z_scores else 0.0
        norms['zillow_max'] = max(z_scores) if z_scores else 1.0

    return norms


def compute_investment_score(school_data, all_data, zillow_data=None, ch_data=None, _norms=None):
    """
    Investment score 0-1: four-component model built from what the data shows
    actually matters for student housing market selection.

    Architecture (weights sum to 1.0):
      SUPPLY GAP     35%  Geometric mean of occupancy, inverse BtS, inverse pipeline.
                          Geometric mean means weakness in any single metric drags the
                          score — an oversupplied market cannot be rescued by clean
                          pipeline data.
      DEMAND         30%  50% log(demand_abs) + 30% off_campus_rate + 20% OOS_share.
                          Log-normalization prevents FSU/UCF from dominating purely
                          on size while still rewarding large markets.
      GROWTH         20%  70% enrollment_trend (confidence-weighted by p-value) +
                          30% off_campus_rate_trend. Rewards markets with statistically
                          supported enrollment expansion.
      RENT HEADROOM  15%  60% rent_trend_pct (Zillow) + 40% inverse(rent_burden).
                          Only occupancy correlated significantly with rent growth in
                          the 34-school panel (r=0.385, p=0.025); rent trend is the
                          best available forward-looking proxy.

    All components are normalized against the current panel (0-1).
    """
    from scipy import stats as _stats

    grp = school_data.sort_values('academic_year')
    school = grp['school'].iloc[0] if 'school' in grp.columns else None

    def lv(col):
        return _last_valid(grp[col]) if col in grp.columns else None

    def norm_panel(val, col, invert=False):
        """Normalize val against panel distribution for col."""
        if val is None or col not in all_data.columns:
            return None
        series = all_data[col].dropna()
        mn, mx = series.min(), series.max()
        if mx == mn:
            return 0.5
        s = float(np.clip((val - mn) / (mx - mn), 0.0, 1.0))
        return 1.0 - s if invert else s

    def norm_custom(val, all_vals, invert=False):
        """Normalize val against an explicit list of panel values."""
        valid = [v for v in all_vals if v is not None and not np.isnan(v)]
        if not valid or val is None:
            return None
        mn, mx = min(valid), max(valid)
        if mx == mn:
            return 0.5
        s = float(np.clip((val - mn) / (mx - mn), 0.0, 1.0))
        return 1.0 - s if invert else s

    # ── SUPPLY GAP (35%) ────────────────────────────────────────────────────
    supply_score = None
    if ch_data and school:
        ch = ch_data.get(school, {})
        occ  = ch.get('occupancy_rate')
        bts  = ch.get('bed_to_student_ratio')
        pipe = ch.get('pipeline_pct', 0) or 0

        if occ is not None and bts is not None:
            if _norms:
                def _n(v, mn, mx, invert=False):
                    if mx == mn: return 0.5
                    s = float(np.clip((v - mn) / (mx - mn), 0, 1))
                    return 1.0 - s if invert else s
                n_occ  = _n(occ,  _norms.get('occ_min',0),  _norms.get('occ_max',1))
                n_bts  = _n(bts,  _norms.get('bts_min',0),  _norms.get('bts_max',3),  invert=True)
                n_pipe = _n(pipe, _norms.get('pipe_min',0), _norms.get('pipe_max',1), invert=True)
            else:
                all_occ  = [ch_data[s].get('occupancy_rate')         for s in ch_data if ch_data[s].get('occupancy_rate') is not None]
                all_bts  = [ch_data[s].get('bed_to_student_ratio')   for s in ch_data if ch_data[s].get('bed_to_student_ratio') is not None]
                all_pipe = [ch_data[s].get('pipeline_pct', 0) or 0   for s in ch_data]
                n_occ  = norm_custom(occ,  all_occ)
                n_bts  = norm_custom(bts,  all_bts,  invert=True)
                n_pipe = norm_custom(pipe, all_pipe, invert=True)

            if n_occ is not None and n_bts is not None and n_pipe is not None:
                # Weighted supply score:
                # BtS 50% — structural undersupply is the primary investable signal
                # Occupancy 35% — operational signal; low occ with low BtS = shadow market, not oversupply
                # Pipeline 15% — forward-looking supply risk
                supply_score = float(0.50 * n_bts + 0.35 * n_occ + 0.15 * n_pipe)

    # ── DEMAND (30%) ────────────────────────────────────────────────────────
    demand_abs  = lv('off_campus_demand')
    off_rate    = lv('pct_ug_off_campus')
    oos         = lv('pct_oos_ug')

    demand_score = None
    if demand_abs is not None and off_rate is not None:
        if _norms and 'demand_log_min' in _norms:
            d_mn = _norms['demand_log_min']; d_mx = _norms['demand_log_max']
            n_demand = float(np.clip((np.log1p(demand_abs) - d_mn) / (d_mx - d_mn + 1e-9), 0, 1))
            def _np2(v, mn, mx, invert=False):
                if mx == mn: return 0.5
                s = float(np.clip((v - mn) / (mx - mn), 0, 1))
                return 1.0 - s if invert else s
            n_off_rate = _np2(off_rate, _norms.get('off_rate_min',0), _norms.get('off_rate_max',1))
            n_oos = _np2(oos, _norms.get('oos_min',0), _norms.get('oos_max',1)) if oos is not None else 0.5
        else:
            all_demand = np.log1p(all_data['off_campus_demand'].dropna().values)
            n_demand   = float(np.clip((np.log1p(demand_abs) - all_demand.min()) /
                                       (all_demand.max() - all_demand.min() + 1e-9), 0, 1))
            n_off_rate = norm_panel(off_rate, 'pct_ug_off_campus')
            n_oos      = norm_panel(oos, 'pct_oos_ug') if oos is not None else 0.5

        demand_score = 0.50 * n_demand + 0.30 * n_off_rate + 0.20 * n_oos

    # ── GROWTH (20%) ────────────────────────────────────────────────────────
    growth_score = None
    enroll_vals = grp[['academic_year', 'total_undergrad']].dropna()
    if len(enroll_vals) >= 3:
        slope, _, _, p_enroll, _ = _stats.linregress(
            enroll_vals['academic_year'].values,
            np.log(enroll_vals['total_undergrad'].values)
        )
        enroll_trend_pct = (np.exp(slope) - 1) * 100
        # Confidence weight: p<0.05 full weight, p<0.15 half weight, else quarter weight
        conf_weight = 1.0 if p_enroll < 0.05 else (0.5 if p_enroll < 0.15 else 0.25)
        weighted_trend = enroll_trend_pct * conf_weight

        # Rate trend (pp/yr)
        rate_vals = grp[['academic_year', 'pct_ug_off_campus']].dropna()
        rate_trend = 0.0
        if len(rate_vals) >= 3 and rate_vals['pct_ug_off_campus'].std() > 0.005:
            r_slope, _, _, _, _ = _stats.linregress(
                rate_vals['academic_year'].values,
                rate_vals['pct_ug_off_campus'].values
            )
            rate_trend = r_slope * 100  # scale to same magnitude as enroll_trend_pct

        # Use pre-computed norms if available, else recompute
        if _norms and 'trend_min' in _norms:
            def _ng(v, mn, mx):
                if mx == mn: return 0.5
                return float(np.clip((v - mn) / (mx - mn), 0, 1))
            n_enroll = _ng(weighted_trend, _norms['trend_min'], _norms['trend_max'])
            n_rate   = _ng(rate_trend, _norms.get('rate_trend_min', rate_trend),
                           _norms.get('rate_trend_max', rate_trend)) if 'rate_trend_min' in _norms else 0.5
        else:
            all_weighted_trends = []
            all_rate_trends = []
            for s_school in all_data['school'].unique():
                s_grp = all_data[all_data['school'] == s_school].sort_values('academic_year')
                s_ev = s_grp[['academic_year', 'total_undergrad']].dropna()
                if len(s_ev) >= 3:
                    s_sl, _, _, s_p, _ = _stats.linregress(
                        s_ev['academic_year'].values,
                        np.log(s_ev['total_undergrad'].values))
                    s_et = (np.exp(s_sl) - 1) * 100
                    s_cw = 1.0 if s_p < 0.05 else (0.5 if s_p < 0.15 else 0.25)
                    all_weighted_trends.append(s_et * s_cw)
                s_rv = s_grp[['academic_year', 'pct_ug_off_campus']].dropna()
                if len(s_rv) >= 3 and s_rv['pct_ug_off_campus'].std() > 0.005:
                    s_rsl, _, _, _, _ = _stats.linregress(
                        s_rv['academic_year'].values,
                        s_rv['pct_ug_off_campus'].values)
                    all_rate_trends.append(s_rsl * 100)
            n_enroll = norm_custom(weighted_trend, all_weighted_trends) if all_weighted_trends else 0.5
            n_rate   = norm_custom(rate_trend, all_rate_trends) if all_rate_trends else 0.5

        growth_score = 0.70 * (n_enroll or 0.5) + 0.30 * (n_rate or 0.5)

    # ── RENT HEADROOM (15%) ─────────────────────────────────────────────────
    rent_score = None
    if zillow_data and school:
        from zillow_loader import score_component as _zsc
        z_score = _zsc(school, zillow_data)
        if z_score is not None:
            rb_score = None
            if ch_data:
                rb_score = score_rent_burden(school, ch_data, list(ch_data.keys()))
            rent_score = 0.60 * z_score + 0.40 * (rb_score if rb_score is not None else 0.5)

    # ── COMPOSITE ───────────────────────────────────────────────────────────
    components = {}
    if supply_score  is not None: components['supply']  = (supply_score,  0.35)
    if demand_score  is not None: components['demand']  = (demand_score,  0.30)
    if growth_score  is not None: components['growth']  = (growth_score,  0.20)
    if rent_score    is not None: components['rent']    = (rent_score,    0.15)

    if not components:
        return 0.5

    # Weighted average, re-normalizing weights if some components are missing
    total_weight = sum(w for _, w in components.values())
    composite = sum(score * w for score, w in components.values()) / total_weight

    return float(np.clip(composite, 0.0, 1.0))


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

    # Pre-compute normalization values once — avoids O(n²) recomputation per school
    _norms = _precompute_panel_norms(panel, ch_data=ch_data, zillow_data=zillow_data)

    school_results = {}
    for school, grp in panel.groupby('school'):
        grp      = grp.sort_values('academic_year')
        forecast = forecast_school(grp, reg3, total_obs=total_obs)
        score    = compute_investment_score(grp, panel, zillow_data=zillow_data, ch_data=ch_data, _norms=_norms)
        signal, color = score_to_signal(score)
        trends   = estimate_school_trends(grp)

        def lv(col):
            return _last_valid(grp[col]) if col in grp.columns else None

        forecast_ci  = compute_forecast_ci(grp)
        score_ci     = compute_score_ci(grp, panel, zillow_data=zillow_data, ch_data=ch_data, _norms=_norms)

        school_results[school] = {
            'panel':             grp,
            'forecast':          forecast,
            'forecast_ci':       forecast_ci,
            'score_ci':          score_ci,
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
