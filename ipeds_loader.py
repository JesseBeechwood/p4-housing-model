"""
IPEDS Data Loader
Loads the IPEDS Compare Institutions export (2024-25 academic year).
Provides enrollment, retention, tuition, and admissions data for 57 Power 4 schools
that do NOT yet have CDS files uploaded.

Role in the model:
1. Populates Market Rankings with all 57 schools — not just the 3 with CDS files
2. Computes investment scores for every school using IPEDS variables as proxies
3. Shows your boss a complete 57-school ranking instead of just 3 schools
4. As CDS files are uploaded school by school, IPEDS data is automatically
   replaced by the richer CDS data (which has off-campus rate, OOS share, etc.)
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Schools that already have CDS files — skip them in IPEDS load
# (CDS data is richer and takes precedence)
CDS_SCHOOLS = {'UCBerkeley', 'BostonCollege', 'UniversityOfMaryland'}

def load_ipeds(filepath=None):
    """
    Load IPEDS CSV and return a DataFrame in the same format as CDS panel data.
    Missing variables (OOS share, off-campus rate) are estimated from available data
    or left as None with a flag so the model knows the source.
    """
    if filepath is None:
        filepath = Path(__file__).parent / 'ipeds_2024.csv'
    if not Path(filepath).exists():
        return pd.DataFrame()

    filepath = str(filepath)
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)
    else:
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='latin-1')

    # Skip schools already covered by CDS
    df = df[~df['school'].isin(CDS_SCHOOLS)].copy()

    # Flag all rows as IPEDS-sourced
    df['data_source'] = 'IPEDS'
    df['ipeds_only']  = True

    # Compute derived fields where possible
    # off_campus_demand cannot be computed without pct_ug_off_campus
    # We estimate it using a national average off-campus rate by school type
    # Public large (>30k): ~60% off campus | Private small (<15k): ~25% off campus
    # This is clearly labeled as estimated and replaced when CDS is uploaded
    def estimate_off_campus_rate(row):
        ug = row.get('total_undergrad', 0) or 0
        tuit = row.get('tuition_instate', 15000) or 15000
        # Public schools (low tuition) tend to have higher off-campus rates
        # Private schools keep more students on campus
        if tuit < 15000 and ug > 30000:   return 0.62  # large public
        elif tuit < 15000 and ug > 20000: return 0.58  # medium public
        elif tuit < 15000:                return 0.52  # small public
        elif tuit > 50000:                return 0.22  # elite private
        elif tuit > 30000:                return 0.30  # private
        else:                             return 0.45  # mid private

    def estimate_oos_rate(row):
        # Estimate OOS from admission rate — elite schools draw more OOS
        admit = row.get('admission_rate', 0.5) or 0.5
        tuit  = row.get('tuition_instate', 15000) or 15000
        if admit < 0.10:  return 0.72  # very selective = national draw
        elif admit < 0.20: return 0.55
        elif admit < 0.35: return 0.40
        elif admit < 0.50: return 0.30
        elif tuit < 15000: return 0.22  # large public
        else:              return 0.35

    df['pct_ug_off_campus'] = df.apply(estimate_off_campus_rate, axis=1)
    df['pct_oos_ug']        = df.apply(estimate_oos_rate, axis=1)
    df['off_campus_demand'] = (df['total_undergrad'] * df['pct_ug_off_campus']).round(0)
    df['pct_ug_on_campus']  = 1 - df['pct_ug_off_campus']

    # pct_need_met: use actual IPEDS value if available, else estimate
    # (already in the file from financial aid sheet)

    # Mark estimates
    df['off_campus_rate_estimated'] = True
    df['oos_rate_estimated']        = True

    return df


def get_ipeds_school_result(row, all_panel):
    """
    Build a school_results-style dict from an IPEDS row.
    Used to populate Market Rankings for schools without CDS files.
    """
    from regression_engine import compute_investment_score, score_to_signal

    ug   = row.get('total_undergrad')
    ret  = row.get('retention_rate')
    oos  = row.get('pct_oos_ug')
    off  = row.get('pct_ug_off_campus')
    need = row.get('pct_need_met')
    dem  = row.get('off_campus_demand')
    tuit = row.get('tuition_instate')

    # Build a minimal panel-like DataFrame for scoring
    grp = pd.DataFrame([{
        'school':            row['school'],
        'academic_year':     2024,
        'total_undergrad':   ug,
        'retention_rate':    ret,
        'pct_oos_ug':        oos,
        'pct_ug_off_campus': off,
        'pct_need_met':      need,
        'off_campus_demand': dem,
        'tuition_instate':   tuit,
    }])

    score = compute_investment_score(grp, all_panel)
    signal, color = score_to_signal(score)

    return {
        'panel':             grp,
        'forecast':          [],  # no forecast without multi-year history
        'trends':            {},
        'investment_score':  round(score, 3),
        'signal':            signal,
        'signal_color':      color,
        'latest_year':       2024,
        'total_undergrad':   ug,
        'off_campus_demand': dem,
        'pct_ug_off_campus': off,
        'retention_rate':    ret,
        'pct_oos_ug':        oos,
        'tuition_instate':   tuit,
        'avg_aid_package':   row.get('avg_aid_package'),
        'pct_need_met':      need,
        'on_campus_housing': None,
        'off_campus_housing':None,
        'rent_premium':      None,
        'data_source':       'IPEDS',
        'ipeds_only':        True,
        'off_campus_rate_estimated': True,
        'oos_rate_estimated':        True,
    }
