"""
wiche_loader.py
---------------
Loads WICHE Knocking at the College Door (11th edition) high school
graduate projections and exposes per-school trend signals.

Data: state-level HS graduate counts, actual 2009–2024, projected 2025–2041.
Source: knocking.wiche.edu — free public dataset, updated 1/31/2025.

Usage:
    from wiche_loader import load_wiche, get_wiche_signal

    wiche = load_wiche(Path('wiche_projections.xlsx'))
    signal = get_wiche_signal('BYU', wiche)
    # returns dict with trend_pct, tier, years dict, etc.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── School → primary feeder state mapping ─────────────────────────────────
# For flagship state schools the home state is the dominant feeder.
# For schools with heavy OOS draws we use the primary feeder state
# (typically the home state for flagship, or dominant OOS market for private).
SCHOOL_STATE = {
    # ACC
    'BostonCollege':        'MA',
    'Clemson':              'SC',
    'Duke':                 'NC',
    'FloridaState':         'FL',
    'GeorgiaTech':          'GA',
    'Louisville':           'KY',
    'Miami':                'FL',
    'NCState':              'NC',
    'Pittsburgh':           'PA',
    'SMU':                  'TX',
    'Stanford':             'CA',
    'Syracuse':             'NY',
    'UCBerkeley':           'CA',
    'UNC':                  'NC',
    'UniversityOfMaryland': 'MD',
    'UVA':                  'VA',
    'VirginiaTech':         'VA',
    'WakeForest':           'NC',
    # Big 12
    'Arizona':              'AZ',
    'ArizonaState':         'AZ',
    'Baylor':               'TX',
    'BYU':                  'UT',
    'Cincinnati':           'OH',
    'ColoradoBoulder':      'CO',
    'Houston':              'TX',
    'IowaState':            'IA',
    'Kansas':               'KS',
    'KansasState':          'KS',
    'OklahomaState':        'OK',
    'TCU':                  'TX',
    'TexasTech':            'TX',
    'WestVirginia':         'WV',
    'UCF':                  'FL',
    'Utah':                 'UT',
    # Big Ten
    'Illinois':             'IL',
    'Indiana':              'IN',
    'IowaHawkeyes':         'IA',
    'Michigan':             'MI',
    'MichiganState':        'MI',
    'Minnesota':            'MN',
    'Nebraska':             'NE',
    'OhioState':            'OH',
    'Oregon':               'OR',
    'PennState':            'PA',
    'Purdue':               'IN',
    'Rutgers':              'NJ',
    'UCLA':                 'CA',
    'USC':                  'CA',
    'Washington':           'WA',
    'Wisconsin':            'WI',
    # ── SEC ──────────────────────────────────────────────────────────────
    'Alabama':          'AL',
    'Arkansas':         'AR',
    'Auburn':           'AL',
    'Florida':          'FL',
    'Georgia':          'GA',
    'Kentucky':         'KY',
    'LSU':              'LA',
    'Mississippi':      'MS',
    'MississippiState': 'MS',
    'Missouri':         'MO',
    'Oklahoma':         'OK',
    'SouthCarolina':    'SC',
    'Tennessee':        'TN',
    'Texas':            'TX',
    'TexasAM':          'TX',
    'Vanderbilt':       'TN',
}


def load_wiche(filepath=None):
    """
    Load and parse the WICHE projections dataset.
    Returns a dict: {state_abbr: {year: hs_grad_count}}
    covering 2009-2041 for all 50 states + DC.
    """
    if filepath is None:
        filepath = Path(__file__).parent / 'wiche_projections.xlsx'
    filepath = Path(filepath)
    if not filepath.exists():
        return {}

    try:
        df = pd.read_excel(
            filepath,
            sheet_name='Data',
            dtype={'Stabbr': str, 'Students': float}
        )
    except Exception:
        return {}

    # Filter to: HS graduates, grand total, all races, state level
    mask = (
        (df['Grade'] == 'High school graduates') &
        (df['SchoolSector'] == 'Grand Total (public+private)') &
        (df['RaceEthnicity'] == 'Total/any') &
        (df['Stabbr'].str.len() == 2) &
        (~df['Stabbr'].str.startswith('_'))
    )
    filtered = df[mask].copy()
    filtered['year'] = filtered['ClassOf'].str.extract(r'(\d{4})').astype(int)

    result = {}
    for state, grp in filtered.groupby('Stabbr'):
        result[state] = {
            int(row['year']): float(row['Students'])
            for _, row in grp.iterrows()
            if pd.notna(row['Students'])
        }
    return result


def get_wiche_signal(school, wiche_data, horizon=10):
    """
    Compute a WICHE-based demand signal for a school.

    Uses the school's primary feeder state HS graduate projections to
    estimate how the pipeline of potential students will change over
    the next `horizon` years.

    Returns dict with:
        trend_pct   — projected % change in HS grads over horizon years
        tier        — 'GROWING' / 'STABLE' / 'DECLINING' / 'SHARP_DECLINE'
        base_year   — most recent actual data year
        horizon_year — base_year + horizon
        base_grads  — HS grad count at base_year
        horizon_grads — projected count at horizon_year
        state       — feeder state abbreviation
        available   — bool, whether data was found
    """
    state = SCHOOL_STATE.get(school)
    if not state or not wiche_data or state not in wiche_data:
        return {'available': False, 'state': state, 'tier': 'UNKNOWN', 'trend_pct': 0.0}

    state_data = wiche_data[state]
    years = sorted(state_data.keys())

    # Base: most recent actual year (2024 or latest available)
    actual_years = [y for y in years if y <= 2024]
    proj_years   = [y for y in years if y > 2024]

    if not actual_years or not proj_years:
        return {'available': False, 'state': state, 'tier': 'UNKNOWN', 'trend_pct': 0.0}

    base_year    = max(actual_years)
    horizon_year = min(base_year + horizon, max(proj_years))
    base_grads   = state_data[base_year]
    horizon_grads = state_data.get(horizon_year)

    if not horizon_grads or base_grads == 0:
        return {'available': False, 'state': state, 'tier': 'UNKNOWN', 'trend_pct': 0.0}

    trend_pct = (horizon_grads / base_grads - 1) * 100

    # Classify
    if trend_pct >= 5.0:
        tier = 'GROWING'
    elif trend_pct >= -5.0:
        tier = 'STABLE'
    elif trend_pct >= -15.0:
        tier = 'DECLINING'
    else:
        tier = 'SHARP_DECLINE'

    return {
        'available':     True,
        'state':         state,
        'tier':          tier,
        'trend_pct':     round(trend_pct, 2),
        'base_year':     base_year,
        'horizon_year':  horizon_year,
        'base_grads':    int(base_grads),
        'horizon_grads': int(horizon_grads),
        'years':         {y: int(state_data[y]) for y in years if y >= 2022},
    }


def score_wiche(school, wiche_data, horizon=10):
    """
    Return a 0-1 normalized WICHE score for use in the investment model.
    Panel-wide normalization is handled externally in regression_engine.
    Returns the raw trend_pct for normalization, or None if unavailable.
    """
    sig = get_wiche_signal(school, wiche_data, horizon=horizon)
    if not sig.get('available'):
        return None
    return sig['trend_pct']
