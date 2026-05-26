"""
Zillow Data Loader
ZORI  = actual dollar rents ($/mo), smoothed, back to 2015
ZORDI = renter demand index (rate of change), back to 2020
"""
import pandas as pd
import numpy as np
from pathlib import Path

SCHOOL_METRO_MAP = {
    'UniversityOfMaryland': 'Baltimore, MD',
    'BostonCollege':        'Boston, MA',
    'UCBerkeley':           'San Francisco, CA',
    'Michigan':             'Detroit, MI',
    'OhioState':            'Columbus, OH',
    'PennState':            'State College, PA',
    'Wisconsin':            'Milwaukee, WI',
    'Minnesota':            'Minneapolis, MN',
    'Iowa':                 'Iowa City, IA',
    'Illinois':             'Chicago, IL',
    'Indiana':              'Indianapolis, IN',
    'Purdue':               'Indianapolis, IN',
    'Northwestern':         'Chicago, IL',
    'Nebraska':             'Omaha, NE',
    'Rutgers':              'New York, NY',
    'UCLA':                 'Los Angeles, CA',
    'USC':                  'Los Angeles, CA',
    'Washington':           'Seattle, WA',
    'Oregon':               'Portland, OR',
    'Alabama':              'Tuscaloosa, AL',
    'Auburn':               'Auburn, AL',
    'Georgia':              'Atlanta, GA',
    'Florida':              'Gainesville, FL',
    'Tennessee':            'Knoxville, TN',
    'LSU':                  'Baton Rouge, LA',
    'OleMiss':              'Memphis, TN',
    'MississippiState':     'Memphis, TN',
    'Arkansas':             'Fayetteville, AR',
    'Missouri':             'Kansas City, MO',
    'Kentucky':             'Lexington, KY',
    'SouthCarolina':        'Columbia, SC',
    'Vanderbilt':           'Nashville, TN',
    'Texas':                'Austin, TX',
    'TexasAM':              'Houston, TX',
    'Oklahoma':             'Oklahoma City, OK',
    'TCU':                  'Dallas, TX',
    'Baylor':               'Waco, TX',
    'TexasTech':            'Lubbock, TX',
    'Kansas':               'Kansas City, MO',
    'KansasState':          'Kansas City, MO',
    'IowaState':            'Des Moines, IA',
    'WestVirginia':         'Charleston, WV',
    'Cincinnati':           'Cincinnati, OH',
    'Houston':              'Houston, TX',
    'UCF':                  'Orlando, FL',
    'BYU':                  'Salt Lake City, UT',
    'Utah':                 'Salt Lake City, UT',
    'Colorado':             'Denver, CO',
    'Arizona':              'Phoenix, AZ',
    'ArizonaState':         'Phoenix, AZ',
    'Duke':                 'Durham, NC',
    'UNC':                  'Durham, NC',
    'NCState':              'Raleigh, NC',
    'Virginia':             'Richmond, VA',
    'VirginiaTech':         'Richmond, VA',
    'Clemson':              'Greenville, SC',
    'FloridaState':         'Tallahassee, FL',
    'Miami':                'Miami, FL',
    'GeorgiaTech':          'Atlanta, GA',
    'Louisville':           'Louisville, KY',
    'Pittsburgh':           'Pittsburgh, PA',
    'Syracuse':             'Syracuse, NY',
    'WakeForest':           'Winston, NC',
    'UVA':                  'Charlottesville, VA',
    'NotreDame':            'South Bend, IN',
    'Stanford':             'San Jose, CA',
    'Cal':                  'San Francisco, CA',
    'SMU':                  'Dallas, TX',
    'ColoradoBoulder':      'Denver, CO',
    'OklahomaState':        'Oklahoma City, OK',
    # ── Big Ten ──────────────────────────────────────────────────────────
    'Illinois':             'Chicago, IL',
    'Indiana':              'Indianapolis, IN',
    'IowaHawkeyes':         'Iowa City, IA',
    'Michigan':             'Detroit, MI',
    'MichiganState':        'Detroit, MI',
    'Minnesota':            'Minneapolis, MN',
    'Nebraska':             'Omaha, NE',
    'OhioState':            'Columbus, OH',
    'Oregon':               'Eugene, OR',
    'PennState':            'State College, PA',
    'Purdue':               'Lafayette, IN',
    'Rutgers':              'New York, NY',
    'UCLA':                 'Los Angeles, CA',
    'USC':                  'Los Angeles, CA',
    'Washington':           'Seattle, WA',
    'Wisconsin':            'Madison, WI',
    # ── SEC ──────────────────────────────────────────────────────────────
    'Alabama':          'Tuscaloosa, AL',
    'Arkansas':         'Fayetteville, AR',
    'Auburn':           'Auburn, AL',
    'Florida':          'Gainesville, FL',
    'Georgia':          'Athens, GA',
    'Kentucky':         'Lexington, KY',
    'LSU':              'Baton Rouge, LA',
    'Mississippi':      'Oxford, MS',
    'MississippiState': 'Starkville, MS',
    'Missouri':         'Columbia, MO',
    'Oklahoma':         'Norman, OK',
    'SouthCarolina':    'Columbia, SC',
    'Tennessee':        'Knoxville, TN',
    'Texas':            'Austin, TX',
    'TexasAM':          'College Station, TX',
    'Vanderbilt':       'Nashville, TN',
}

def _annual_avg(series):
    annual = {}
    for col, val in series.items():
        yr = int(str(col)[:4])
        annual.setdefault(yr, []).append(float(val))
    return {yr: np.mean(v) for yr, v in annual.items()}

def _trend_pct(annual_dict):
    yrs  = sorted(annual_dict.keys())
    vals = [annual_dict[y] for y in yrs]
    if len(yrs) < 2: return 0.0
    slope = np.polyfit(range(len(yrs)), vals, 1)[0]
    base  = np.median(vals)  # use median not first value to avoid anomalies
    raw   = (slope / abs(base) * 100) if base != 0 else 0.0
    return round(max(-15.0, min(25.0, raw)), 2)  # cap at realistic range

def _yoy(annual_dict):
    yrs  = sorted(annual_dict.keys())
    vals = [annual_dict[y] for y in yrs]
    return {yrs[i]: (vals[i]-vals[i-1])/vals[i-1]*100 for i in range(1, len(yrs))}

def load_zillow(folder=None):
    if folder is None:
        folder = Path(__file__).parent
    folder = Path(folder)
    zori_df  = pd.read_csv(folder/'zillow_zori.csv')  if (folder/'zillow_zori.csv').exists()  else None
    zordi_df = pd.read_csv(folder/'zillow_zordi.csv') if (folder/'zillow_zordi.csv').exists() else None
    if zori_df is None and zordi_df is None:
        return {}

    results = {}
    for school, metro in SCHOOL_METRO_MAP.items():
        entry = {'metro': metro}

        # ZORI: dollar rents
        if zori_df is not None:
            row = zori_df[zori_df['RegionName']==metro]
            if not row.empty:
                dcols = [c for c in zori_df.columns if str(c)[0].isdigit()]
                vals  = row[dcols].iloc[0].dropna()
                annual = _annual_avg(vals)
                entry.update({
                    'latest_rent':     float(vals.iloc[-1]),
                    'latest_date':     str(vals.index[-1]),
                    'rent_annual_avg': annual,
                    'rent_trend_pct':  _trend_pct(annual),
                    'rent_yoy':        _yoy(annual),
                })
            else:
                entry['latest_rent'] = None

        # ZORDI: demand index
        if zordi_df is not None:
            row = zordi_df[zordi_df['RegionName']==metro]
            if not row.empty:
                dcols = [c for c in zordi_df.columns if str(c)[0].isdigit()]
                vals  = row[dcols].iloc[0].dropna()
                recent_12  = float(vals[-12:].mean())
                prior_12   = float(vals[-24:-12].mean()) if len(vals)>=24 else recent_12
                accelerating = recent_12 > prior_12
                latest_zordi = float(vals.iloc[-1])
                entry.update({
                    'latest_zordi':  latest_zordi,
                    'accelerating':  accelerating,
                    'above_baseline':latest_zordi > 0,
                    'momentum_label':'Strengthening ↑' if accelerating else 'Cooling ↓',
                    'recent_12_avg': recent_12,
                    'prior_12_avg':  prior_12,
                })
            else:
                entry['accelerating'] = None

        if entry.get('latest_rent') is not None or entry.get('latest_zordi') is not None:
            results[school] = entry

    return results

def score_component(school, zillow_data):
    """
    0-1 rent market health score, normalized across the panel.
    Formula: 70% long-run trend (%/yr since 2015) + 30% recent 3yr avg YoY.
    Both components normalized against min/max across all schools in zillow_data.
    This replaces the old bucket system (above_baseline/accelerating) which gave
    16 of 18 schools identical scores of 0.60.
    """
    if not zillow_data or school not in zillow_data:
        return None

    def _school_metrics(d):
        trend = d.get('rent_trend_pct') or 0
        yoy = d.get('rent_yoy') or {}
        recent = [yoy.get(y, 0) for y in [2023, 2024, 2025] if yoy.get(y) is not None]
        recent_avg = float(np.mean(recent)) if recent else trend
        return trend, recent_avg

    # Collect panel-wide values for normalization
    all_trends, all_recents = [], []
    for d in zillow_data.values():
        t, r = _school_metrics(d)
        all_trends.append(t)
        all_recents.append(r)

    t_min, t_max = min(all_trends), max(all_trends)
    r_min, r_max = min(all_recents), max(all_recents)

    t, r = _school_metrics(zillow_data[school])
    ts = (t - t_min) / (t_max - t_min) if t_max > t_min else 0.5
    rs = (r - r_min) / (r_max - r_min) if r_max > r_min else 0.5

    return round(0.7 * ts + 0.3 * rs, 3)
