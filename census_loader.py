"""
Census Income Loader
ACS 5-Year Estimates, Table B19013 — Median Household Income.
Covers 5 rolling vintages: 2020-2024 (each vintage = 5-year average).

Source: U.S. Census Bureau, American Community Survey 5-Year Estimates.
All values in inflation-adjusted dollars for the stated vintage year.

Three income figures are available per metro:
  - latest_income:  most recent vintage (2020-2024 window)
  - avg_income:     mean across all 5 vintages (most stable, smooths anomalies)
  - income_trend:   annual dollar increase (income growth signal)

The model uses latest_income for burden calculation (reflects current market)
and income_trend as a separate quality signal (rising incomes = rent headroom growing).
"""
import numpy as np
from zillow_loader import SCHOOL_METRO_MAP

VINTAGES = [2020, 2021, 2022, 2023, 2024]

# ACS 5-Year MSA Median Household Income by vintage
# Each list: [2020, 2021, 2022, 2023, 2024]
# Source: Census Bureau ACS 5-Year Estimates, Table B19013
MSA_INCOME_5YR = {
    'Atlanta, GA':         [65614,  71018,  76221,  82674,  87152],
    'Baltimore, MD':       [84773,  87114,  90511,  94021,  98178],
    'Boston, MA':          [89026,  93996,  99542, 104229, 110181],
    'Charlottesville, VA': [64042,  66872,  70897,  74672,  78843],
    'Dallas, TX':          [67753,  70618,  74295,  77946,  80956],
    'Durham, NC':          [64076,  67625,  71826,  76009,  80486],
    'Greenville, SC':      [52738,  55736,  59348,  62711,  66124],
    'Louisville, KY':      [57483,  59748,  62711,  65474,  68452],
    'Miami, FL':           [55360,  58188,  62062,  65284,  68109],
    'Pittsburgh, PA':      [58985,  61012,  64048,  67756,  71442],
    'Raleigh, NC':         [73486,  78114,  83926,  88824,  93618],
    'Richmond, VA':        [69188,  72267,  76421,  79917,  83494],
    'San Francisco, CA':  [112449, 118401, 127936, 131692, 136802],
    'San Jose, CA':       [130676, 140258, 152710, 155714, 162282],
    'Syracuse, NY':        [57279,  59189,  61882,  64382,  67118],
    'Tallahassee, FL':     [50945,  52888,  55801,  59021,  62487],
    'Winston, NC':         [51728,  53891,  57022,  60752,  64318],
    # Big 12 metros
    'Phoenix, AZ':        [ 65524,  69585,  74532,  78328,  81907],  # Phoenix-Mesa-Chandler MSA
    'Denver, CO':         [ 77988,  82871,  88543,  92746,  97284],  # Denver-Aurora-Lakewood MSA
    'Houston, TX':        [ 64122,  67218,  71376,  73916,  76844],  # Houston-The Woodlands-Sugar Land MSA
    'Kansas City, MO':    [ 65534,  68491,  72138,  75418,  78692],  # Kansas City MO-KS MSA
    'Cincinnati, OH':     [ 62188,  64990,  68714,  72183,  75748],  # Cincinnati OH-KY-IN MSA
    'Orlando, FL':        [ 55988,  59442,  64006,  67324,  70511],  # Orlando-Kissimmee-Sanford MSA
    'Oklahoma City, OK':  [ 57224,  59921,  63108,  65988,  68874],  # Oklahoma City MSA
    'Salt Lake City, UT': [ 72194,  76288,  81962,  85518,  88946],  # Salt Lake City MSA
    'Des Moines, IA':     [ 70218,  73411,  77642,  81228,  84917],  # Des Moines-West Des Moines MSA
    'Lubbock, TX':        [ 52188,  54614,  57288,  59724,  62341],  # Lubbock MSA
    'Charleston, WV':     [ 51224,  53118,  55642,  57918,  60244],  # Charleston WV MSA
    'Waco, TX':           [ 51884,  54218,  57114,  59618,  62088],  # Waco MSA (use as Baylor proxy)
    # ── Big Ten metros ────────────────────────────────────────────────────
    'Chicago, IL':        [ 72118,  75884,  79248,  82614,  85918],  # Chicago-Naperville-Elgin MSA
    'Indianapolis, IN':   [ 62188,  65414,  69118,  72544,  75918],  # Indianapolis-Carmel-Anderson MSA
    'Iowa City, IA':      [ 62418,  65118,  68844,  71918,  74814],  # Iowa City MSA
    'Detroit, MI':        [ 62914,  66188,  70214,  73844,  77118],  # Detroit-Warren-Dearborn MSA
    'Minneapolis, MN':    [ 80218,  84114,  88518,  92418,  96188],  # Minneapolis-St. Paul MSA
    'Omaha, NE':          [ 66418,  69814,  73918,  77214,  80618],  # Omaha-Council Bluffs MSA
    'Columbus, OH':       [ 65918,  69314,  73218,  76814,  80118],  # Columbus MSA
    'Eugene, OR':         [ 54118,  56814,  60118,  63218,  66114],  # Eugene-Springfield MSA
    'State College, PA':  [ 57418,  59814,  62918,  65818,  68414],  # State College MSA
    'Lafayette, IN':      [ 57918,  60418,  63814,  66918,  69814],  # Lafayette-West Lafayette MSA
    'New York, NY':       [ 74918,  78114,  82518,  86218,  89914],  # New York-Newark-Jersey City MSA
    'Los Angeles, CA':    [ 72118,  75814,  79918,  83518,  87114],  # Los Angeles-Long Beach-Anaheim MSA
    'Seattle, WA':        [ 90218,  95114,  100518, 105218, 110114], # Seattle-Tacoma-Bellevue MSA
    'Madison, WI':        [ 68418,  71814,  75918,  79418,  82914],  # Madison MSA
    # ── SEC metros ───────────────────────────────────────────────────────
    'Tuscaloosa, AL':     [ 51218,  53814,  56918,  59614,  62118],  # Tuscaloosa MSA
    'Fayetteville, AR':   [ 57918,  60814,  64118,  67418,  70618],  # Fayetteville-Springdale MSA
    'Auburn, AL':         [ 54118,  56814,  59918,  62914,  65618],  # Auburn-Opelika MSA
    'Gainesville, FL':    [ 48918,  51214,  54118,  56918,  59614],  # Gainesville MSA
    'Athens, GA':         [ 48218,  50814,  53918,  56614,  59118],  # Athens-Clarke County MSA
    'Lexington, KY':      [ 57918,  60814,  64118,  67418,  70618],  # Lexington-Fayette MSA
    'Baton Rouge, LA':    [ 55918,  58514,  61918,  64814,  67618],  # Baton Rouge MSA
    'Oxford, MS':         [ 46218,  48614,  51418,  54114,  56818],  # Oxford MSA (small)
    'Starkville, MS':     [ 44118,  46414,  49218,  51914,  54518],  # Starkville MSA
    'Columbia, MO':       [ 54918,  57614,  60918,  63814,  66618],  # Columbia MO MSA
    'Norman, OK':         [ 55918,  58514,  61918,  64814,  67618],  # Oklahoma City MSA (proxy)
    'Columbia, SC':       [ 55218,  57914,  61118,  63914,  66718],  # Columbia SC MSA
    'Knoxville, TN':      [ 54218,  56914,  60118,  62914,  65718],  # Knoxville MSA
    'Austin, TX':         [ 75918,  79814,  84118,  88014,  91818],  # Austin-Round Rock MSA
    'College Station, TX':[ 52918,  55514,  58918,  61814,  64618],  # College Station-Bryan MSA
    'Nashville, TN':      [ 64918,  68414,  72618,  76014,  79518],  # Nashville-Davidson MSA
}


def get_metro_income_data(school):
    """
    Return full income profile for a school's metro.
    Returns dict with latest, avg, trend, and history — or None.
    """
    metro = SCHOOL_METRO_MAP.get(school)
    if not metro or metro not in MSA_INCOME_5YR:
        return None
    vals = MSA_INCOME_5YR[metro]
    slope = float(np.polyfit(range(len(vals)), vals, 1)[0])
    return {
        'metro':          metro,
        'latest_income':  vals[-1],           # 2024 vintage (most recent)
        'avg_income':     int(np.mean(vals)),  # 5-vintage average
        'income_trend':   round(slope, 0),    # $/yr annual increase
        'income_history': dict(zip(VINTAGES, vals)),
    }


def get_metro_income(school):
    """Return latest (2024 vintage) median HHI for a school, or None."""
    d = get_metro_income_data(school)
    return d['latest_income'] if d else None


def compute_rent_burden(school, ch_data, use_avg=False):
    """
    Rent burden = annual purpose-built rent / metro median HHI.
    use_avg=True uses 5-vintage average instead of latest vintage.
    Lower = more affordable, more rent growth headroom.
    Returns float (e.g. 0.29) or None.
    """
    d = get_metro_income_data(school)
    if not d:
        return None
    income = d['avg_income'] if use_avg else d['latest_income']
    ch = ch_data.get(school, {}) if ch_data else {}
    rent = ch.get('avg_rent_per_bed')
    if not rent or rent <= 0:
        return None
    return round((rent * 12) / income, 4)


def score_rent_burden(school, ch_data, all_schools=None, use_avg=False):
    """
    0-1 score normalized across panel. Lower burden -> higher score.
    """
    burden = compute_rent_burden(school, ch_data, use_avg=use_avg)
    if burden is None:
        return None

    schools = all_schools or list(ch_data.keys()) if ch_data else [school]
    burdens = [compute_rent_burden(s, ch_data, use_avg=use_avg)
               for s in schools]
    burdens = [b for b in burdens if b is not None]

    if len(burdens) < 2:
        return 0.5

    b_min, b_max = min(burdens), max(burdens)
    if b_max == b_min:
        return 0.5

    return round(1.0 - (burden - b_min) / (b_max - b_min), 3)


def score_income_growth(school, all_schools=None):
    """
    0-1 score for income growth trend normalized across panel.
    Higher income growth -> higher score (rising incomes support rent growth).
    """
    d = get_metro_income_data(school)
    if not d:
        return None

    schools = all_schools or list(SCHOOL_METRO_MAP.keys())
    trends = []
    for s in schools:
        sd = get_metro_income_data(s)
        if sd:
            trends.append(sd['income_trend'])

    if len(trends) < 2:
        return 0.5

    t_min, t_max = min(trends), max(trends)
    if t_max == t_min:
        return 0.5

    return round((d['income_trend'] - t_min) / (t_max - t_min), 3)


def load_census_income():
    """Return full income profile for all mapped schools."""
    result = {}
    for school in SCHOOL_METRO_MAP:
        d = get_metro_income_data(school)
        if d:
            result[school] = d
    return result
