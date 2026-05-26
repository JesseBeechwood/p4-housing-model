"""
CollegeHouse Data Loader
Loads property-level student housing supply data from CollegeHouse Excel exports.
Each school gets its own file named: collegehouse_SCHOOLNAME.xlsx

Provides the true supply-side picture:
- Purpose-built bed counts (actual, not estimated)
- Real occupancy rates (property-level and market-level)
- Pre-lease rates (leading indicator of future demand)
- Pipeline beds under construction
- Sales comps (price per bed for deal underwriting)
- Historical enrollment vs bed supply ratio
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_collegehouse(school, folder=None):
    """
    Load CollegeHouse data for a given school.
    Returns a dict with all supply metrics or None if file not found.
    """
    if folder is None:
        folder = Path(__file__).parent

    # Try multiple filename patterns
    # Build list of possible filenames to try
    school_aliases = {
        'UniversityOfMaryland': ['umd','universityofmaryland','university_of_maryland'],
        'BostonCollege':        ['bostoncollege','boston_college','bc'],
        'UCBerkeley':           ['ucberkeley','uc_berkeley','ucb','berkeley'],
        'NCState':              ['ncstate','nc_state','ncsu','raleigh-nc'],
        'UNC':                  ['unc','chapel-hill-nc','unc_chapel_hill','chapel_hill'],
        'Pittsburgh':           ['pittsburgh','pitt','pittsburgh-pa'],
        'SMU':                  ['smu','dallas-tx','southern_methodist'],
        'Stanford':             ['stanford','stanford-ca','stanford_university'],
        'Syracuse':             ['syracuse','syracuse-ny','syracuse_university'],
        'UVA':                  ['uva','charlottesville-va','university_of_virginia'],
        'VirginiaTech':         ['virginiatech','virginia_tech','blacksburg-va','vt'],
        'WakeForest':           ['wakeforest','wake_forest','winston-salem-nc','wfu'],
        'FloridaState':         ['floridastate','florida_state','tallahassee-fl','fsu'],
        'Arizona':              ['arizona','tucson-az','university_of_arizona'],
        'ArizonaState':         ['arizonastate','arizona_state','tempe-az','asu'],
        'Baylor':               ['baylor','waco-tx','baylor_university'],
        'BYU':                  ['byu','provo-ut','brigham_young'],
        'Cincinnati':           ['cincinnati','cincinnati-oh','uc_cincinnati'],
        'ColoradoBoulder':      ['coloradoboulder','colorado_boulder','boulder-co','cu_boulder'],
        'Houston':              ['houston','houston-tx','university_of_houston'],
        'IowaState':            ['iowastate','iowa_state','ames-ia','isu'],
        'Kansas':               ['kansas','lawrence-ks','university_of_kansas','ku'],
        'KansasState':          ['kansasstate','kansas_state','manhattan-ks','ksu'],
        'OklahomaState':        ['oklahomastate','oklahoma_state','stillwater-ok','osu'],
        'TCU':                  ['tcu','fort-worth-tx','texas_christian'],
        'TexasTech':            ['texastech','texas_tech','lubbock-tx','ttu'],
        'UCF':                  ['ucf','orlando-fl','central_florida'],
        'Utah':                 ['utah','salt-lake-city-ut','university_of_utah'],
        'WestVirginia':         ['westvirginia','west_virginia','morgantown-wv','wvu'],
        # ── Big Ten ──────────────────────────────────────────────────────────
        'Illinois':         ['illinois','champaign-il','uiuc'],
        'Indiana':          ['indiana','bloomington-in','indiana_university'],
        'IowaHawkeyes':     ['iowahawkeyes','iowa-city-ia','iowa_hawkeyes'],
        'Michigan':         ['michigan','ann-arbor-mi','university_of_michigan'],
        'MichiganState':    ['michiganstate','east-lansing-mi','michigan_state'],
        'Minnesota':        ['minnesota','minneapolis-mn','university_of_minnesota'],
        'Nebraska':         ['nebraska','lincoln-ne','university_of_nebraska'],
        'OhioState':        ['ohiostate','columbus-oh','ohio_state'],
        'Oregon':           ['oregon','eugene-or','university_of_oregon'],
        'PennState':        ['pennstate','state-college-pa','penn_state'],
        'Purdue':           ['purdue','west-lafayette-in','purdue_university'],
        'Rutgers':          ['rutgers','new-brunswick-nj','rutgers_university'],
        'UCLA':             ['ucla','uclasc','los-angeles-ca'],
        'USC':              ['usc','uclasc','los-angeles-ca'],
        'Washington':       ['washington','seattle-wa','university_of_washington'],
        'Wisconsin':        ['wisconsin','madison-wi','university_of_wisconsin'],
        # ── SEC ─────────────────────────────────────────────────────────────
        'Alabama':          ['alabama','tuscaloosa-al','university_of_alabama'],
        'Arkansas':         ['arkansas','fayetteville-ar','university_of_arkansas'],
        'Auburn':           ['auburn','auburn-al','auburn_university'],
        'Florida':          ['florida','gainesville-fl','university_of_florida'],
        'Georgia':          ['georgia','athens-ga','university_of_georgia'],
        'Kentucky':         ['kentucky','lexington-ky','university_of_kentucky'],
        'LSU':              ['lsu','baton-rouge-la','louisiana_state'],
        'Mississippi':      ['mississippi','oxford-ms','ole_miss'],
        'MississippiState': ['mississippistate','starkville-ms','mississippi_state'],
        'Missouri':         ['missouri','columbia-mo','university_of_missouri'],
        'Oklahoma':         ['oklahoma','norman-ok','university_of_oklahoma'],
        'SouthCarolina':    ['southcarolina','columbia-sc','university_of_south_carolina'],
        'Tennessee':        ['tennessee','knoxville-tn','university_of_tennessee'],
        'Texas':            ['texas','austin-tx','university_of_texas'],
        'TexasAM':          ['texasam','college-station-tx','texas_a_m'],
        'Vanderbilt':       ['vanderbilt','nashville-tn','vanderbilt_university'],
    }
    aliases = school_aliases.get(school, [school.lower()])
    patterns = []
    for alias in aliases:
        patterns.append(f'collegehouse_{alias}.xlsx')
    patterns.append(f'collegehouse_{school}.xlsx')
    patterns.append(f'collegehouse_{school.lower()}.xlsx')
    filepath = None
    for p in patterns:
        candidate = Path(folder) / p
        if candidate.exists():
            filepath = candidate
            break

    if filepath is None:
        return None

    try:
        xl = pd.ExcelFile(filepath)
        result = {'school': school, 'source': 'CollegeHouse', 'filepath': str(filepath)}

        # ── Overview sheet ────────────────────────────────────────────────
        if 'Overview' in xl.sheet_names:
            ov = pd.read_excel(xl, sheet_name='Overview').iloc[0]

            result.update({
                'on_campus_beds':           _safe_int(ov.get('On Campus Beds')),
                'purpose_built_beds':       _safe_int(ov.get('Purpose Built Beds')),
                'property_count':           _safe_int(ov.get('Purpose Built Property Count')),
                'unit_count':               _safe_int(ov.get('Purpose Built Unit Count')),
                'occupancy_rate':           _safe_float(ov.get('Occupancy')),
                'pre_lease_rate':           _safe_float(ov.get('Pre-lease')),
                'avg_rent_per_bed':         _safe_float(ov.get('Average Rate')),
                'avg_rate_per_sf':          _safe_float(ov.get('Avg Rate Per SF')),
                'beds_under_construction':  _safe_int(ov.get('Under Construction Beds')),
                'beds_planned':             _safe_int(ov.get('Planned Beds')),
                'estimated_excess':         _safe_int(ov.get('Estimated Excess')),
                'current_availability':     _safe_int(ov.get('Current Availability')),
                'shadow_market_avg_rate':   _safe_float(ov.get('Shadow Market Average Rate')),
                'shadow_market_properties': _safe_int(ov.get('Shadow Market Property Count')),

                # Bedroom type rents
                'rent_1br':  _safe_float(ov.get('1 Bedroom Average Rate *')),
                'rent_2br':  _safe_float(ov.get('2 Bedroom Average Rate *')),
                'rent_3br':  _safe_float(ov.get('3 Bedroom Average Rate *')),
                'rent_4br':  _safe_float(ov.get('4 Bedroom Average Rate *')),
            })

        # ── Universities sheet (enrollment history) ───────────────────────
        if 'Universities' in xl.sheet_names:
            univ = pd.read_excel(xl, sheet_name='Universities')
            if not univ.empty:
                univ_sorted = univ.sort_values('Year')
                result['enrollment_history'] = univ_sorted.to_dict('records')
                result['latest_enrollment_year'] = int(univ_sorted['Year'].max())

                latest = univ_sorted.iloc[-1]
                result['total_enrollment_latest']  = _safe_int(latest.get('Total Enrollment'))
                result['ftug_latest']              = _safe_int(latest.get('Full-Time Undergraduate'))

        # ── Properties sheet ──────────────────────────────────────────────
        if 'Properties' in xl.sheet_names:
            props = pd.read_excel(xl, sheet_name='Properties')
            if not props.empty:
                # Property-level summary
                prop_summary = props.groupby('Property').agg(
                    total_beds=('Property Total Beds','first'),
                    occupancy=('Property Occupancy','first'),
                    avg_rent=('Rental Rate','mean'),
                ).reset_index()
                prop_summary = prop_summary[prop_summary['total_beds'] > 0]
                result['properties'] = prop_summary.to_dict('records')
                result['property_count_detail'] = len(prop_summary)

                # Active supply = beds in properties that are open (occ > 0)
                # Zero-occupancy properties are pipeline/not-yet-open — exclude from supply
                active = prop_summary[prop_summary['occupancy'] > 0.0]
                pipeline_from_props = prop_summary[prop_summary['occupancy'] == 0.0]
                result['active_supply_beds'] = int(active['total_beds'].sum())
                result['pipeline_beds_zero_occ'] = int(pipeline_from_props['total_beds'].sum())

                # Avg rent across operating properties
                operating = prop_summary[prop_summary['occupancy'] > 0.05]
                if not operating.empty:
                    weighted_rent = (operating['avg_rent'] * operating['total_beds']).sum() / operating['total_beds'].sum()
                    result['weighted_avg_rent'] = round(float(weighted_rent), 2)

        # ── Sales Comps sheet ─────────────────────────────────────────────
        if 'Sales Comps' in xl.sheet_names:
            sales = pd.read_excel(xl, sheet_name='Sales Comps')
            if not sales.empty and 'Price Per Bed' in sales.columns:
                recent = sales[sales['Price Per Bed'].notna()].sort_values('Date', ascending=False)
                result['sales_comps'] = recent.head(5).to_dict('records')
                result['avg_price_per_bed_all']    = _safe_float(sales['Price Per Bed'].dropna().mean())
                result['avg_price_per_bed_recent'] = _safe_float(
                    sales[sales['Date'] >= '2020']['Price Per Bed'].dropna().mean()
                )
                if 'Price Per Unit' in sales.columns:
                    result['avg_price_per_unit'] = _safe_float(
                        sales[sales['Date'] >= '2020']['Price Per Unit'].dropna().mean()
                    )

        # ── Computed metrics ──────────────────────────────────────────────
        pb  = result.get('purpose_built_beds', 0) or 0
        # Use active supply (excludes zero-occ pipeline properties) for BtS calculation
        active_pb = result.get('active_supply_beds', pb) or pb
        oc  = result.get('on_campus_beds', 0) or 0
        enr = result.get('ftug_latest', 0) or 0

        # Handle multi-school markets (Boston has BC + BU + Northeastern etc)
        # Use school-specific enrollment vs market-level purpose-built beds
        # proportionally allocated by that school's share of off-campus need
        enr_history = result.get('enrollment_history', [])
        if enr_history:
            # Sum all universities' latest year enrollment to get market total
            latest_yr = max(r['Year'] for r in enr_history)
            market_total_ftug = sum(r.get('Full-Time Undergraduate', 0)
                                    for r in enr_history if r['Year'] == latest_yr)
            market_total_oc   = sum(r.get('Total On-Campus Beds', 0)
                                    for r in enr_history if r['Year'] == latest_yr)
            market_need_off   = max(market_total_ftug - market_total_oc, 1)

            # School-specific share
            school_need_off = max(enr - oc, 0)
            result['students_needing_off_campus'] = school_need_off
            result['market_students_needing_off']  = market_need_off
            result['market_total_ftug']            = int(market_total_ftug)

            # Market-level bed-to-student ratio (how tight is the whole market)
            market_bts = round(active_pb / market_need_off, 3) if market_need_off > 0 else None
            result['bed_to_student_ratio']        = market_bts
            result['bed_to_student_ratio_market'] = market_bts

            if market_bts:
                result['market_saturation'] = 'Oversupplied' if market_bts > 1.1 else \
                                              'Balanced'     if market_bts > 0.9 else \
                                              'Undersupplied'
        elif pb > 0 and oc > 0 and enr > 0:
            students_needing_off_campus = max(enr - oc, 1)
            result['students_needing_off_campus'] = students_needing_off_campus
            result['bed_to_student_ratio'] = round(active_pb / students_needing_off_campus, 3)
            if result['bed_to_student_ratio']:
                result['market_saturation'] = 'Oversupplied' if result['bed_to_student_ratio'] > 1.1 else \
                                              'Balanced'     if result['bed_to_student_ratio'] > 0.9 else \
                                              'Undersupplied'

        # Pipeline pressure: beds under construction as % of existing stock
        uc = result.get('beds_under_construction', 0) or 0
        zero_occ_beds = result.get('pipeline_beds_zero_occ', 0) or 0
        total_pipeline = max(uc, zero_occ_beds)  # use whichever is larger — more accurate
        if active_pb > 0 and total_pipeline is not None:
            result['pipeline_pct'] = round(total_pipeline / active_pb, 4)

        return result

    except Exception as e:
        return {'school': school, 'error': str(e)}


def get_supply_score_ch(ch_data):
    """
    Compute 0-1 supply score from CollegeHouse data.
    More precise than Marcus & Millichap estimates because it uses
    actual bed counts, actual occupancy, and actual pipeline.

    Weights:
      40% occupancy rate         (higher = tighter market)
      30% bed-to-student ratio   (lower = more undersupplied)
      20% pipeline pressure      (lower = less new competition)
      10% pre-lease rate         (higher = strong forward demand)
    """
    if not ch_data or 'error' in ch_data:
        return None

    occ      = ch_data.get('occupancy_rate')
    bts      = ch_data.get('bed_to_student_ratio')
    pipeline = ch_data.get('pipeline_pct')
    prelease = ch_data.get('pre_lease_rate')

    scores = {}

    if occ is not None:
        # 100% occupancy = 1.0, 80% = 0.0
        scores['occupancy'] = max(0, min(1, (occ - 0.80) / 0.20))

    if bts is not None:
        # Ratio < 0.8 (undersupplied) = 1.0, ratio > 1.2 (oversupplied) = 0.0
        scores['bed_supply'] = max(0, min(1, (1.2 - bts) / 0.40))

    if pipeline is not None:
        # 0% pipeline = 1.0, 15%+ pipeline = 0.0
        scores['pipeline'] = max(0, min(1, 1 - pipeline / 0.15))

    if prelease is not None:
        # 90%+ pre-lease = 1.0, 50% = 0.0
        scores['prelease'] = max(0, min(1, (prelease - 0.50) / 0.40))

    if not scores:
        return None

    weights = {'occupancy': 0.40, 'bed_supply': 0.30, 'pipeline': 0.20, 'prelease': 0.10}
    total_weight = sum(weights[k] for k in scores)
    weighted_sum = sum(scores[k] * weights[k] for k in scores)
    return round(weighted_sum / total_weight, 3)


def supply_signal_ch(score):
    if score is None:     return 'No Data',      '#6B7A9E'
    if score >= 0.70:     return 'Tight Market',  '#10B981'
    if score >= 0.55:     return 'Balanced',      '#F59E0B'
    if score >= 0.40:     return 'Softening',     '#F97316'
    return 'Oversupplied', '#EF4444'


def load_all_collegehouse(folder=None):
    """Load CollegeHouse data for all schools that have files."""
    if folder is None:
        folder = Path(__file__).parent
    results = {}
    # Master name map — add new schools here as files are added
    name_map = {
        'umd':                  'UniversityOfMaryland',
        'universityofmaryland': 'UniversityOfMaryland',
        'bostoncollege':        'BostonCollege',
        'boston':               'BostonCollege',
        'ucberkeley':           'UCBerkeley',
        'ucb':                  'UCBerkeley',
        'berkeley':             'UCBerkeley',
        'clemson':              'Clemson',
        'duke':                 'Duke',
        'georgiatech':          'GeorgiaTech',
        'louisville':           'Louisville',
        'miami':                'Miami',
        'georgia_tech':         'GeorgiaTech',
        'gt':                   'GeorgiaTech',
        'ncstate':              'NCState',
        'nc_state':             'NCState',
        'ncsu':                 'NCState',
        'raleigh-nc':           'NCState',
        'unc':                  'UNC',
        'chapel-hill-nc':       'UNC',
        'unc_chapel_hill':      'UNC',
        'chapel_hill':          'UNC',
        'pittsburgh':           'Pittsburgh',
        'pitt':                 'Pittsburgh',
        'pittsburgh-pa':        'Pittsburgh',
        'smu':                  'SMU',
        'dallas-tx':            'SMU',
        'southern_methodist':   'SMU',
        'stanford':             'Stanford',
        'stanford-ca':          'Stanford',
        'stanford_university':  'Stanford',
        'syracuse':             'Syracuse',
        'syracuse-ny':          'Syracuse',
        'syracuse_university':  'Syracuse',
        'uva':                  'UVA',
        'charlottesville-va':   'UVA',
        'university_of_virginia': 'UVA',
        'virginiatech':         'VirginiaTech',
        'virginia_tech':        'VirginiaTech',
        'blacksburg-va':        'VirginiaTech',
        'vt':                   'VirginiaTech',
        'wakeforest':           'WakeForest',
        'wake_forest':          'WakeForest',
        'winston-salem-nc':     'WakeForest',
        'wfu':                  'WakeForest',
        'floridastate':         'FloridaState',
        'florida_state':        'FloridaState',
        'tallahassee-fl':       'FloridaState',
        'fsu':                  'FloridaState',
        'arizona':              'Arizona',
        'tucson-az':            'Arizona',
        'arizonastate':         'ArizonaState',
        'arizona_state':        'ArizonaState',
        'tempe-az':             'ArizonaState',
        'asu':                  'ArizonaState',
        'baylor':               'Baylor',
        'waco-tx':              'Baylor',
        'byu':                  'BYU',
        'provo-ut':             'BYU',
        'cincinnati':           'Cincinnati',
        'cincinnati-oh':        'Cincinnati',
        'coloradoboulder':      'ColoradoBoulder',
        'colorado_boulder':     'ColoradoBoulder',
        'boulder-co':           'ColoradoBoulder',
        'houston':              'Houston',
        'houston-tx':           'Houston',
        'iowastate':            'IowaState',
        'iowa_state':           'IowaState',
        'ames-ia':              'IowaState',
        'isu':                  'IowaState',
        'kansas':               'Kansas',
        'lawrence-ks':          'Kansas',
        'ku':                   'Kansas',
        'kansasstate':          'KansasState',
        'kansas_state':         'KansasState',
        'manhattan-ks':         'KansasState',
        'ksu':                  'KansasState',
        'oklahomastate':        'OklahomaState',
        'oklahoma_state':       'OklahomaState',
        'stillwater-ok':        'OklahomaState',
        'tcu':                  'TCU',
        'fort-worth-tx':        'TCU',
        'texastech':            'TexasTech',
        'texas_tech':           'TexasTech',
        'lubbock-tx':           'TexasTech',
        'ttu':                  'TexasTech',
        'ucf':                  'UCF',
        'orlando-fl':           'UCF',
        'utah':                 'Utah',
        'salt-lake-city-ut':    'Utah',
        'westvirginia':         'WestVirginia',
        'west_virginia':        'WestVirginia',
        'morgantown-wv':        'WestVirginia',
        'wvu':                  'WestVirginia',
    }
    for f in Path(folder).glob('collegehouse_*.xlsx'):
        raw = f.stem.replace('collegehouse_', '').lower()
        school_key = name_map.get(raw, raw.capitalize())
        data = load_collegehouse(school_key, folder)
        if data and 'error' not in data:
            results[school_key] = data
    return results


def _safe_int(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)): return None
        return int(float(v))
    except: return None

def _safe_float(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)): return None
        return float(v)
    except: return None
