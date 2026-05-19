"""
Supply Data Module
Source: Marcus & Millichap Multifamily Market Reports 1Q 2026
All three markets now fully populated.
"""

SUPPLY_DATA = {
    'BostonCollege': {
        'market':               'Boston, MA',
        'report_date':          '1Q 2026',
        'source':               'Marcus & Millichap',
        'vacancy_rate':         0.042,
        'vacancy_trend':        'Rising +30bps but 20bps below 20yr avg',
        'submarket_vacancy':    0.040,
        'units_delivered_2025': 8000,
        'units_pipeline_2026':  5000,
        'inventory_growth_pct': 0.011,
        'pipeline_signal':      'TIGHT — decade-low delivery slate',
        'avg_effective_rent':   3170,
        'rent_growth_pct':      0.022,
        'rent_rank':            'Top 5 nationally',
        'avg_cap_rate':         0.045,
        'avg_cap_rate_range':   '4-5%',
        'avg_price_per_unit':   340000,
        'avg_price_per_unit_range': '$310-370K',
        'tailwinds': [
            'Strongest net absorption since 2021 recorded in 2025',
            'Population growth outpacing other Northeastern metros',
            'Housing affordability crisis sustains renter demand',
            'Decade-low 2026 pipeline limits new competition',
            'Cambridge/Downtown vacancy at 4% or below',
        ],
        'headwinds': [
            'International student enrollment declining (federal grant cuts)',
            'Vacancy rising moderately in 2026 (+30bps)',
            'Economic uncertainty from federal policy changes',
        ],
        'overall_signal': 'POSITIVE',
    },

    'UniversityOfMaryland': {
        'market':               'Baltimore, MD',
        'report_date':          '1Q 2026',
        'source':               'Marcus & Millichap',
        'vacancy_rate':         0.041,
        'vacancy_trend':        'Declining 3rd straight year — lowest since 2021',
        'submarket_vacancy':    0.038,
        'units_delivered_2025': None,
        'units_pipeline_2026':  1500,
        'inventory_growth_pct': 0.006,
        'pipeline_signal':      'TIGHT — 0.6% inventory growth 2nd straight year',
        'avg_effective_rent':   1813,
        'rent_growth_pct':      0.024,
        'rent_rank':            '4th highest among 13 major East Coast metros',
        'avg_cap_rate':         0.060,
        'avg_cap_rate_range':   '5-7%',
        'avg_price_per_unit':   125000,
        'avg_price_per_unit_range': '$100-150K',
        'tailwinds': [
            'Vacancy declining for 3rd straight year — lowest since 2021',
            '90bps below trailing decade mean vacancy',
            'Suburban submarkets (Columbia, Towson) at lowest vacancy metrics',
            'Vacancy fell 100+ bps in core-surrounding submarkets in 2025',
            'Rent growth ranks 4th highest among East Coast metros',
            'Exposed to European markets not Chinese imports (tariff resilience)',
        ],
        'headwinds': [
            'Federal government downsizing risk (DC spillover)',
            'Class C downtown vacancy elevated at ~9%',
            'Modest employment growth expected in 2026',
            'Lower rent level ($1,813) limits absolute income potential',
        ],
        'overall_signal': 'POSITIVE',
    },

    'UCBerkeley': {
        'market':               'San Francisco, CA',
        'report_date':          '1Q 2026',
        'source':               'Marcus & Millichap',
        'vacancy_rate':         0.043,
        'vacancy_trend':        'Ticking up +30bps but 70bps below long-term avg',
        'submarket_vacancy':    0.040,
        'units_delivered_2025': None,
        'units_pipeline_2026':  1500,
        'inventory_growth_pct': 0.006,
        'pipeline_signal':      'TIGHT — lowest pipeline since 2012',
        'avg_effective_rent':   3220,
        'rent_growth_pct':      0.034,
        'rent_rank':            'Top 5 major US markets for rent growth',
        'avg_cap_rate':         0.045,
        'avg_cap_rate_range':   '4-5%',
        'avg_price_per_unit':   400000,
        'avg_price_per_unit_range': '$350-500K',
        'tailwinds': [
            'AI/tech boom driving high-paying job creation and renter demand',
            'Downtown SoMa and Mission Bay rents up 10%+ YoY',
            'Lowest construction pipeline since 2012',
            'Transaction volume highest since 2020 — investor confidence returning',
            'Price per unit 25% below 2019-20 peak — compelling entry point',
            'Class B/C vacancy below 4% in San Mateo-Burlingame',
        ],
        'headwinds': [
            'Net job losses continuing in 2026 (white-collar roles)',
            'Vacancy ticking up +30bps in 2026',
            'San Mateo Class A vacancy above 10% — submarket polarization',
            'Rent-by-choice demand concentrated in luxury segment',
        ],
        'overall_signal': 'POSITIVE',
    },
}


def get_supply_score(school):
    if school not in SUPPLY_DATA:
        return None, None
    d = SUPPLY_DATA[school]
    vacancy  = d.get('vacancy_rate')
    pipeline = d.get('inventory_growth_pct')
    rent     = d.get('avg_effective_rent')
    if vacancy is None or pipeline is None or rent is None:
        return None, d
    vacancy_score  = max(0, min(1, 1 - (vacancy - 0.02) / 0.08))
    pipeline_score = max(0, min(1, 1 - pipeline / 0.05))
    rent_score     = max(0, min(1, (rent - 1500) / 2500))
    score = vacancy_score * 0.50 + pipeline_score * 0.30 + rent_score * 0.20
    return round(score, 3), d


def supply_signal(score):
    if score is None:     return 'No Data',     '#6B7A9E'
    if score >= 0.70:     return 'Tight Market', '#10B981'
    if score >= 0.55:     return 'Balanced',     '#F59E0B'
    if score >= 0.40:     return 'Softening',    '#F97316'
    return 'Oversupplied', '#EF4444'
