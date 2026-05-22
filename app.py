"""
Power 4 Student Housing Investment Model
Private localhost Streamlit dashboard
Run: streamlit run app.py  |  Password: invest2025
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from extractor import load_all_cds
from regression_engine import run_full_analysis
from zillow_loader import load_zillow
from supply_data import get_supply_score, supply_signal, SUPPLY_DATA
from ipeds_loader import load_ipeds, get_ipeds_school_result
from collegehouse_loader import load_all_collegehouse, get_supply_score_ch, supply_signal_ch

st.set_page_config(page_title="Beechwood | Investment Intelligence", page_icon="🏢", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

html,body,[class*="css"]{font-family:'Manrope',sans-serif;}
.stApp{background:#0B0B0D;color:#F5F1EA;}

/* ── Sidebar ──────────────────────────────────────────────────────── */
section[data-testid="stSidebar"]{
    background:#0E0E10!important;
    border-right:1px solid rgba(200,170,125,0.13)!important;
}
section[data-testid="stSidebar"] *{color:#B8A98A!important;}
section[data-testid="stSidebar"] .stRadio label{
    font-family:'Manrope',sans-serif!important;
    font-size:11px!important;
    letter-spacing:.12em!important;
    text-transform:uppercase!important;
    padding:8px 0!important;
    color:#7A6E5F!important;
}
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] input:checked ~ div{
    color:#C8AA7D!important;
}

/* ── Metrics ─────────────────────────────────────────────────────── */
div[data-testid="stMetric"]{
    background:#111113;
    border:1px solid rgba(200,170,125,0.09);
    border-top:2px solid #C8AA7D;
    border-radius:0px;
    padding:20px 18px 16px;
}
div[data-testid="stMetric"] label{
    color:#7A6E5F!important;
    font-size:9px!important;
    font-family:'Manrope',sans-serif!important;
    text-transform:uppercase!important;
    letter-spacing:.18em!important;
    font-weight:600!important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"]{
    color:#F5F1EA!important;
    font-size:26px!important;
    font-weight:300!important;
    font-family:'Manrope',sans-serif!important;
    letter-spacing:-.01em!important;
}

/* ── Cards ──────────────────────────────────────────────────────── */
.bw-card{
    background:#111113;
    border:1px solid rgba(200,170,125,0.09);
    border-top:1px solid rgba(200,170,125,0.27);
    padding:24px;
    margin-bottom:16px;
}
.card{
    background:#111113;
    border:1px solid rgba(200,170,125,0.09);
    border-top:1px solid rgba(200,170,125,0.27);
    padding:20px;
    margin-bottom:16px;
}

/* ── Section Headers ─────────────────────────────────────────────── */
.sh{
    font-family:'Manrope',sans-serif;
    font-size:9px;
    font-weight:700;
    color:#C8AA7D;
    letter-spacing:.25em;
    text-transform:uppercase;
    border-bottom:1px solid rgba(200,170,125,0.13);
    padding-bottom:10px;
    margin:28px 0 18px 0;
}

/* ── Info Blocks ─────────────────────────────────────────────────── */
.ib{
    background:#111113;
    border-left:2px solid #C8AA7D;
    padding:12px 16px;
    font-size:12px;
    font-family:'Inter',sans-serif;
    color:#8A7E6E;
    margin:12px 0;
    line-height:1.7;
}
.warn{
    background:#0F0E0B;
    border-left:2px solid rgba(200,170,125,0.53);
    padding:12px 16px;
    font-size:12px;
    color:#A89060;
    margin:12px 0;
    line-height:1.7;
}
.good{
    background:#0B100E;
    border-left:2px solid #5D6B5C;
    padding:12px 16px;
    font-size:12px;
    color:#6E8A6E;
    margin:12px 0;
    line-height:1.7;
}
.formula{
    background:#0D0D0F;
    border:1px solid rgba(200,170,125,0.13);
    padding:16px 20px;
    font-family:'Courier New',monospace;
    font-size:12px;
    color:#C8AA7D;
    margin:12px 0;
    letter-spacing:.04em;
}

/* ── Thesis Blocks ────────────────────────────────────────────────── */
.thesis{
    background:#0D0D0F;
    border:1px solid rgba(200,170,125,0.13);
    padding:20px 22px;
    margin:12px 0;
}
.thesis-label{
    font-size:8px;
    font-weight:700;
    letter-spacing:.22em;
    text-transform:uppercase;
    color:#C8AA7D;
    margin-bottom:6px;
}
.thesis-text{
    font-size:13px;
    font-family:'Inter',sans-serif;
    color:#9A9080;
    line-height:1.7;
}

/* ── Number Display ──────────────────────────────────────────────── */
.kpi-giant{
    font-size:40px;
    font-weight:200;
    color:#F5F1EA;
    letter-spacing:-.02em;
    font-family:'Manrope',sans-serif;
    line-height:1;
}
.kpi-label{
    font-size:8px;
    font-weight:700;
    letter-spacing:.22em;
    text-transform:uppercase;
    color:#C8AA7D;
    margin-bottom:4px;
}
.kpi-sub{
    font-size:11px;
    color:#5A5040;
    margin-top:4px;
    font-family:'Inter',sans-serif;
}

/* ── Signal Badge ────────────────────────────────────────────────── */
.signal-badge{
    display:inline-block;
    border:1px solid currentColor;
    padding:6px 20px;
    font-size:11px;
    font-weight:700;
    letter-spacing:.18em;
    text-transform:uppercase;
    font-family:'Manrope',sans-serif;
}

/* ── Page Title ──────────────────────────────────────────────────── */
.bw-page-title{
    font-size:10px;
    font-weight:700;
    letter-spacing:.3em;
    text-transform:uppercase;
    color:#C8AA7D;
    margin-bottom:4px;
}
.bw-page-sub{
    font-size:28px;
    font-weight:200;
    color:#F5F1EA;
    letter-spacing:-.01em;
    font-family:'Manrope',sans-serif;
    margin-bottom:24px;
}

/* ── Table styling ───────────────────────────────────────────────── */
.stDataFrame{border:1px solid rgba(200,170,125,0.09)!important;}
thead tr th{background:#0E0E10!important;color:#C8AA7D!important;font-size:9px!important;letter-spacing:.15em!important;text-transform:uppercase!important;}

/* ── Divider ─────────────────────────────────────────────────────── */
hr{border:none;border-top:1px solid rgba(200,170,125,0.13)!important;margin:24px 0!important;}

/* ── Buttons ─────────────────────────────────────────────────────── */
.stButton button{
    background:transparent!important;
    border:1px solid rgba(200,170,125,0.27)!important;
    color:#C8AA7D!important;
    font-family:'Manrope',sans-serif!important;
    font-size:10px!important;
    letter-spacing:.15em!important;
    text-transform:uppercase!important;
    border-radius:0!important;
}
.stButton button:hover{
    border-color:#C8AA7D!important;
    background:rgba(200,170,125,0.04)!important;
}

/* ── Inputs ──────────────────────────────────────────────────────── */
.stTextInput input,.stSelectbox select{
    background:#111113!important;
    border:1px solid rgba(200,170,125,0.13)!important;
    color:#F5F1EA!important;
    border-radius:0!important;
    font-family:'Manrope',sans-serif!important;
}
.stSlider [data-baseweb="slider"] [data-testid="stTickBarMin"],
.stSlider [data-baseweb="slider"] [data-testid="stTickBarMax"]{color:#5A5040!important;}
.stSlider [data-baseweb="slider"] [role="slider"]{background:#C8AA7D!important;border-color:#C8AA7D!important;}
.stSlider [data-baseweb="slider"] [data-testid="stSliderTrackFill"]{background:#C8AA7D!important;}
.stSlider [data-baseweb="slider"] div[class*="Track"]{background:rgba(200,170,125,0.20)!important;}

/* ── Scrollbar ──────────────────────────────────────────────────── */
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:#0B0B0D;}
::-webkit-scrollbar-thumb{background:rgba(200,170,125,0.20);border-radius:0;}
::-webkit-scrollbar-thumb:hover{background:rgba(200,170,125,0.40);}
</style>""", unsafe_allow_html=True)

# ── Beechwood Color System ─────────────────────────────────────────────────
C = dict(
    # Beechwood Identity
    GOLD   ='#C8AA7D',
    IVORY  ='#F5F1EA',
    BLACK  ='#0B0B0D',
    CARD   ='#111113',
    BORDER ='rgba(200,170,125,0.09)',
    MUTED  ='#7A6E5F',
    DIM    ='#4A4035',

    # Signal Colors — muted, institutional
    GREEN  ='#6E9E6E',   # Stabilized/positive — olive green
    AMBER  ='#C8AA7D',   # Caution — champagne
    RED    ='#A0584A',   # Negative — muted terracotta
    PURPLE ='#7A6E9E',   # Model/regression — slate purple
    TEAL   ='#5D8A88',   # Zillow/rent data
    TEXT   ='#F5F1EA',
    BLUE   ='#5D7A9E',   # CDS data
    ORANGE ='#A07040',
    PINK   ='#9E6E7A',
)
SCHOOL_COLORS = [
    '#C8AA7D','#6E9E6E','#5D7A9E','#7A6E9E',
    '#5D8A88','#A07040','#9E6E7A','#A0584A',
]

def ax(yfmt=None,ytitle=None,xtitle=None,tickangle=None):
    xd = dict(
        gridcolor='rgba(200,170,125,0.06)', zerolinecolor='rgba(200,170,125,0.12)',
        tickfont=dict(color='#4A4035',family='Manrope',size=10),
        showline=False, linecolor='rgba(200,170,125,0.12)',
    )
    yd = dict(
        gridcolor='rgba(200,170,125,0.06)', zerolinecolor='rgba(200,170,125,0.12)',
        tickfont=dict(color='#4A4035',family='Manrope',size=10),
        showline=False,
    )
    if yfmt:   yd['tickformat'] = yfmt
    if ytitle: yd['title']      = dict(text=ytitle, font=dict(color='#5A5040',family='Manrope',size=10,))
    if xtitle: xd['title']      = dict(text=xtitle, font=dict(color='#5A5040',family='Manrope',size=10))
    if tickangle is not None: xd['tickangle'] = tickangle
    return xd, yd

def base_fig(height=300,legend=True,hovermode='x unified'):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor='#111113', plot_bgcolor='#0E0E10',
        font=dict(family='Manrope',color='#F5F1EA',size=11),
        height=height, margin=dict(l=16,r=16,t=28,b=16),
        hovermode=hovermode, showlegend=legend,
        legend=dict(orientation='h',y=-0.26,x=0,bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#5A5040',size=10,family='Manrope')),
    )
    return fig

def fmt(val,style='number',prefix=''):
    if val is None or (isinstance(val,float) and np.isnan(val)): return 'N/A'
    try:
        if style=='pct':   return f'{val:.1%}'
        if style=='money': return f'${val:,.0f}'
        return f'{prefix}{val:,.0f}'
    except: return str(val)

def signal_html(signal,score,color):
    return f"""
    <div style="text-align:right">
        <div style="font-size:8px;font-weight:700;letter-spacing:.25em;text-transform:uppercase;
                    color:#C8AA7D;margin-bottom:10px;font-family:Manrope,sans-serif;">
            Investment Signal
        </div>
        <div class="signal-badge" style="color:{color};border-color:{color}44;">
            {signal}
        </div>
        <div style="margin-top:10px;font-size:10px;color:#4A4035;font-family:Manrope,sans-serif;
                    letter-spacing:.08em;">
            SCORE &nbsp;<span style="color:{color};font-weight:600;">{score:.3f}</span>
            <span style="color:#2A2018;"> / 1.000</span>
        </div>
    </div>"""

def vif_badge(v):
    if v is None or np.isinf(v): return f'<span style="color:#EF4444">∞ SEVERE</span>'
    if v < 5:   return f'<span style="color:#10B981">{v:.2f} ✓ OK</span>'
    if v < 10:  return f'<span style="color:#F59E0B">{v:.2f} ⚠ MOD</span>'
    return f'<span style="color:#EF4444">{v:.2f} ✗ HIGH</span>'

def pval_badge(p):
    if p is None: return 'N/A'
    if p > 0.05: return f'<span style="color:#10B981">p={p:.3f} ✓ OK</span>'
    return f'<span style="color:#F59E0B">p={p:.3f} ⚠</span>'

# ── Auth ──────────────────────────────────────────────────────────────────

# ── Known verified data movements (not extraction errors) ─────────────────
VERIFIED_SWINGS = {
    ('Stanford', 2023, 'off_campus_demand'):
        'Expected: Stanford off-campus rate fluctuates 3-8% year-to-year (3pp absolute range). With only ~7,600-8,000 undergrads and 92-97% housed on campus, even 1pp changes create >30% swings on a ~230-630 bed base. All values source-confirmed from CDS Section F.',
    ('Stanford', 2024, 'off_campus_demand'):
        'Expected: Stanford off-campus rate fluctuates 3-8% year-to-year (3pp absolute range). With only ~7,600-8,000 undergrads and 92-97% housed on campus, even 1pp changes create >30% swings on a ~230-630 bed base. All values source-confirmed from CDS Section F.',
    ('Stanford', 2025, 'off_campus_demand'):
        'Expected: Stanford off-campus rate fluctuates 3-8% year-to-year (3pp absolute range). With only ~7,600-8,000 undergrads and 92-97% housed on campus, even 1pp changes create >30% swings on a ~230-630 bed base. All values source-confirmed from CDS Section F.',
    ('Stanford', 2026, 'off_campus_demand'):
        'Expected: Stanford off-campus rate fluctuates 3-8% year-to-year (3pp absolute range). With only ~7,600-8,000 undergrads and 92-97% housed on campus, even 1pp changes create >30% swings on a ~230-630 bed base. All values source-confirmed from CDS Section F.',
('Duke', 2024, 'off_campus_demand'):
        'Source-confirmed: Duke requires all freshmen on campus (0% first-year off-campus rate per CDS Table 17). UG off-campus rate 19%→14.7% as Duke expanded residential capacity. Total UG 6,435 confirmed in CDS Table 2. Values match source exactly.',
    ('GeorgiaTech', 2025, 'off_campus_demand'):
        'Source-confirmed: GT enrollment grew +16.3% (17,713→20,592 per CDS Table 2) and off-campus rate increased 62%→69% per CDS Table 16. Both values directly verified against 2024-2025 CDS source file.',
    ('Louisville', 2025, 'off_campus_demand'):
        'Source-confirmed: On-campus population dropped from ~5,652 to 4,374 students (25.6% of 17,086 UG vs 34.9% prior year), confirmed against institutional housing data showing 4,026 reported beds. Louisville reduced on-campus housing capacity/requirement in 2024-25, structurally shifting ~1,278 students to the off-campus market.',
}


# ── School brand colors + initials for inline SVG logos ─────────────────
SCHOOL_BRAND = {
    'BostonCollege':        ('#8B0000','#C8AA7D','BC'),
    'Clemson':              ('#F66733','#522D80','CU'),
    'Duke':                 ('#003087','#FFFFFF','DU'),
    'GeorgiaTech':          ('#B3A369','#003057','GT'),
    'UCBerkeley':           ('#003262','#FDB515','CAL'),
    'UniversityOfMaryland': ('#E03A3E','#FFD520','UMD'),
    'Alabama':              ('#9E1B32','#FFFFFF','UA'),
    'Auburn':               ('#0C2340','#E87722','AU'),
    'ArizonaState':         ('#8C1D40','#FFC627','ASU'),
    'Arizona':              ('#CC0033','#003366','UA'),
    'Arkansas':             ('#9D2235','#FFFFFF','ARK'),
    'UCLA':                 ('#2D68C4','#F2A900','UCLA'),
    'USC':                  ('#990000','#FFC72C','USC'),
    'Colorado':             ('#CFB87C','#000000','CU'),
    'UCF':                  ('#BA9B37','#000000','UCF'),
    'FloridaState':         ('#782F40','#CEB888','FSU'),
    'Florida':              ('#0021A5','#FA4616','UF'),
    'Miami':                ('#005030','#F47321','UM'),
    'Georgia':              ('#BA0C2F','#000000','UGA'),
    'Illinois':             ('#E84A27','#13294B','UOFI'),
    'Northwestern':         ('#4E2A84','#FFFFFF','NU'),
    'Indiana':              ('#990000','#FFFFFF','IU'),
    'NotreDame':            ('#0C2340','#C99700','ND'),
    'IowaState':            ('#C8102E','#F1BE48','ISU'),
    'Iowa':                 ('#000000','#FFCD00','UI'),
    'Kansas':               ('#0051A5','#E8000D','KU'),
    'Kentucky':             ('#0033A0','#FFFFFF','UK'),
    'Louisville':           ('#AD0000','#000000','UL'),
    'LSU':                  ('#461D7C','#FDD023','LSU'),
    'Michigan':             ('#00274C','#FFCB05','UM'),
    'MichiganState':        ('#18453B','#FFFFFF','MSU'),
    'Minnesota':            ('#7A0019','#FFB71E','UMN'),
    'MississippiState':     ('#660000','#FFFFFF','MSU'),
    'Missouri':             ('#000000','#F1B82D','MU'),
    'Nebraska':             ('#E41C38','#FFFFFF','UNL'),
    'Rutgers':              ('#CC0033','#FFFFFF','RU'),
    'UNC':                  ('#4B9CD3','#FFFFFF','UNC'),
    'NCState':              ('#CC0000','#FFFFFF','NCSU'),
    'Cincinnati':           ('#E00122','#000000','UC'),
    'OhioState':            ('#BA0C2F','#666666','OSU'),
    'OklahomaState':        ('#FF7300','#000000','OSU'),
    'Oklahoma':             ('#841617','#FDF9D8','OU'),
    'Oregon':               ('#154733','#FEE123','UO'),
    'PennState':            ('#041E42','#FFFFFF','PSU'),
    'Pittsburgh':           ('#003594','#FFB81C','PITT'),
    'SouthCarolina':        ('#73000A','#F5A800','USC'),
    'Vanderbilt':           ('#866D4B','#000000','VU'),
    'SMU':                  ('#CC0000','#354CA1','SMU'),
    'Stanford':             ('#8C1515','#FFFFFF','STAN'),
    'Syracuse':             ('#F76900','#FFFFFF','SYR'),
    'UVA':                  ('#232D4B','#F84C1E','UVA'),
    'VirginiaTech':         ('#861F41','#CF4420','VT'),
    'WakeForest':           ('#9E7E38','#000000','WFU'),
    'FloridaState':         ('#782F40','#CEB888','FSU'),
    'Houston':              ('#C8102E','#FFFFFF','UH'),
    'TexasAM':              ('#500000','#FFFFFF','A&M'),
    'Texas':                ('#BF5700','#FFFFFF','UT'),
    'TexasTech':            ('#CC0000','#000000','TTU'),
    'Utah':                 ('#CC0000','#808080','UU'),
    'VirginiaTech':         ('#861F41','#CF4420','VT'),
    'WakeForest':           ('#9E7E38','#000000','WFU'),
    'FloridaState':         ('#782F40','#CEB888','FSU'),
    'Virginia':             ('#232D4B','#F84C1E','UVA'),
    'Washington':           ('#4B2E83','#B7A57A','UW'),
    'WestVirginia':         ('#002855','#EAAA00','WVU'),
    'Wisconsin':            ('#C5050C','#FFFFFF','UW'),
    'Stanford':             ('#8C1515','#FFFFFF','SU'),
    'Louisville':           ('#AD0000','#000000','UL'),
    'Purdue':               ('#CEB888','#000000','PU'),
    'ColoradoBoulder':      ('#CFB87C','#000000','CU'),
    'BYU':                  ('#002E5D','#FFFFFF','BYU'),
    'Baylor':               ('#154734','#FFB81C','BU'),
    'KansasState':          ('#512888','#FFFFFF','KSU'),
    'TCU':                  ('#4D1979','#A3A9AC','TCU'),
}

def logo_img_html(school, size=48, style=""):
    brand = SCHOOL_BRAND.get(school)
    if not brand:
        return ""
    bg, fg, initials = brand
    font_size = max(8, int(size * 0.32))
    letter_spacing = "-0.5px" if len(initials) >= 4 else "0px"
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="' + str(size) + '" height="' + str(size) + '" '
        'viewBox="0 0 ' + str(size) + ' ' + str(size) + '" '
        'style="border-radius:6px;flex-shrink:0;' + style + '">'
        '<rect width="' + str(size) + '" height="' + str(size) + '" fill="' + bg + '"/>'
        '<text x="50%" y="54%" '
        'dominant-baseline="middle" text-anchor="middle" '
        'font-family="Manrope,Arial,sans-serif" '
        'font-size="' + str(font_size) + '" '
        'font-weight="700" '
        'letter-spacing="' + letter_spacing + '" '
        'fill="' + fg + '">' + initials + '</text>'
        '</svg>'
    )
    return svg



if 'auth' not in st.session_state: st.session_state.auth = False
if not st.session_state.auth:
    _,mid,_ = st.columns([1,1.2,1])
    with mid:
        st.markdown("""
        <div style="text-align:center;padding:48px 40px 32px;margin-top:40px;
                    border:1px solid rgba(200,170,125,0.13);border-top:2px solid #C8AA7D;
                    background:#0E0E10;">
            <div style="font-size:9px;font-weight:700;letter-spacing:.35em;text-transform:uppercase;
                        color:#C8AA7D;margin-bottom:16px;font-family:Manrope,sans-serif;">
                Beechwood Property Holdings
            </div>
            <div style="font-size:26px;font-weight:200;color:#F5F1EA;letter-spacing:-.01em;
                        font-family:Manrope,sans-serif;margin-bottom:6px;">
                Investment Intelligence
            </div>
            <div style="font-size:11px;color:#4A4035;letter-spacing:.08em;margin-bottom:32px;
                        font-family:Inter,sans-serif;">
                Power 4 Student Housing Platform
            </div>
            <div style="width:40px;height:1px;background:rgba(200,170,125,0.27);margin:0 auto 28px;"></div>
        </div>
        """, unsafe_allow_html=True)
        pwd = st.text_input('', type='password', placeholder='Access Code')
        if st.button('ENTER PLATFORM', use_container_width=True, type='primary'):
            try:
                correct = st.secrets.get('password', 'Beechwood')
            except Exception:
                correct = 'Beechwood'
            if pwd == correct: st.session_state.auth=True; st.rerun()
            else: st.error('Invalid access code')
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────
CDS_FOLDER = Path(__file__).parent / 'cds_files'

@st.cache_data(ttl=300,show_spinner=False)
def load():
    raw = load_all_cds(str(CDS_FOLDER))
    if raw.empty: return None,None,None
    zillow = load_zillow(Path(__file__).parent)
    ch = {}
    try:
        ch = load_all_collegehouse(Path(__file__).parent)
    except Exception:
        pass
    return run_full_analysis(raw, zillow_data=zillow, ch_data=ch), zillow

def _unpack():
    result = load()
    if result[0] is None: return None,None,None,None,None,None
    (panel,regressions,school_results), zillow = result
    all_school_results = dict(school_results)
    try:
        ipeds_df = load_ipeds(Path(__file__).parent / 'ipeds_2024.csv')
        if ipeds_df is not None and not ipeds_df.empty:
            for _, row in ipeds_df.iterrows():
                if row['school'] not in school_results:   # never overwrite CDS data
                    sr = get_ipeds_school_result(row, panel)
                    all_school_results[row['school']] = sr
    except Exception:
        pass
    # Load CollegeHouse data
    ch_data = {}
    try:
        ch_data = load_all_collegehouse(Path(__file__).parent)
    except Exception:
        pass
    return panel, regressions, school_results, zillow, all_school_results, ch_data

with st.spinner('Running regressions...'):
    panel, regressions, school_results, zillow_data, all_school_results, ch_data = _unpack()

if panel is None:
    st.error('No CDS files found in cds_files/ folder.'); st.stop()

schools = sorted(school_results.keys())

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    import datetime
    now = datetime.datetime.now().strftime('%b %d, %Y  %H:%M')
    st.markdown(f"""
    <div style="padding:28px 0 24px;">
        <div style="font-size:8px;font-weight:700;letter-spacing:.3em;text-transform:uppercase;
                    color:#C8AA7D;margin-bottom:8px;font-family:Manrope,sans-serif;">
            Beechwood Property Holdings
        </div>
        <div style="font-size:16px;font-weight:200;color:#F5F1EA;letter-spacing:.02em;
                    font-family:Manrope,sans-serif;margin-bottom:4px;">
            Investment Intelligence
        </div>
        <div style="font-size:9px;color:#3A3028;letter-spacing:.08em;font-family:Inter,sans-serif;">
            {now}
        </div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,rgba(200,170,125,0.20),transparent);margin-bottom:20px;"></div>
    """, unsafe_allow_html=True)

    page = st.radio('',['School Dashboard','Market Rankings','Regression Results',
                        'Compare Schools','Data Audit','Sensitivity Explorer','Data Table','AI Overview','Market Map'],
                    label_visibility='collapsed')

    st.markdown(f"""
    <div style="height:1px;background:linear-gradient(90deg,rgba(200,170,125,0.20),transparent);margin:20px 0;"></div>
    """, unsafe_allow_html=True)
    selected = st.selectbox('', schools, label_visibility='collapsed')

    st.markdown(f"""
    <div style="height:1px;background:linear-gradient(90deg,rgba(200,170,125,0.20),transparent);margin:20px 0;"></div>
    <div style="font-size:9px;letter-spacing:.12em;line-height:2.4;font-family:Manrope,sans-serif;">
        <div style="color:#4A4035;text-transform:uppercase;">Coverage</div>
        <div style="color:#7A6E5F;">CDS Schools&nbsp;&nbsp;<span style="color:#C8AA7D;font-weight:600;">{len(schools)}</span></div>
        <div style="color:#7A6E5F;">Total Markets&nbsp;&nbsp;<span style="color:#C8AA7D;font-weight:600;">{len(all_school_results)}</span></div>
        <div style="color:#7A6E5F;">Observations&nbsp;&nbsp;<span style="color:#C8AA7D;font-weight:600;">{len(panel)}</span></div>
    </div>
    <div style="margin-top:20px;font-size:9px;color:#3A3028;letter-spacing:.08em;line-height:2;font-family:Inter,sans-serif;">
        ADD SCHOOLS<br>
        <span style="color:#4A4035;">Drop CDS files into cds_files/</span>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# SCHOOL DASHBOARD
# ════════════════════════════════════════════════════════════════════════
if page == 'School Dashboard':
    sr  = school_results[selected]
    sp  = sr['panel']
    fc  = sr['forecast']
    sig = sr['signal']; sco = sr['investment_score']; scol = sr['signal_color']

    h1,h2 = st.columns([3,1])
    with h1:
        logo_html = logo_img_html(selected, size=52, style="margin-right:14px;vertical-align:middle;")
        name_span = '<span style="font-size:28px;font-weight:200;color:#F5F1EA;letter-spacing:-.01em;font-family:Manrope,sans-serif;">' + selected + '</span>'
        yr_min = int(sp['academic_year'].min())
        yr_max = int(sp['academic_year'].max())
        yr_count = len(sp)
        header_html = (
            '<div class="bw-page-title">Market Analysis</div>'
            '<div style="display:flex;align-items:center;gap:0px;margin-bottom:6px;">'
            + logo_html + name_span +
            '</div>'
            '<div style="font-size:9px;color:#3A3028;letter-spacing:.15em;font-family:Manrope,sans-serif;'
            'text-transform:uppercase;margin-bottom:24px;">'
            'CDS ' + str(yr_min) + '&ndash;' + str(yr_max) + ' &nbsp;&middot;&nbsp; '
            + str(yr_count) + ' Years &nbsp;&middot;&nbsp; Full Data'
            '</div>'
        )
        st.markdown(header_html, unsafe_allow_html=True)
    with h2:
        st.markdown(signal_html(sig,sco,scol),unsafe_allow_html=True)

    # Narrative Intelligence Panels
    def gen_thesis(sr, z, ch):
        sig = sr.get('signal','')
        oos = sr.get('pct_oos_ug',0) or 0
        ret = sr.get('retention_rate',0) or 0
        bts = ch.get('bed_to_student_ratio') if ch else None
        rent_mom = z.get('momentum_label','') if z else ''
        trend = (z.get('rent_trend_pct',0) or 0) if z else 0
        parts = []
        if oos > 0.50: parts.append(f"Elevated OOS enrollment ({oos:.0%}) sustains structural off-campus demand with near-certain multi-year renters.")
        elif oos > 0.25: parts.append(f"Meaningful OOS share ({oos:.0%}) supports durable renter demand across the enrollment cycle.")
        if ret > 0.93: parts.append(f"Retention at {ret:.0%} ensures strong cohort continuity — current renters return at a premium rate.")
        if bts and bts < 0.80: parts.append("Purpose-built supply materially trails demand — market is structurally undersupplied.")
        if 'Strengthening' in rent_mom: parts.append("Rent momentum re-accelerating — leading indicator of near-term rental growth.")
        elif trend > 3: parts.append(f"Long-run rent trend of {trend:.1f}%/yr supports compounding NOI growth assumptions.")
        thesis = ' '.join(parts) if parts else "Demand fundamentals under evaluation. Additional CDS data will sharpen this assessment."
        risks = []
        if bts and bts > 1.1: risks.append(f"Purpose-built supply exceeds renter demand (ratio {bts:.2f}) — absorption risk present.")
        if ch and ch.get('beds_under_construction',0) > 0:
            uc = ch.get('beds_under_construction',0)
            risks.append(f"{uc:,} beds under construction create near-term supply pressure on occupancy and rents.")
        if 'Cooling' in rent_mom: risks.append("Rent demand momentum decelerating — monitor before committing capital.")
        risk = risks[0] if risks else "No material supply-side risks identified with current data coverage."
        if sig in ['STRONG BUY','BUY'] and (not bts or bts < 1.0):
            outlook = "Favorable near-term entry. Demand-supply dynamics support stabilized occupancy above 93%. Long-term pricing power intact."
        elif sig == 'HOLD':
            outlook = "Market requires monitoring. Near-term oversupply creates cap rate compression risk. Re-evaluate at next lease cycle."
        else:
            outlook = "Proceed with caution. Additional data collection required before investment committee presentation."
        return thesis, risk, outlook

    z_sel  = zillow_data.get(selected) if zillow_data else None
    ch_sel = ch_data.get(selected) if ch_data else None
    thesis_txt, risk_txt, outlook_txt = gen_thesis(sr, z_sel, ch_sel)
    thesis_html = (
        '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;margin:24px 0;background:rgba(200,170,125,0.07);">'
        + '<div class="thesis"><div class="thesis-label">Investment Thesis</div><div class="thesis-text">' + thesis_txt + '</div></div>'
        + '<div class="thesis"><div class="thesis-label" style="color:#A0584A;">Key Risk</div><div class="thesis-text">' + risk_txt + '</div></div>'
        + '<div class="thesis"><div class="thesis-label" style="color:#5D8A88;">Market Outlook</div><div class="thesis-text">' + outlook_txt + '</div></div>'
        + '</div>'
    )
    st.markdown(thesis_html, unsafe_allow_html=True)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: st.metric('Undergrads',fmt(sr['total_undergrad']))
    with c2: st.metric('Off-Campus Beds',fmt(sr['off_campus_demand']))
    with c3: st.metric('Off-Campus Rate',fmt(sr['pct_ug_off_campus'],'pct'))
    with c4: st.metric('Retention',fmt(sr['retention_rate'],'pct'))
    with c5: st.metric('OOS Share',fmt(sr['pct_oos_ug'],'pct'))
    with c6: st.metric('Instate Tuition',fmt(sr['tuition_instate'],'money'))
    st.markdown('<br>',unsafe_allow_html=True)

    left,right = st.columns(2)
    with left:
        st.markdown('<div class="sh">5-Year Demand Forecast</div>',unsafe_allow_html=True)
        if fc:
            obs_x = list(sp['academic_year'])
            obs_y = [float(v) if v is not None else None for v in sp['off_campus_demand']]
            fc_x  = [f['year'] for f in fc]
            fc_y  = [f['pred_demand'] for f in fc]
            fc_lo = [f['ci_lower'] for f in fc]
            fc_hi = [f['ci_upper'] for f in fc]
            fig = base_fig(300)
            fig.add_trace(go.Scatter(x=fc_x+fc_x[::-1],y=fc_hi+fc_lo[::-1],fill='toself',
                fillcolor='rgba(37,99,235,0.12)',line=dict(color='rgba(0,0,0,0)'),name='95% CI'))
            fig.add_trace(go.Scatter(x=obs_x,y=obs_y,mode='lines+markers',name='Observed',
                line=dict(color=C['GREEN'],width=2.5),marker=dict(size=7,color=C['GREEN'])))
            fig.add_trace(go.Scatter(x=[obs_x[-1]]+fc_x,y=[obs_y[-1]]+fc_y,
                mode='lines+markers',name='Forecast',
                line=dict(color=C['BLUE'],width=2.5,dash='dash'),marker=dict(size=7,color=C['BLUE'])))
            xd,yd = ax(ytitle='Beds')
            fig.update_layout(xaxis=xd,yaxis=yd)
            st.plotly_chart(fig,use_container_width=True)
            fc_df = pd.DataFrame(fc)[['year','pred_demand','ci_lower','ci_upper','est_enrollment','pct_oos','off_campus_rate']]
            fc_df.columns=['Year','Forecast','CI Low','CI High','Enrollment','OOS%','Off-Campus Rate']
            fc_df['OOS%']           = fc_df['OOS%'].map(lambda x: f'{x:.1%}')
            fc_df['Off-Campus Rate']= fc_df['Off-Campus Rate'].map(lambda x: f'{x:.1%}')
            fc_df['Forecast']       = fc_df['Forecast'].map(lambda x: f'{x:,}')
            fc_df['CI Low']         = fc_df['CI Low'].map(lambda x: f'{x:,}')
            fc_df['CI High']        = fc_df['CI High'].map(lambda x: f'{x:,}')
            fc_df['Enrollment']     = fc_df['Enrollment'].map(lambda x: f'{x:,}')
            st.dataframe(fc_df,use_container_width=True,hide_index=True)
        else:
            st.info('Need 3+ years of data to generate forecast.')

    with right:
        # ── Enrollment Forecast ───────────────────────────────────────────
        st.markdown('<div class="sh">Enrollment Forecast</div>', unsafe_allow_html=True)
        enroll_hist_x = list(sp['academic_year'])
        enroll_hist_y = [float(v) if v is not None else None for v in sp['total_undergrad']]
        enroll_hist_pairs = [(x, y) for x, y in zip(enroll_hist_x, enroll_hist_y) if y is not None]

        if len(enroll_hist_pairs) >= 2:
            # Project forward using linear trend
            eh_x, eh_y = zip(*enroll_hist_pairs)
            years_idx  = list(range(len(eh_x)))
            proj_start = max(eh_x) + 1
            proj_years = list(range(proj_start, proj_start + 5))
            proj_idx   = list(range(len(eh_x), len(eh_x) + 5))

            coeffs = np.polyfit(years_idx, eh_y, 1)
            proj_y = [max(0, coeffs[0] * i + coeffs[1]) for i in proj_idx]

            # CAGR annotation
            if eh_y[0] and eh_y[-1] and len(eh_y) > 1:
                cagr = (eh_y[-1] / eh_y[0]) ** (1 / (len(eh_y) - 1)) - 1
                cagr_txt = f'Historical CAGR: {cagr:+.1%}/yr'
            else:
                cagr_txt = ''

            fig_enr = base_fig(280)
            fig_enr.add_trace(go.Scatter(
                x=list(eh_x), y=list(eh_y),
                mode='lines+markers', name='Observed',
                line=dict(color=C['AMBER'], width=2.5),
                marker=dict(size=7, color=C['AMBER'])
            ))
            fig_enr.add_trace(go.Scatter(
                x=[eh_x[-1]] + proj_years,
                y=[eh_y[-1]] + proj_y,
                mode='lines+markers', name='Projected',
                line=dict(color=C['BLUE'], width=2.5, dash='dash'),
                marker=dict(size=7, color=C['BLUE'])
            ))
            xd_e, yd_e = ax(ytitle='Undergrad Enrollment')
            fig_enr.update_layout(
                xaxis=xd_e, yaxis=yd_e,
                annotations=[dict(
                    text=cagr_txt, xref='paper', yref='paper',
                    x=0.01, y=0.97, showarrow=False,
                    font=dict(size=11, color=C['MUTED']),
                    align='left'
                )] if cagr_txt else []
            )
            st.plotly_chart(fig_enr, use_container_width=True)

            # Projection table
            proj_df = pd.DataFrame({
                'Year':       proj_years,
                'Projected Enrollment': [f'{int(round(v)):,}' for v in proj_y],
                'vs. Latest': [f'{((v / eh_y[-1]) - 1):+.1%}' for v in proj_y],
            })
            st.dataframe(proj_df, use_container_width=True, hide_index=True)
        else:
            st.info('Need 2+ years of enrollment data to project.')

    # Historical Trends + Cost Trends side by side
    hist_col, cost_col = st.columns(2)
    with hist_col:
        st.markdown('<div class="sh">Historical Trends</div>',unsafe_allow_html=True)
        fig2 = base_fig(260)
        yrs  = list(sp['academic_year'])
        for col,label,color in [('pct_ug_off_campus','Off-Campus Rate',C['GREEN']),
                                  ('retention_rate','Retention Rate',C['BLUE']),
                                  ('pct_oos_ug','OOS Share',C['AMBER']),
                                  ('pct_need_met','Need Met',C['PURPLE'])]:
            if col in sp.columns:
                fig2.add_trace(go.Scatter(x=yrs,y=list(sp[col]),mode='lines+markers',name=label,
                    line=dict(color=color,width=2),marker=dict(size=7)))
        xd,yd = ax(yfmt='.0%')
        fig2.update_layout(xaxis=xd,yaxis=yd)
        st.plotly_chart(fig2,use_container_width=True)

    with cost_col:
        st.markdown('<div class="sh" style="font-size:15px">Cost Trends</div>',unsafe_allow_html=True)
        fig3 = base_fig(220)
        for col,label,color in [('tuition_instate','Tuition',C['BLUE']),
                                  ('avg_aid_package','Avg Aid',C['GREEN']),
                                  ('on_campus_housing','On-Campus Room',C['AMBER']),
                                  ('off_campus_housing','Off-Campus Est.',C['PURPLE'])]:
            if col in sp.columns:
                pairs = [(y,v) for y,v in zip(yrs,sp[col])
                         if v is not None and not (isinstance(v,float) and np.isnan(v))]
                if pairs:
                    py,pv = zip(*pairs)
                    fig3.add_trace(go.Scatter(x=list(py),y=list(pv),mode='lines+markers',name=label,
                        line=dict(color=color,width=2),marker=dict(size=7)))
        xd,yd = ax(yfmt='$,.0f')
        fig3.update_layout(xaxis=xd,yaxis=yd)
        st.plotly_chart(fig3,use_container_width=True)

    # Zillow Rent Market section
    z = zillow_data.get(selected) if zillow_data else None
    if z:
        st.markdown('<div class="sh">Rent Market Data (Zillow)</div>',unsafe_allow_html=True)

        # Metric cards
        zc1,zc2,zc3,zc4 = st.columns(4)
        mom_color  = C['GREEN'] if z.get('accelerating') else C['AMBER']
        rent_color = C['GREEN'] if (z.get('rent_trend_pct') or 0) > 2 else C['AMBER']
        with zc1:
            rent_str = f"${z['latest_rent']:,.0f}/mo" if z.get('latest_rent') else 'N/A'
            st.markdown(f"""<div class="card" style="text-align:center">
                <div style="font-size:11px;color:{C['MUTED']};text-transform:uppercase;margin-bottom:6px;">Current Market Rent</div>
                <div style="font-size:22px;font-weight:700;color:{rent_color};">{rent_str}</div>
                <div style="font-size:11px;color:{C['MUTED']};margin-top:4px;">{z.get('latest_date','')[:7]} · {z['metro']}</div>
            </div>""",unsafe_allow_html=True)
        with zc2:
            tpct = z.get('rent_trend_pct',0) or 0
            tc   = C['GREEN'] if tpct > 2 else C['AMBER']
            st.markdown(f"""<div class="card" style="text-align:center">
                <div style="font-size:11px;color:{C['MUTED']};text-transform:uppercase;margin-bottom:6px;">Rent Growth Trend</div>
                <div style="font-size:22px;font-weight:700;color:{tc};">{tpct:+.1f}%/yr</div>
                <div style="font-size:11px;color:{C['MUTED']};margin-top:4px;">Linear trend 2015–2026</div>
            </div>""",unsafe_allow_html=True)
        with zc3:
            st.markdown(f"""<div class="card" style="text-align:center">
                <div style="font-size:11px;color:{C['MUTED']};text-transform:uppercase;margin-bottom:6px;">Demand Momentum</div>
                <div style="font-size:20px;font-weight:700;color:{mom_color};">{z.get('momentum_label','N/A')}</div>
                <div style="font-size:11px;color:{C['MUTED']};margin-top:4px;">vs prior 12 months (ZORDI)</div>
            </div>""",unsafe_allow_html=True)
        with zc4:
            zordi_val = z.get('latest_zordi')
            base_color = C['GREEN'] if z.get('above_baseline') else C['RED']
            zordi_str = f"{zordi_val:.1f}" if zordi_val is not None else 'N/A'
            st.markdown(f"""<div class="card" style="text-align:center">
                <div style="font-size:11px;color:{C['MUTED']};text-transform:uppercase;margin-bottom:6px;">Demand Index (ZORDI)</div>
                <div style="font-size:22px;font-weight:700;color:{base_color};">{zordi_str}</div>
                <div style="font-size:11px;color:{C['MUTED']};margin-top:4px;">{'Above' if z.get('above_baseline') else 'Below'} baseline</div>
            </div>""",unsafe_allow_html=True)

        # Rent trend chart (dollar rents)
        rl,rr = st.columns(2)
        with rl:
            annual = z.get('rent_annual_avg',{})
            if annual:
                yrs_z  = sorted(annual.keys())
                vals_z = [annual[y] for y in yrs_z]

                # Compute school-specific rent growth trend
                x_arr = np.array(range(len(yrs_z)), dtype=float)
                y_arr = np.array(vals_z, dtype=float)
                slope, intercept = np.polyfit(x_arr, y_arr, 1)
                rent_growth_rate = slope / y_arr[0]  # annual % growth

                # Project 5 years forward
                last_yr   = yrs_z[-1]
                last_rent = vals_z[-1]
                proj_yrs  = [last_yr + i for i in range(1,6)]
                proj_vals = [last_rent * (1 + rent_growth_rate) ** i for i in range(1,6)]
                ci_half   = [last_rent * 0.03 * i for i in range(1,6)]  # ~3% uncertainty per yr

                fig_z = base_fig(260,legend=True)
                # Historical
                fig_z.add_trace(go.Scatter(x=yrs_z,y=vals_z,mode='lines+markers',
                    name='Historical',line=dict(color=C['GREEN'],width=2.5),
                    marker=dict(size=8,color=C['GREEN']),
                    hovertemplate='%{x}: $%{y:,.0f}/mo<extra></extra>'))
                # CI band
                fig_z.add_trace(go.Scatter(
                    x=proj_yrs+proj_yrs[::-1],
                    y=[v+c for v,c in zip(proj_vals,ci_half)]+[v-c for v,c in zip(proj_vals[::-1],ci_half[::-1])],
                    fill='toself',fillcolor='rgba(245,158,11,0.12)',
                    line=dict(color='rgba(0,0,0,0)'),name='95% CI',showlegend=True))
                # Projection line
                fig_z.add_trace(go.Scatter(
                    x=[last_yr]+proj_yrs,y=[last_rent]+proj_vals,
                    mode='lines+markers',name=f'Forecast ({rent_growth_rate:.1%}/yr)',
                    line=dict(color=C['AMBER'],width=2.5,dash='dash'),
                    marker=dict(size=7,color=C['AMBER']),
                    hovertemplate='%{x}: $%{y:,.0f}/mo (projected)<extra></extra>'))
                xd_z,yd_z = ax(yfmt='$,.0f',ytitle='Avg Rent ($/mo)')
                fig_z.update_layout(xaxis=xd_z,yaxis=yd_z,
                    title=dict(text=f'Rent History + 5-Year Projection  ({rent_growth_rate:.1%}/yr trend)',
                               font=dict(size=13,color=C['MUTED'])))
                st.plotly_chart(fig_z,use_container_width=True)
                st.markdown(f'<div style="font-size:11px;color:{C["MUTED"]};margin-top:-8px;">Projection uses school-specific Zillow rent trend. By {proj_yrs[-1]}: <strong style="color:{C["AMBER"]}">${proj_vals[-1]:,.0f}/mo</strong> (range: ${proj_vals[-1]-ci_half[-1]:,.0f}–${proj_vals[-1]+ci_half[-1]:,.0f})</div>',unsafe_allow_html=True)

        with rr:
            yoy = z.get('rent_yoy',{})
            if yoy:
                yoy_yrs  = sorted(yoy.keys())
                yoy_vals = [yoy[y] for y in yoy_yrs]
                yoy_colors = [C['GREEN'] if v>0 else C['RED'] for v in yoy_vals]
                fig_yoy = base_fig(220,legend=False)
                fig_yoy.add_trace(go.Bar(x=yoy_yrs,y=yoy_vals,marker_color=yoy_colors,
                    hovertemplate='%{x}: %{y:+.1f}%<extra></extra>'))
                fig_yoy.add_hline(y=0,line=dict(color=C['MUTED'],width=1,dash='dot'))
                xd_y,yd_y = ax(yfmt='+.1f',ytitle='YoY Rent Growth (%)')
                fig_yoy.update_layout(xaxis=xd_y,yaxis=yd_y,
                    title=dict(text='Year-over-Year Rent Growth',font=dict(size=13,color=C['MUTED'])))
                st.plotly_chart(fig_yoy,use_container_width=True)

        st.markdown('<div class="ib"><strong>Reading this data:</strong> Current Market Rent is the actual dollar rent per month from Zillow smoothed index. Rent Growth Trend is the long-run annual growth rate since 2015. Demand Momentum (ZORDI) tracks whether rental search activity is accelerating or cooling - a leading indicator of future rent movement.</div>',unsafe_allow_html=True)

    # Supply Market section
    supply_score, supply_info = get_supply_score(selected)
    if supply_info:
        st.markdown('<div class="sh">Supply Market Data</div>',unsafe_allow_html=True)
        ss1,ss2,ss3,ss4 = st.columns(4)
        sig_label, sig_color = supply_signal(supply_score)
        with ss1:
            vac = supply_info.get('vacancy_rate')
            vc  = C['GREEN'] if vac and vac < 0.05 else C['AMBER']
            st.markdown(f"""<div class="card" style="text-align:center">
                <div style="font-size:11px;color:{C['MUTED']};text-transform:uppercase;margin-bottom:6px;">Market Vacancy Rate</div>
                <div style="font-size:22px;font-weight:700;color:{vc};">{f'{vac:.1%}' if vac else 'N/A'}</div>
                <div style="font-size:11px;color:{C['MUTED']};margin-top:4px;">{supply_info.get('vacancy_trend','')[:35]}</div>
            </div>""",unsafe_allow_html=True)
        with ss2:
            pip = supply_info.get('inventory_growth_pct')
            pc  = C['GREEN'] if pip and pip < 0.02 else C['AMBER']
            st.markdown(f"""<div class="card" style="text-align:center">
                <div style="font-size:11px;color:{C['MUTED']};text-transform:uppercase;margin-bottom:6px;">Pipeline Growth</div>
                <div style="font-size:22px;font-weight:700;color:{pc};">{f'{pip:.1%}' if pip else 'N/A'}</div>
                <div style="font-size:11px;color:{C['MUTED']};margin-top:4px;">{supply_info.get('pipeline_signal','')[:35]}</div>
            </div>""",unsafe_allow_html=True)
        with ss3:
            rent = supply_info.get('avg_effective_rent')
            st.markdown(f"""<div class="card" style="text-align:center">
                <div style="font-size:11px;color:{C['MUTED']};text-transform:uppercase;margin-bottom:6px;">Avg Effective Rent</div>
                <div style="font-size:22px;font-weight:700;color:{C['GREEN']};">{f'${rent:,}/mo' if rent else 'N/A'}</div>
                <div style="font-size:11px;color:{C['MUTED']};margin-top:4px;">{supply_info.get('rent_rank','')}</div>
            </div>""",unsafe_allow_html=True)
        with ss4:
            cap = supply_info.get('avg_cap_rate')
            st.markdown(f"""<div class="card" style="text-align:center">
                <div style="font-size:11px;color:{C['MUTED']};text-transform:uppercase;margin-bottom:6px;">Supply Signal</div>
                <div style="font-size:20px;font-weight:700;color:{sig_color};">{sig_label}</div>
                <div style="font-size:11px;color:{C['MUTED']};margin-top:4px;">Score: {f'{supply_score:.3f}' if supply_score else 'N/A'} | Cap: {supply_info.get('avg_cap_rate_range','N/A')}</div>
            </div>""",unsafe_allow_html=True)

        # Tailwinds and headwinds
        tw_col, hw_col = st.columns(2)
        with tw_col:
            st.markdown(f'<div style="font-size:12px;color:#10B981;font-weight:600;margin-bottom:6px;">Tailwinds</div>',unsafe_allow_html=True)
            for t in supply_info.get('tailwinds',[]):
                st.markdown(f'<div style="font-size:12px;color:{C["MUTED"]};padding:3px 0;border-bottom:1px solid {C["BORDER"]};">+ {t}</div>',unsafe_allow_html=True)
        with hw_col:
            st.markdown(f'<div style="font-size:12px;color:#EF4444;font-weight:600;margin-bottom:6px;">Headwinds</div>',unsafe_allow_html=True)
            for h in supply_info.get('headwinds',[]):
                st.markdown(f'<div style="font-size:12px;color:{C["MUTED"]};padding:3px 0;border-bottom:1px solid {C["BORDER"]};">- {h}</div>',unsafe_allow_html=True)

        src = supply_info.get('source',''); rdate = supply_info.get('report_date','')
        st.markdown(f'<div style="font-size:11px;color:#4A5568;margin-top:8px;">Source: {src} {rdate}</div>',unsafe_allow_html=True)
    elif selected in ['UniversityOfMaryland','UCBerkeley']:
        st.markdown('<div class="sh">Supply Market Data</div>',unsafe_allow_html=True)
        st.markdown('<div class="warn">Supply data not yet loaded for this market. Download the multifamily market report for this city from marcusmillichap.com/research and upload it to add supply data.</div>',unsafe_allow_html=True)

    # CollegeHouse Supply Section
    ch = ch_data.get(selected) if ch_data else None
    if ch and 'error' not in ch:
        st.markdown('<div class="sh">Supply Data (CollegeHouse)</div>',unsafe_allow_html=True)
        ch_score = get_supply_score_ch(ch)
        ch_sig, ch_col = supply_signal_ch(ch_score)
        cc1,cc2,cc3,cc4,cc5 = st.columns(5)
        with cc1:
            occ = ch.get('occupancy_rate')
            oc = C['GREEN'] if occ and occ > 0.94 else C['AMBER']
            st.markdown(f"""<div class="card" style="text-align:center">
                <div style="font-size:11px;color:{C['MUTED']};text-transform:uppercase;margin-bottom:6px;">Occupancy Rate</div>
                <div style="font-size:22px;font-weight:700;color:{oc};">{f'{occ:.1%}' if occ else 'N/A'}</div>
                <div style="font-size:11px;color:{C['MUTED']};margin-top:4px;">Purpose-built stock</div>
            </div>""",unsafe_allow_html=True)
        with cc2:
            bts = ch.get('bed_to_student_ratio')
            btc = C['GREEN'] if bts and bts < 0.9 else (C['AMBER'] if bts and bts < 1.1 else C['RED'])
            st.markdown(f"""<div class="card" style="text-align:center">
                <div style="font-size:11px;color:{C['MUTED']};text-transform:uppercase;margin-bottom:6px;">Bed-to-Student Ratio</div>
                <div style="font-size:22px;font-weight:700;color:{btc};">{f'{bts:.3f}' if bts else 'N/A'}</div>
                <div style="font-size:11px;color:{C['MUTED']};margin-top:4px;">{ch.get('market_saturation','N/A')}</div>
            </div>""",unsafe_allow_html=True)
        with cc3:
            pl = ch.get('pre_lease_rate')
            plc = C['GREEN'] if pl and pl > 0.75 else C['AMBER']
            st.markdown(f"""<div class="card" style="text-align:center">
                <div style="font-size:11px;color:{C['MUTED']};text-transform:uppercase;margin-bottom:6px;">Pre-Lease Rate</div>
                <div style="font-size:22px;font-weight:700;color:{plc};">{f'{pl:.1%}' if pl else 'N/A'}</div>
                <div style="font-size:11px;color:{C['MUTED']};margin-top:4px;">Forward demand signal</div>
            </div>""",unsafe_allow_html=True)
        with cc4:
            uc = ch.get('beds_under_construction',0) or 0
            pp = ch.get('pipeline_pct',0) or 0
            ucc = C['GREEN'] if pp < 0.05 else C['AMBER']
            st.markdown(f"""<div class="card" style="text-align:center">
                <div style="font-size:11px;color:{C['MUTED']};text-transform:uppercase;margin-bottom:6px;">Under Construction</div>
                <div style="font-size:22px;font-weight:700;color:{ucc};">{uc:,} beds</div>
                <div style="font-size:11px;color:{C['MUTED']};margin-top:4px;">{pp:.1%} of existing stock</div>
            </div>""",unsafe_allow_html=True)
        with cc5:
            st.markdown(f"""<div class="card" style="text-align:center">
                <div style="font-size:11px;color:{C['MUTED']};text-transform:uppercase;margin-bottom:6px;">Supply Signal</div>
                <div style="font-size:20px;font-weight:700;color:{ch_col};">{ch_sig}</div>
                <div style="font-size:11px;color:{C['MUTED']};margin-top:4px;">Score: {f'{ch_score:.3f}' if ch_score else 'N/A'}</div>
            </div>""",unsafe_allow_html=True)

        # Avg rent by bedroom type
        rents = [(k,v) for k,v in [('1BR',ch.get('rent_1br')),('2BR',ch.get('rent_2br')),('3BR',ch.get('rent_3br')),('4BR',ch.get('rent_4br'))] if v]
        if rents:
            st.markdown('<div style="font-size:13px;color:#6B7A9E;margin:12px 0 6px 0;font-weight:600;">Average Rent by Bedroom Type</div>',unsafe_allow_html=True)
            rcols = st.columns(len(rents))
            for col_,( label,rent) in zip(rcols,rents):
                with col_:
                    st.metric(label, f'${rent:,.0f}/mo')

        # CollegeHouse insight
        pb = ch.get('purpose_built_beds',0) or 0
        excess = ch.get('estimated_excess',0) or 0
        pl_str = f'{pl:.1%}' if pl is not None else 'N/A'
        demand_str = 'strong' if pl and pl > 0.75 else 'moderate'
        st.markdown(f'<div class="ib">CollegeHouse tracks {pb:,} purpose-built beds across {ch.get("property_count",0)} properties near campus. Estimated excess supply of {excess:,} beds. Pre-lease at {pl_str} suggests {demand_str} forward demand for next lease cycle. Source: CollegeHouse {ch.get("source","")}</div>',unsafe_allow_html=True)

    st.markdown('<div class="sh">Investment Score Components</div>',unsafe_allow_html=True)
    lr  = sp.iloc[-1]
    cs  = st.columns(4)
    for col_,tup in zip(cs,[
        ('pct_ug_off_campus','Demand Pressure','% of UG body off-campus',C['GREEN']),
        ('retention_rate',   'Retention Signal','1-yr retention rate',   C['BLUE']),
        ('pct_oos_ug',       'OOS Indicator',  'Out-of-state share',    C['AMBER']),
        ('pct_need_met',     'Affordability',  '% of need met',         C['PURPLE']),
    ]):
        col,label,desc,color = tup
        v = lr.get(col)
        with col_:
            if v is not None and not (isinstance(v,float) and np.isnan(v)):
                bw = int(v*100)
                st.markdown(f"""<div class="card">
                    <div style="font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:{C['MUTED']};margin-bottom:6px;">{label}</div>
                    <div style="font-size:24px;font-weight:700;color:{color};">{v:.1%}</div>
                    <div style="background:{C['BORDER']};border-radius:4px;height:5px;margin:8px 0;">
                        <div style="background:{color};border-radius:4px;height:5px;width:{bw}%;"></div>
                    </div>
                    <div style="font-size:12px;color:#4A5568;">{desc}</div>
                </div>""",unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="card">
                    <div style="font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:{C['MUTED']};margin-bottom:6px;">{label}</div>
                    <div style="font-size:24px;font-weight:700;color:{C['MUTED']};">N/A</div>
                    <div style="font-size:12px;color:#4A5568;">{desc}</div>
                </div>""",unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# MARKET RANKINGS
# ════════════════════════════════════════════════════════════════════════
elif page == 'Market Rankings':
    st.markdown('''
    <div class="bw-page-title">Market Intelligence</div>
    <div class="bw-page-sub">Power 4 Student Housing Rankings</div>
    ''', unsafe_allow_html=True)
    cds_count  = len(school_results)
    total_count= len(all_school_results)
    ipeds_count= total_count - cds_count
    st.markdown(f'''<div style="background:#111827;border:1px solid #1E2540;border-radius:8px;padding:12px 16px;margin-bottom:16px;font-size:13px;">
        <span style="color:#10B981;font-weight:600;">{cds_count} schools</span>
        <span style="color:#6B7A9E;"> have full CDS data (enrollment, OOS share, off-campus rate, financial aid) — </span>
        <span style="color:#F59E0B;font-weight:600;">{ipeds_count} schools</span>
        <span style="color:#6B7A9E;"> use IPEDS data (enrollment, retention, tuition, admissions). Off-campus rate and OOS share are estimated for IPEDS schools — replace with CDS files for precise scores.</span>
    </div>''',unsafe_allow_html=True)
    rows = []
    for sch,sr in all_school_results.items():
        rows.append({'School':sch,'Signal':sr['signal'],'Score':sr['investment_score'],
            'Off-Campus Rate':sr['pct_ug_off_campus'],'Off-Campus Demand':sr['off_campus_demand'],
            'Retention':sr['retention_rate'],'OOS Share':sr['pct_oos_ug'],
            'Tuition':sr['tuition_instate'],
            'Data Source':sr.get('data_source','CDS'),
            'Yrs':len(sr['panel']),'_color':sr['signal_color']})
    rdf = pd.DataFrame(rows).sort_values('Score',ascending=False).reset_index(drop=True)
    rdf.index += 1

    sigs = st.multiselect('Filter',['STRONG BUY','BUY','HOLD','CAUTION','AVOID'],
                          default=['STRONG BUY','BUY','HOLD','CAUTION','AVOID'])
    rdf  = rdf[rdf['Signal'].isin(sigs)]
    cmap = {'STRONG BUY':C['GREEN'],'BUY':C['TEAL'],'HOLD':C['AMBER'],'CAUTION':C['ORANGE'],'AVOID':C['RED']}

    fig = base_fig(280,legend=True,hovermode='closest')
    for sig in ['STRONG BUY','BUY','HOLD','CAUTION','AVOID']:
        sub = rdf[rdf['Signal']==sig]
        if not sub.empty:
            fig.add_trace(go.Bar(x=list(sub['School']),y=list(sub['Score']),name=sig,
                marker_color=cmap.get(sig,'gray'),hovertemplate='%{x}<br>Score: %{y:.3f}<extra></extra>'))
    xd,yd = ax(ytitle='Investment Score',tickangle=-35)
    yd['range'] = [0,1]
    fig.update_layout(barmode='stack',xaxis=xd,yaxis=yd)
    st.plotly_chart(fig,use_container_width=True)

    out = rdf.copy()
    for col,style in [('Score',None),('Off-Campus Rate','pct'),('Retention','pct'),('OOS Share','pct'),('Tuition','money')]:
        if style:
            out[col] = out[col].map(lambda x: fmt(x,style))
        else:
            out[col] = out[col].map(lambda x: f'{x:.3f}')
    out['Off-Campus Demand'] = out['Off-Campus Demand'].map(lambda x: fmt(x))
    st.dataframe(out.drop(columns=['_color']),use_container_width=True)
    st.markdown('<div class="ib">⚡ <strong>CDS</strong> = full primary source data. <strong>IPEDS</strong> = estimated off-campus rate and OOS share — upload CDS files to replace estimates with real data for any school.</div>',unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# REGRESSION RESULTS
# ════════════════════════════════════════════════════════════════════════
elif page == 'Regression Results':
    st.markdown('''
    <div class="bw-page-title">Quantitative Research</div>
    <div class="bw-page-sub">Regression Model Output</div>
    ''', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{C["MUTED"]};margin-bottom:20px;">Ridge Regression · {len(panel)} observations · Full diagnostic suite</div>',unsafe_allow_html=True)

    reg_meta = {
        'reg1': {
            'color': C['GREEN'],
            'interpretation': 'Explains what structural factors determine how large a school\'s off-campus market is. High OOS share and high need-met % both push more students off campus. High on-campus policy keeps students on campus.',
            'coef_meaning': {
                'retention_rate': 'Higher retention → slightly fewer students living off-campus (well-retained students tend to stay in campus community)',
                'pct_oos_ug':     'Higher OOS share → more off-campus demand (OOS students have no local family housing option)',
                'pct_need_met':   'Higher need met → more students can afford off-campus housing',
            }
        },
        'reg2': {
            'color': C['BLUE'],
            'interpretation': 'Explains enrollment size differences across schools. Schools with lower tuition and higher OOS share tend to be larger institutions.',
            'coef_meaning': {
                'tuition_instate': 'Higher tuition → slightly fewer total students (cost barrier)',
                'pct_oos_ug':      'Higher OOS share → larger schools (major universities attract national applicants)',
                'retention_rate':  'Higher retention → larger enrollment (fewer students drop out, building up the stock)',
            }
        },
        'reg3': {
            'color': C['PURPLE'],
            'interpretation': 'Predicts next year\'s off-campus bed demand. Current demand is the strongest predictor (autoregressive). OOS share amplifies future demand. Retention sustains it.',
            'coef_meaning': {
                'off_campus_demand': 'Current off-campus demand → best predictor of next year\'s demand (markets are sticky)',
                'retention_rate':    'Higher retention → more returning students seeking off-campus housing next year',
                'pct_oos_ug':        'Higher OOS share → stronger off-campus demand growth next year',
            }
        },
    }

    for key in ['reg1','reg2','reg3']:
        if key not in regressions: continue
        reg  = regressions[key]
        meta = reg_meta[key]
        color = meta['color']

        st.markdown(f"""<div class="card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
                <div>
                    <div style="font-size:18px;font-weight:600;color:{color};">{reg['name']}</div>
                    <div style="font-size:13px;color:{C['MUTED']};margin-top:4px;">n = {reg['n']} | DOF = {reg['dof']} | Ridge α = {reg['ridge_alpha']:.2f}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:11px;color:{C['MUTED']};text-transform:uppercase;">R²</div>
                    <div style="font-size:28px;font-weight:700;color:{color};">{reg['r2']:.4f}</div>
                </div>
            </div>
        </div>""",unsafe_allow_html=True)

        # Formula
        st.markdown(f'<div class="formula">📐 {reg["formula"]}</div>',unsafe_allow_html=True)

        # Interpretation
        st.markdown(f'<div class="ib">{meta["interpretation"]}</div>',unsafe_allow_html=True)

        # Coefficients + meaning
        col_coef, col_diag = st.columns([2,1])
        with col_coef:
            st.markdown(f'<div class="sh" style="font-size:15px;">Coefficients (standardized)</div>',unsafe_allow_html=True)
            coefs = reg['coefs']
            fig_c = base_fig(max(160,len(coefs)*50),legend=False)
            fig_c.add_trace(go.Bar(
                x=list(coefs.values()),y=list(coefs.keys()),orientation='h',
                marker_color=[C['GREEN'] if v>0 else C['RED'] for v in coefs.values()],
                hovertemplate='%{y}: %{x:.4f}<extra></extra>'))
            xd,yd = ax(xtitle='Standardized Coefficient')
            yd['autorange'] = 'reversed'
            fig_c.update_layout(xaxis=xd,yaxis=yd,margin=dict(l=12,r=12,t=12,b=12))
            st.plotly_chart(fig_c,use_container_width=True)

            st.markdown(f'<div class="sh" style="font-size:15px;">What each coefficient means</div>',unsafe_allow_html=True)
            for var,coef in coefs.items():
                meaning = meta['coef_meaning'].get(var,'')
                direction = '↑' if coef > 0 else '↓'
                dc = C['GREEN'] if coef > 0 else C['RED']
                st.markdown(f"""<div style="padding:8px 0;border-bottom:1px solid {C['BORDER']};font-size:12px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                        <span style="color:{C['TEXT']};font-weight:600;">{var}</span>
                        <span style="color:{dc};font-family:'Courier New';">{direction} {abs(coef):.4f}</span>
                    </div>
                    <div style="color:{C['MUTED']};font-size:11px;">{meaning}</div>
                </div>""",unsafe_allow_html=True)

        with col_diag:
            st.markdown(f'<div class="sh" style="font-size:15px;">Diagnostics</div>',unsafe_allow_html=True)

            # VIF
            st.markdown('<div style="font-size:12px;color:#6B7A9E;font-weight:600;margin-bottom:6px;">VIF — Multicollinearity</div>',unsafe_allow_html=True)
            st.markdown('<div style="font-size:11px;color:#4A5568;margin-bottom:8px;">Rule: &lt;5 OK | 5–10 moderate | &gt;10 high. Ridge regression handles high VIF.</div>',unsafe_allow_html=True)
            for var,v in reg['vifs'].items():
                st.markdown(f'<div style="font-size:12px;padding:4px 0;border-bottom:1px solid {C["BORDER"]};">{var}: {vif_badge(v)}</div>',unsafe_allow_html=True)

            # Condition number
            cn = reg['condition_number']
            cn_col = '#10B981' if cn < 30 else ('#F59E0B' if cn < 100 else '#EF4444')
            cn_label = '✓ OK' if cn < 30 else ('⚠ MOD' if cn < 100 else '✗ HIGH')
            st.markdown(f'<div style="margin-top:12px;font-size:12px;color:#6B7A9E;font-weight:600;">Condition Number</div>',unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:12px;padding:4px 0;"><span style="color:{cn_col}">{cn:.1f} {cn_label}</span><br><span style="font-size:11px;color:#4A5568;">Rule: &lt;30 OK | 30–100 moderate | &gt;100 severe</span></div>',unsafe_allow_html=True)

            # Breusch-Pagan
            st.markdown(f'<div style="margin-top:12px;font-size:12px;color:#6B7A9E;font-weight:600;">Breusch-Pagan (heteroskedasticity)</div>',unsafe_allow_html=True)
            if reg['bp_pval'] is not None:
                bp_ok = reg['bp_pval'] > 0.05
                bp_col = '#10B981' if bp_ok else '#F59E0B'
                bp_label = '✓ Homoskedastic' if bp_ok else '⚠ Heteroskedastic'
                st.markdown(f'<div style="font-size:12px;padding:4px 0;"><span style="color:{bp_col}">{bp_label}</span><br><span style="font-size:11px;color:#4A5568;">p = {reg["bp_pval"]:.3f} | stat = {reg["bp_stat"]:.3f}</span><br><span style="font-size:11px;color:#4A5568;">p &gt; 0.05 = no issue</span></div>',unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-size:11px;color:#4A5568;">Insufficient obs</div>',unsafe_allow_html=True)

            # Shapiro-Wilk
            st.markdown(f'<div style="margin-top:12px;font-size:12px;color:#6B7A9E;font-weight:600;">Shapiro-Wilk (residual normality)</div>',unsafe_allow_html=True)
            if reg['sw_pval'] is not None:
                sw_ok = reg['sw_pval'] > 0.05
                sw_col = '#10B981' if sw_ok else '#F59E0B'
                sw_label = '✓ Normal residuals' if sw_ok else '⚠ Non-normal'
                st.markdown(f'<div style="font-size:12px;padding:4px 0;"><span style="color:{sw_col}">{sw_label}</span><br><span style="font-size:11px;color:#4A5568;">p = {reg["sw_pval"]:.3f}</span></div>',unsafe_allow_html=True)

            # RMSE / MAE
            st.markdown(f'<div style="margin-top:12px;font-size:12px;color:#6B7A9E;font-weight:600;">Model Accuracy</div>',unsafe_allow_html=True)
            rmse_str = f'{reg["rmse"]:.4f}' if reg["rmse"] < 1 else f'{reg["rmse"]:,.0f}'
            mae_str  = f'{reg["mae"]:.4f}'  if reg["mae"]  < 1 else f'{reg["mae"]:,.0f}'
            mape_str = f'{reg["mape"]:.1f}%' if reg['mape'] else 'N/A'
            st.markdown(f'<div style="font-size:12px;padding:4px 0;color:{C["MUTED"]};">RMSE: {rmse_str}<br>MAE: {mae_str}<br>MAPE: {mape_str}</div>',unsafe_allow_html=True)

        # Marginal effects for reg3
        if key == 'reg3' and reg.get('marginal_effects'):
            st.markdown('<div class="sh" style="font-size:15px;">Marginal Effects — Beds Added per Unit Change</div>',unsafe_allow_html=True)
            me = reg['marginal_effects']; ul = reg['unit_labels']
            fig_me = base_fig(220,legend=False)
            fig_me.add_trace(go.Bar(
                x=[ul.get(k,k) for k in me],y=list(me.values()),
                marker_color=[C['GREEN'] if v>0 else C['RED'] for v in me.values()],
                hovertemplate='%{x}: %{y:+.0f} beds<extra></extra>'))
            xd,yd = ax(ytitle='Change in Off-Campus Beds')
            fig_me.update_layout(xaxis=xd,yaxis=yd)
            st.plotly_chart(fig_me,use_container_width=True)

        # DOF warning
        if reg['dof'] <= 0:
            st.markdown(f'<div class="warn">⚠ Zero degrees of freedom — model is overfit on current {reg["n"]}-observation panel. R² reflects training fit, not predictive accuracy. Confidence in coefficients will grow as more schools are added. Ridge regularization (α={reg["ridge_alpha"]:.2f}) partially mitigates this.</div>',unsafe_allow_html=True)
        elif reg['dof'] < 5:
            st.markdown(f'<div class="warn">⚠ Low degrees of freedom ({reg["dof"]}). Coefficients are directionally useful but imprecise. Add more schools to improve reliability.</div>',unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="good">✓ Sufficient degrees of freedom ({reg["dof"]}). Coefficient estimates are statistically meaningful.</div>',unsafe_allow_html=True)

        st.markdown('<br>',unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# COMPARE SCHOOLS
# ════════════════════════════════════════════════════════════════════════
elif page == 'Compare Schools':
    st.markdown('''
    <div class="bw-page-title">Comparative Analysis</div>
    <div class="bw-page-sub">Market Comparison Matrix</div>
    ''', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{C["MUTED"]};margin-bottom:20px;">Select 2 or more schools to compare side by side</div>',unsafe_allow_html=True)

    all_schools_list = sorted(all_school_results.keys())
    compare_schools = st.multiselect('Select schools to compare', all_schools_list,
                                      default=schools[:min(3,len(schools))])
    if len(compare_schools) < 2:
        st.info('Select at least 2 schools to compare.')
        st.stop()

    color_map = {sch: SCHOOL_COLORS[i % len(SCHOOL_COLORS)] for i,sch in enumerate(compare_schools)}
    # Use all_school_results so IPEDS schools can be compared too
    _sr_source = all_school_results

    # ── Investment Score Summary ─────────────────────────────────────────
    st.markdown('<div class="sh">Investment Score Comparison</div>',unsafe_allow_html=True)
    score_cols = st.columns(len(compare_schools))
    for col_,sch in zip(score_cols,compare_schools):
        sr  = _sr_source.get(sch, school_results.get(sch,{}))
        sig = sr['signal']; sco = sr['investment_score']; scol = sr['signal_color']
        with col_:
            st.markdown(f"""<div class="card" style="text-align:center">
                <div style="font-size:13px;font-weight:600;color:{C['TEXT']};margin-bottom:10px;">{sch}</div>
                <div style="background:{scol}22;border:1px solid {scol};border-radius:6px;
                            padding:6px 14px;display:inline-block;color:{scol};font-weight:700;font-size:14px;">{sig}</div>
                <div style="margin-top:8px;font-size:22px;font-weight:700;color:{color_map[sch]};">{sco:.3f}</div>
                <div style="font-size:11px;color:{C['MUTED']};">investment score</div>
            </div>""",unsafe_allow_html=True)

    # ── Key Metrics Table ────────────────────────────────────────────────
    st.markdown('<div class="sh">Side-by-Side Metrics</div>',unsafe_allow_html=True)
    metric_rows = {
        'Total Undergrads':    ('total_undergrad',    'number'),
        'Off-Campus Demand':   ('off_campus_demand',  'number'),
        'Off-Campus Rate':     ('pct_ug_off_campus',  'pct'),
        'Retention Rate':      ('retention_rate',     'pct'),
        'OOS Share':           ('pct_oos_ug',         'pct'),
        'Instate Tuition':     ('tuition_instate',    'money'),
        'Avg Aid Package':     ('avg_aid_package',    'money'),
        'Need Met %':          ('pct_need_met',       'pct'),
    }
    table_data = {'Metric': list(metric_rows.keys())}
    for sch in compare_schools:
        sr = school_results[sch]
        vals = []
        for label,(col,style) in metric_rows.items():
            v = sr.get(col)
            vals.append(fmt(v,style))
        table_data[sch] = vals
    st.dataframe(pd.DataFrame(table_data),use_container_width=True,hide_index=True)

    # ── Trend Charts ─────────────────────────────────────────────────────
    st.markdown('<div class="sh">Historical Trends — All Selected Schools</div>',unsafe_allow_html=True)

    trend_metrics = [
        ('off_campus_demand', 'Off-Campus Demand (beds)', '$,.0f' if False else ',.0f', False),
        ('pct_ug_off_campus', 'Off-Campus Rate (%)', '.0%', True),
        ('retention_rate',    'Retention Rate (%)',  '.0%', True),
        ('pct_oos_ug',        'OOS Share (%)',        '.0%', True),
        ('tuition_instate',   'Instate Tuition ($)',  '$,.0f', False),
        ('avg_aid_package',   'Avg Aid Package ($)',  '$,.0f', False),
    ]

    for row_i in range(0, len(trend_metrics), 2):
        cols_ = st.columns(2)
        for ci, (col,title,yfmt,is_pct) in enumerate(trend_metrics[row_i:row_i+2]):
            with cols_[ci]:
                fig_ = base_fig(240)
                for sch in compare_schools:
                    sp = school_results[sch]['panel']
                    if col not in sp.columns: continue
                    yrs  = list(sp['academic_year'])
                    vals = list(sp[col])
                    pairs = [(y,v) for y,v in zip(yrs,vals)
                             if v is not None and not (isinstance(v,float) and np.isnan(v))]
                    if pairs:
                        px_,pv_ = zip(*pairs)
                        fig_.add_trace(go.Scatter(x=list(px_),y=list(pv_),
                            mode='lines+markers',name=sch,
                            line=dict(color=color_map[sch],width=2.5),
                            marker=dict(size=8,color=color_map[sch])))
                xd_,yd_ = ax(yfmt=yfmt,ytitle=title)
                fig_.update_layout(xaxis=xd_,yaxis=yd_,
                    title=dict(text=title,font=dict(size=13,color=C['MUTED'])))
                st.plotly_chart(fig_,use_container_width=True)

    # ── Forecast Comparison ──────────────────────────────────────────────
    st.markdown('<div class="sh">5-Year Demand Forecast Comparison</div>',unsafe_allow_html=True)
    fig_fc = base_fig(360)
    for sch in compare_schools:
        sr = school_results[sch]
        sp = sr['panel']
        fc = sr['forecast']
        color_ = color_map[sch]
        obs_x = list(sp['academic_year'])
        obs_y = [float(v) if v is not None else None for v in sp['off_campus_demand']]
        # Observed (solid)
        fig_fc.add_trace(go.Scatter(x=obs_x,y=obs_y,mode='lines+markers',
            name=f'{sch} (observed)',line=dict(color=color_,width=2),marker=dict(size=6)))
        if fc:
            fc_x = [f['year'] for f in fc]
            fc_y = [f['pred_demand'] for f in fc]
            # Forecast (dashed)
            fig_fc.add_trace(go.Scatter(x=[obs_x[-1]]+fc_x,y=[obs_y[-1]]+fc_y,
                mode='lines+markers',name=f'{sch} (forecast)',
                line=dict(color=color_,width=2,dash='dash'),marker=dict(size=6),
                showlegend=True))
    xd_fc,yd_fc = ax(ytitle='Off-Campus Beds')
    fig_fc.update_layout(xaxis=xd_fc,yaxis=yd_fc)
    st.plotly_chart(fig_fc,use_container_width=True)

    # ── Radar / Spider Chart ─────────────────────────────────────────────
    st.markdown('<div class="sh">Multi-Dimensional Profile</div>',unsafe_allow_html=True)
    radar_cols = ['pct_ug_off_campus','retention_rate','pct_oos_ug','pct_need_met','investment_score']
    radar_labels = ['Off-Campus Rate','Retention','OOS Share','Need Met','Investment Score']

    fig_r = go.Figure()
    for sch in compare_schools:
        sr = school_results[sch]
        sp = sr['panel']
        vals = []
        for col in radar_cols:
            if col == 'investment_score':
                v = sr['investment_score']
            else:
                v = sr.get(col)
            # Normalize 0-1
            all_vals = [school_results[s].get(col,0) or 0 for s in schools if col != 'investment_score']
            if col == 'investment_score':
                all_vals = [school_results[s]['investment_score'] for s in schools]
            mn,mx = min(all_vals),max(all_vals)
            normed = ((v or 0) - mn)/(mx-mn) if mx>mn else 0.5
            vals.append(round(normed,3))
        vals.append(vals[0])  # close the polygon
        labels = radar_labels + [radar_labels[0]]
        fig_r.add_trace(go.Scatterpolar(r=vals,theta=labels,fill='toself',
            name=sch,line=dict(color=color_map[sch],width=2),
            fillcolor=color_map[sch].replace('1)','0.12)')))
    fig_r.update_layout(
        paper_bgcolor=C['CARD'],plot_bgcolor=C['CARD'],
        font=dict(family='Segoe UI',color=C['TEXT'],size=12),
        height=400,margin=dict(l=60,r=60,t=40,b=40),
        polar=dict(
            bgcolor=C['CARD'],
            radialaxis=dict(visible=True,range=[0,1],tickfont=dict(color=C['MUTED']),gridcolor='rgba(200,170,125,0.09)'),
            angularaxis=dict(tickfont=dict(color=C['TEXT']),gridcolor='rgba(200,170,125,0.09)'),
        ),
        legend=dict(orientation='h',y=-0.1,bgcolor='rgba(0,0,0,0)',font=dict(color=C['MUTED'])),
    )
    st.plotly_chart(fig_r,use_container_width=True)
    st.markdown('<div class="ib">Radar chart shows each school\'s normalized position (0–1) relative to all schools in the panel. A school closer to the outer edge on any dimension is performing better on that metric relative to peers.</div>',unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# DATA AUDIT
# ════════════════════════════════════════════════════════════════════════
elif page == 'Data Audit':
    st.markdown('''
    <div class="bw-page-title">Data Integrity</div>
    <div class="bw-page-sub">Source Validation Audit</div>
    ''', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{C["MUTED"]};margin-bottom:20px;">Validation of all school-specific historical data against expected ranges and internal consistency checks</div>',unsafe_allow_html=True)

    expected = {
        'total_undergrad':   (1000, 80000, 'Total enrollment'),
        'pct_ug_off_campus': (0.01, 0.98,  'Off-campus rate'),  # min 0.01 to accommodate Stanford (3-7% off-campus)
        'pct_ug_on_campus':  (0.02, 1.00,  'On-campus rate'),  # allow up to 100% (e.g. Stanford/Duke)
        'retention_rate':    (0.60, 1.00,  'Retention rate'),
        'pct_oos_ug':        (0.00, 0.95,  'OOS share'),
        'tuition_instate':   (1000, 80000, 'Instate tuition'),
        'avg_aid_package':   (500, 100000, 'Avg aid package'),
        'off_campus_demand': (50, 80000,   'Off-campus demand'),
    }

    total_flags = 0
    for sch in schools:
        sr = school_results[sch]
        sp = sr['panel'].sort_values('academic_year')
        flags = []

        for col,(lo,hi,label) in expected.items():
            if col not in sp.columns: continue
            for _,row in sp[[col,'academic_year']].dropna().iterrows():
                v = row[col]; yr = int(row['academic_year'])
                if not (lo <= v <= hi):
                    flags.append(('range',yr,f'{label} = {v:.4f} (expected {lo}–{hi})'))

        # YoY swing check
        for col,label in [('total_undergrad','Enrollment'),('off_campus_demand','Off-campus demand')]:
            if col not in sp.columns: continue
            vals = sp[col].dropna()
            for idx,chg in vals.pct_change().dropna().items():
                yr = int(sp.loc[idx,'academic_year'])
                if abs(chg) > 0.20:
                    key = (sch, yr, col)
                    if key in VERIFIED_SWINGS:
                        flags.append(('verified', yr, f'✅ Source-confirmed: {label} changed {chg:+.1%} YoY. {VERIFIED_SWINGS[key]}'))
                    else:
                        flags.append(('swing',yr,f'{label} changed {chg:+.1%} YoY — verify in source'))

        # Balance check
        if 'pct_ug_on_campus' in sp.columns and 'pct_ug_off_campus' in sp.columns:
            for _,row in sp.iterrows():
                on = row.get('pct_ug_on_campus'); off = row.get('pct_ug_off_campus')
                yr = int(row['academic_year'])
                if pd.notna(on) and pd.notna(off):
                    tot = on + off
                    if not (0.88 <= tot <= 1.05):  # allow gap up to 12% for students living with parents
                        flags.append(('balance',yr,f'on({on:.3f}) + off({off:.3f}) = {tot:.3f} ≠ 1.0'))

        total_flags += len(flags)
        flag_icon = '✅' if not flags else '⚠️'
        with st.expander(f'{flag_icon} {sch} — {len(sp)} years | {len(flags)} flag(s)'):
            if flags:
                for ftype,yr,msg in flags:
                    if ftype == 'verified':
                        st.markdown(f'✅ **{yr}**: {msg}')
                    elif ftype == 'range':
                        st.markdown(f'🔴 **{yr}**: {msg}')
                    elif ftype == 'swing':
                        st.markdown(f'🟡 **{yr}**: {msg}')
                    else:
                        st.markdown(f'🟠 **{yr}**: {msg}')
            else:
                st.markdown('✅ All values within expected ranges. No consistency issues detected.')

            # Data table
            show_cols = [c for c in ['academic_year','total_undergrad','pct_ug_off_campus',
                          'off_campus_demand','retention_rate','pct_oos_ug',
                          'tuition_instate','avg_aid_package','pct_need_met'] if c in sp.columns]
            disp = sp[show_cols].copy()
            for col in ['pct_ug_off_campus','retention_rate','pct_oos_ug','pct_need_met']:
                if col in disp.columns:
                    disp[col] = disp[col].map(lambda x: f'{x:.1%}' if pd.notna(x) else 'N/A')
            for col in ['total_undergrad','off_campus_demand']:
                if col in disp.columns:
                    disp[col] = disp[col].map(lambda x: f'{int(x):,}' if pd.notna(x) else 'N/A')
            for col in ['tuition_instate','avg_aid_package']:
                if col in disp.columns:
                    disp[col] = disp[col].map(lambda x: f'${int(x):,}' if pd.notna(x) else 'N/A')
            disp['academic_year'] = disp['academic_year'].astype(int)
            st.dataframe(disp,use_container_width=True,hide_index=True)

            # School-specific trends
            st.markdown('**Historical Trends (school-specific — used in forecast)**')
            trends = sr['trends']
            trend_rows = []
            for var,info in trends.items():
                if info['last_value'] is None: continue
                slope = info['slope']; n = info['n_obs']
                src = f'Linear fit ({n} pts)' if n >= 2 else 'Held constant (1 pt)'
                trend_rows.append({
                    'Variable': var,
                    'Last Value': f"{info['last_value']:.4f}",
                    'Annual Change': f'{slope:+.4f}/yr' if slope is not None else '0 (held)',
                    'Source': src,
                    'R²': f"{info['r2']:.3f}" if info['r2'] is not None else 'N/A',
                })
            if trend_rows:
                st.dataframe(pd.DataFrame(trend_rows),use_container_width=True,hide_index=True)

    unverified = sum(1 for s,ftype,yr,msg in [(s,f[0],f[1],f[2]) for s in [sch for sch in schools] for f in []] if ftype not in ['verified'])
    real_flags = [(s,f) for s in schools for sr2 in [school_results.get(s,{})] for f in []]
    # Recount properly
    unverified_count = sum(1 for ftype,*_ in [(f[0],) for sch2 in schools for f in [] ] if ftype != 'verified')
    st.markdown(f'<br><div class="{"good" if total_flags==0 else "warn"}">{"✅ All data passed validation — 0 flags across all schools." if total_flags==0 else f"⚠ {total_flags} flag(s) found — see details above. Green checkmarks indicate verified expected movements."}</div>',unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# SENSITIVITY EXPLORER
# ════════════════════════════════════════════════════════════════════════
elif page == 'Sensitivity Explorer':
    st.markdown('''
    <div class="bw-page-title">Scenario Analysis</div>
    <div class="bw-page-sub">Sensitivity Explorer</div>
    ''', unsafe_allow_html=True)

    sr  = school_results[selected]
    sp  = sr['panel'].sort_values('academic_year')
    lr  = sp.iloc[-1]

    # ── Baseline values ──────────────────────────────────────────────────
    b_ug   = float(lr.get('total_undergrad')    or 30000)
    b_oos  = float(lr.get('pct_oos_ug')         or 0.22)
    b_ret  = float(lr.get('retention_rate')     or 0.92)
    b_on   = float(lr.get('pct_ug_on_campus')   or 0.38)
    b_off  = 1.0 - b_on
    b_dem  = int(b_ug * b_off)
    b_rent = float(zillow_data.get(selected, {}).get('latest_rent') or 1500)

    # ── Session state: saved scenarios ───────────────────────────────────
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = {}

    # ── Mode toggle ───────────────────────────────────────────────────────
    mode_col, _ = st.columns([2, 3])
    with mode_col:
        mode = st.radio('Mode', ['Scenario Builder', 'Reverse Mode'],
                        horizontal=True, label_visibility='collapsed')

    st.markdown(
        f'<div style="height:1px;background:{C["BORDER"]};margin:4px 0 20px;"></div>',
        unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # SCENARIO BUILDER MODE
    # ════════════════════════════════════════════════════════════════════
    if mode == 'Scenario Builder':

        ctrl_col, results_col = st.columns([1, 2])

        with ctrl_col:
            st.markdown(
                f'<div style="color:{C["MUTED"]};font-size:11px;text-transform:uppercase;'
                f'letter-spacing:.1em;margin-bottom:16px;">Adjust Inputs — {selected}</div>',
                unsafe_allow_html=True)

            e_d  = st.slider('Enrollment growth (%)',    -15, 25,  0, 1)
            n_d  = st.slider('On-campus rate shift (pp)', -20, 20,  0, 1,
                             help='Negative = more students move off campus')
            o_d  = st.slider('OOS share shift (pp)',      -10, 15,  0, 1)
            r_d  = st.slider('Retention shift (pp)',       -8,  8,  0, 1)
            st.markdown(
                f'<div style="height:1px;background:{C["BORDER"]};margin:12px 0;"></div>',
                unsafe_allow_html=True)
            default_rg = int(round(zillow_data.get(selected, {}).get('rent_trend_pct', 3) or 3))
            rg_d = st.slider('Annual rent growth (%)', -5, 15, default_rg, 1)

            st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
            sc_name = st.text_input('Scenario name', placeholder='e.g. Bull Case')
            if st.button('Save Scenario', use_container_width=True):
                if sc_name:
                    st.session_state.scenarios[sc_name] = dict(
                        e_d=e_d, n_d=n_d, o_d=o_d, r_d=r_d, rg_d=rg_d, school=selected)
                    st.success(f'Saved "{sc_name}"')

            if st.session_state.scenarios:
                st.markdown(
                    f'<div style="color:{C["MUTED"]};font-size:11px;text-transform:uppercase;'
                    f'letter-spacing:.1em;margin:12px 0 8px;">Saved Scenarios</div>',
                    unsafe_allow_html=True)
                for sc_n, sc_v in list(st.session_state.scenarios.items()):
                    sc_row1, sc_row2 = st.columns([3, 1])
                    with sc_row1:
                        st.markdown(
                            f'<div style="color:{C["IVORY"]};font-size:12px;padding:4px 0;">'
                            f'{sc_n}<span style="color:{C["MUTED"]};font-size:10px;"> '
                            f'({sc_v.get("school","?")})</span></div>',
                            unsafe_allow_html=True)
                    with sc_row2:
                        if st.button('X', key=f'del_{sc_n}'):
                            del st.session_state.scenarios[sc_n]
                            st.rerun()

        # ── Compute scenario values ───────────────────────────────────────
        new_ug  = b_ug  * (1 + e_d / 100)
        new_on  = min(0.99, max(0.01, b_on  + n_d / 100))
        new_off = 1.0 - new_on
        new_oos = min(1.0,  max(0.0,  b_oos + o_d / 100))
        new_ret = min(1.0,  max(0.5,  b_ret + r_d / 100))
        new_dem = int(new_ug * new_off)
        d_delta = new_dem - b_dem
        d_color = C['GREEN'] if d_delta >= 0 else C['RED']

        # 5-year demand trajectory
        import numpy as _np
        from scipy import stats as _stats
        enroll_vals = sp[['academic_year', 'total_undergrad']].dropna()
        if len(enroll_vals) >= 3:
            _sl, _ic, _, _, _ = _stats.linregress(
                enroll_vals['academic_year'].values,
                _np.log(enroll_vals['total_undergrad'].values))
            base_enroll_trend = (_np.exp(_sl) - 1) * 100
        else:
            base_enroll_trend = 0.0

        years_fwd     = list(range(1, 6))
        yr_labels     = [f'Yr {y}' for y in years_fwd]
        base_traj     = [int(b_ug  * (1 + base_enroll_trend / 100) ** y * b_off)  for y in years_fwd]
        sc_traj       = [int(new_ug * (1 + base_enroll_trend / 100) ** y * new_off) for y in years_fwd]
        base_rent_trend = zillow_data.get(selected, {}).get('rent_trend_pct', 3) or 3
        rent_traj_base  = [b_rent * (1 + base_rent_trend / 100) ** y for y in years_fwd]
        rent_traj_sc    = [b_rent * (1 + rg_d / 100) ** y             for y in years_fwd]

        with results_col:

            # ── KPI hero row ──────────────────────────────────────────────
            h1, h2, h3, h4 = st.columns(4)

            def _kpi(col, label, base_val, sc_val, fmt_fn, invert=False):
                delta  = sc_val - base_val
                is_pos = (delta >= 0) if not invert else (delta <= 0)
                dc     = C['GREEN'] if (delta != 0 and is_pos) else (C['RED'] if delta != 0 else C['MUTED'])
                sign   = '+' if delta > 0 else ''
                col.markdown(
                    f'<div class="card" style="padding:14px 16px;">'
                    f'<div style="font-size:10px;color:{C["MUTED"]};text-transform:uppercase;'
                    f'letter-spacing:.08em;margin-bottom:6px;">{label}</div>'
                    f'<div style="font-size:11px;color:{C["MUTED"]};margin-bottom:2px;">Baseline</div>'
                    f'<div style="font-size:20px;font-weight:600;color:{C["TEXT"]};">{fmt_fn(base_val)}</div>'
                    f'<div style="font-size:11px;color:{C["MUTED"]};margin-top:8px;margin-bottom:2px;">Scenario</div>'
                    f'<div style="font-size:20px;font-weight:600;color:{dc};">{fmt_fn(sc_val)}</div>'
                    f'<div style="font-size:11px;color:{dc};margin-top:4px;">{sign}{fmt_fn(delta)}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

            _kpi(h1, 'Off-Campus Demand', b_dem,  new_dem,  lambda x: f'{int(x):,}')
            _kpi(h2, 'Enrollment',        b_ug,   new_ug,   lambda x: f'{int(x):,}')
            _kpi(h3, 'Off-Campus Rate',   b_off,  new_off,  lambda x: f'{x:.1%}', invert=True)
            rent5_base = b_rent * (1 + base_rent_trend / 100) ** 5
            rent5_sc   = b_rent * (1 + rg_d / 100) ** 5
            _kpi(h4, 'Rent (5yr)', rent5_base, rent5_sc, lambda x: f'${x:,.0f}')

            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

            # ── 5-year demand trajectory ──────────────────────────────────
            st.markdown(
                f'<div style="color:{C["MUTED"]};font-size:11px;text-transform:uppercase;'
                f'letter-spacing:.1em;margin-bottom:10px;">5-Year Demand Trajectory</div>',
                unsafe_allow_html=True)

            fig_traj = base_fig(220, legend=True, hovermode='x unified')
            fig_traj.add_scatter(
                x=yr_labels, y=base_traj, name='Baseline',
                line=dict(color=C['MUTED'], dash='dash', width=2),
                mode='lines+markers', marker=dict(color=C['MUTED'], size=5))
            fig_traj.add_scatter(
                x=yr_labels, y=sc_traj, name='Scenario',
                line=dict(color=C['GOLD'], width=2),
                mode='lines+markers', marker=dict(color=C['GOLD'], size=6),
                fill='tonexty', fillcolor='rgba(200,170,125,0.07)')
            xd, yd = ax(ytitle='Beds')
            fig_traj.update_layout(xaxis=xd, yaxis=yd)
            st.plotly_chart(fig_traj, use_container_width=True)

            # ── Per-input impact cards ────────────────────────────────────
            st.markdown(
                f'<div style="color:{C["MUTED"]};font-size:11px;text-transform:uppercase;'
                f'letter-spacing:.1em;margin:4px 0 12px;">Input Impact on Demand</div>',
                unsafe_allow_html=True)

            inputs = [
                ('Enrollment',      e_d, int(b_ug * (1 + e_d / 100) * b_off) - b_dem,  '%'),
                ('On-Campus Rate',  n_d, int(b_ug * (1 - min(0.99, max(0.01, b_on + n_d / 100)))) - b_dem, 'pp'),
                ('OOS Share',       o_d, int(b_ug * b_off * (1 + o_d * 0.02)) - b_dem,  'pp'),
                ('Retention',       r_d, int(b_ug * b_off * (1 + r_d * 0.005)) - b_dem, 'pp'),
            ]
            ic1, ic2 = st.columns(2)
            for idx, (inp_name, slider_val, impact, unit) in enumerate(inputs):
                col = ic1 if idx % 2 == 0 else ic2
                impact_pct = impact / b_dem * 100 if b_dem else 0
                bar_w      = min(abs(impact_pct) / 20 * 100, 100)
                bar_col    = C['GREEN'] if impact >= 0 else C['RED']
                neutral    = slider_val == 0
                disp_col   = C['MUTED'] if neutral else bar_col
                sign_s     = '+' if slider_val > 0 else ''
                sign_i     = '+' if impact > 0 else ''
                col.markdown(
                    f'<div class="card" style="padding:12px 14px;margin-bottom:8px;">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:baseline;margin-bottom:8px;">'
                    f'<div style="font-size:11px;color:{C["MUTED"]};text-transform:uppercase;'
                    f'letter-spacing:.08em;">{inp_name}</div>'
                    f'<div style="font-size:12px;font-weight:600;color:{disp_col};">'
                    f'{sign_s}{slider_val}{unit}</div></div>'
                    f'<div style="font-size:18px;font-weight:600;color:{disp_col};">'
                    f'{sign_i}{impact:,} beds</div>'
                    f'<div style="margin-top:8px;height:4px;background:{C["BORDER"]};border-radius:2px;">'
                    f'<div style="width:{bar_w}%;height:4px;background:{disp_col};'
                    f'border-radius:2px;"></div></div>'
                    f'<div style="font-size:10px;color:{disp_col};margin-top:4px;">'
                    f'{sign_i}{impact_pct:.1f}% demand change</div>'
                    f'</div>',
                    unsafe_allow_html=True)

            # ── Rent trajectory ───────────────────────────────────────────
            st.markdown(
                f'<div style="color:{C["MUTED"]};font-size:11px;text-transform:uppercase;'
                f'letter-spacing:.1em;margin:12px 0 10px;">Rent Trajectory ($/mo)</div>',
                unsafe_allow_html=True)
            fig_rent = base_fig(200, legend=True, hovermode='x unified')
            fig_rent.add_scatter(
                x=yr_labels, y=rent_traj_base, name='Market Baseline',
                line=dict(color=C['MUTED'], dash='dash', width=2),
                mode='lines+markers', marker=dict(color=C['MUTED'], size=5))
            fig_rent.add_scatter(
                x=yr_labels, y=rent_traj_sc, name='Scenario',
                line=dict(color=C['TEAL'], width=2),
                mode='lines+markers', marker=dict(color=C['TEAL'], size=6))
            xd2, yd2 = ax(ytitle='$/mo', yfmt='$,.0f')
            fig_rent.update_layout(xaxis=xd2, yaxis=yd2)
            st.plotly_chart(fig_rent, use_container_width=True)

            # ── Saved scenario comparison ─────────────────────────────────
            same_school = {k: v for k, v in st.session_state.scenarios.items()
                           if v.get('school') == selected}
            if same_school:
                st.markdown(
                    f'<div style="color:{C["MUTED"]};font-size:11px;text-transform:uppercase;'
                    f'letter-spacing:.1em;margin:12px 0 10px;">Scenario Comparison</div>',
                    unsafe_allow_html=True)
                fig_comp = base_fig(220, legend=True, hovermode='x unified')
                fig_comp.add_scatter(
                    x=yr_labels, y=base_traj, name='Baseline',
                    line=dict(color=C['MUTED'], dash='dash', width=1),
                    mode='lines+markers', marker=dict(size=4))
                comp_colors = [C['GOLD'], C['GREEN'], C['TEAL'], C['PURPLE'], C['ORANGE']]
                for ci, (sc_n, sc_v) in enumerate(same_school.items()):
                    _e       = sc_v.get('e_d', 0)
                    _n       = sc_v.get('n_d', 0)
                    _new_ug  = b_ug  * (1 + _e / 100)
                    _new_off = 1.0 - min(0.99, max(0.01, b_on + _n / 100))
                    _traj    = [int(_new_ug * (1 + base_enroll_trend / 100) ** y * _new_off)
                                for y in years_fwd]
                    fig_comp.add_scatter(
                        x=yr_labels, y=_traj, name=sc_n,
                        line=dict(color=comp_colors[ci % len(comp_colors)], width=2),
                        mode='lines+markers',
                        marker=dict(color=comp_colors[ci % len(comp_colors)], size=5))
                xd3, yd3 = ax(ytitle='Beds')
                fig_comp.update_layout(xaxis=xd3, yaxis=yd3)
                st.plotly_chart(fig_comp, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════
    # REVERSE MODE
    # ════════════════════════════════════════════════════════════════════
    else:
        st.markdown(
            f'<div class="card" style="padding:14px 16px;margin-bottom:20px;">'
            f'<div style="font-size:12px;color:{C["MUTED"]};margin-bottom:6px;">'
            f'Set a demand target and see what combination of inputs gets you there. '
            f'Useful for underwriting a minimum viable market size.</div>'
            f'<div style="font-size:11px;color:{C["DIM"]};">Baseline demand for '
            f'<strong style="color:{C["IVORY"]};">{selected}</strong>: '
            f'<strong style="color:{C["GOLD"]};">{b_dem:,} beds</strong></div>'
            f'</div>',
            unsafe_allow_html=True)

        rev_col1, rev_col2 = st.columns([1, 2])

        with rev_col1:
            st.markdown(
                f'<div style="color:{C["MUTED"]};font-size:11px;text-transform:uppercase;'
                f'letter-spacing:.1em;margin-bottom:16px;">Target</div>',
                unsafe_allow_html=True)

            target_dem = st.number_input(
                'Target off-campus demand (beds)',
                min_value=int(b_dem * 0.5),
                max_value=int(b_dem * 3.0),
                value=int(b_dem * 1.20),
                step=500,
                help='The minimum demand you need for a deal to underwrite')

            req_pct = (target_dem - b_dem) / b_dem * 100 if b_dem else 0
            t_color = C['GREEN'] if target_dem > b_dem else C['AMBER']
            sign_t  = '+' if req_pct >= 0 else ''
            st.markdown(
                f'<div style="margin:12px 0 20px;">'
                f'<div style="font-size:11px;color:{C["MUTED"]};">Requires</div>'
                f'<div style="font-size:24px;font-weight:600;color:{t_color};">{sign_t}{req_pct:.1f}%</div>'
                f'<div style="font-size:11px;color:{C["MUTED"]};">demand growth from baseline</div>'
                f'</div>',
                unsafe_allow_html=True)

            st.markdown(
                f'<div style="color:{C["MUTED"]};font-size:11px;text-transform:uppercase;'
                f'letter-spacing:.1em;margin-bottom:12px;">Lever Limits</div>',
                unsafe_allow_html=True)
            max_enroll = st.slider('Max enrollment growth (%)', 0, 25, 10, 1)
            max_off    = st.slider('Max on-campus rate shift (pp)', -20, 0, -10, 1,
                                   help='Negative = more students off-campus')
            max_oos    = st.slider('Max OOS share shift (pp)', 0, 15, 5, 1)

        with rev_col2:
            gap = target_dem - b_dem

            enroll_dem_max  = int(b_ug * (1 + max_enroll / 100) * b_off)
            enroll_contrib  = enroll_dem_max - b_dem
            enroll_gap_pct  = enroll_contrib / gap * 100 if gap else 0

            rate_dem_max    = int(b_ug * (1 - min(0.99, max(0.01, b_on + max_off / 100))))
            rate_contrib    = rate_dem_max - b_dem
            rate_gap_pct    = rate_contrib / gap * 100 if gap else 0

            oos_dem_max     = int(b_ug * b_off * (1 + max_oos * 0.015))
            oos_contrib     = oos_dem_max - b_dem
            oos_gap_pct     = oos_contrib / gap * 100 if gap else 0

            combined_dem    = int(
                b_ug * (1 + max_enroll / 100) *
                (1 - min(0.99, max(0.01, b_on + max_off / 100))) *
                (1 + max_oos * 0.015))
            can_close       = combined_dem >= target_dem
            verdict_color   = C['GREEN'] if can_close else C['RED']
            verdict_text    = 'TARGET IS ACHIEVABLE' if can_close else 'TARGET NOT ACHIEVABLE'
            verdict_sub     = (
                f'Combined levers reach {combined_dem:,} beds' if can_close
                else f'Combined levers reach {combined_dem:,} beds — '
                     f'{target_dem - combined_dem:,} beds short')

            st.markdown(
                f'<div class="card" style="padding:16px;margin-bottom:16px;'
                f'border-color:{verdict_color}44;">'
                f'<div style="font-size:10px;font-weight:700;letter-spacing:.15em;'
                f'text-transform:uppercase;color:{verdict_color};margin-bottom:4px;">'
                f'{verdict_text}</div>'
                f'<div style="font-size:12px;color:{C["MUTED"]};">{verdict_sub}</div>'
                f'</div>',
                unsafe_allow_html=True)

            # Gap closure bars
            st.markdown(
                f'<div style="color:{C["MUTED"]};font-size:11px;text-transform:uppercase;'
                f'letter-spacing:.1em;margin-bottom:10px;">How Each Lever Closes the Gap</div>',
                unsafe_allow_html=True)

            levers = [
                ('Enrollment Growth',      enroll_contrib, enroll_gap_pct, C['GOLD']),
                ('On-Campus Rate Shift',   rate_contrib,   rate_gap_pct,   C['GREEN']),
                ('OOS Share Growth',       oos_contrib,    oos_gap_pct,    C['TEAL']),
            ]
            for lname, lval, lpct, lcol in levers:
                bar_w   = min(abs(lpct), 100)
                val_col = lcol if lval >= 0 else C['RED']
                sign_l  = '+' if lval >= 0 else ''
                sign_p  = '+' if lpct >= 0 else ''
                st.markdown(
                    f'<div style="margin-bottom:14px;">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
                    f'<div style="font-size:12px;color:{C["IVORY"]};">{lname}</div>'
                    f'<div style="font-size:12px;font-weight:600;color:{val_col};">'
                    f'{sign_l}{lval:,} beds '
                    f'<span style="color:{C["MUTED"]};font-size:11px;">'
                    f'({sign_p}{lpct:.0f}% of gap)</span></div></div>'
                    f'<div style="height:6px;background:{C["BORDER"]};border-radius:3px;">'
                    f'<div style="width:{bar_w}%;height:6px;background:{val_col};'
                    f'border-radius:3px;opacity:0.85;"></div></div>'
                    f'</div>',
                    unsafe_allow_html=True)

            # 5yr line chart
            st.markdown(
                f'<div style="color:{C["MUTED"]};font-size:11px;text-transform:uppercase;'
                f'letter-spacing:.1em;margin:16px 0 10px;">Baseline vs Target vs Combined Levers</div>',
                unsafe_allow_html=True)

            import numpy as _np2
            from scipy import stats as _stats2
            ev2 = sp[['academic_year', 'total_undergrad']].dropna()
            if len(ev2) >= 3:
                _sl2, _, _, _, _ = _stats2.linregress(
                    ev2['academic_year'].values,
                    _np2.log(ev2['total_undergrad'].values))
                _base_et2 = (_np2.exp(_sl2) - 1) * 100
            else:
                _base_et2 = 0.0

            yr2         = [f'Yr {y}' for y in range(1, 6)]
            base_traj2  = [int(b_ug * (1 + _base_et2 / 100) ** y * b_off) for y in range(1, 6)]
            comb_traj2  = [int(b_ug * (1 + max_enroll / 100) *
                               (1 + _base_et2 / 100) ** y *
                               (1 - min(0.99, max(0.01, b_on + max_off / 100))) *
                               (1 + max_oos * 0.015))
                           for y in range(1, 6)]
            target_line = [target_dem] * 5

            fig_rev = base_fig(240, legend=True, hovermode='x unified')
            fig_rev.add_scatter(
                x=yr2, y=base_traj2, name='Baseline',
                line=dict(color=C['MUTED'], dash='dash', width=2),
                mode='lines+markers', marker=dict(size=4))
            fig_rev.add_scatter(
                x=yr2, y=target_line, name='Target',
                line=dict(color=C['AMBER'], dash='dot', width=2),
                mode='lines')
            fig_rev.add_scatter(
                x=yr2, y=comb_traj2, name='Combined Levers',
                line=dict(color=C['GREEN'], width=2),
                mode='lines+markers', marker=dict(size=5),
                fill='tonexty' if can_close else None,
                fillcolor='rgba(110,158,110,0.07)')
            xd_r, yd_r = ax(ytitle='Beds')
            fig_rev.update_layout(xaxis=xd_r, yaxis=yd_r)
            st.plotly_chart(fig_rev, use_container_width=True)

            # Single-lever requirements
            st.markdown(
                f'<div style="color:{C["MUTED"]};font-size:11px;text-transform:uppercase;'
                f'letter-spacing:.1em;margin:4px 0 10px;">What It Takes — Single Lever</div>',
                unsafe_allow_html=True)
            if gap > 0 and b_off > 0:
                req_enroll_pct = ((target_dem / b_off) / b_ug - 1) * 100
                req_off_pp     = (target_dem / b_ug - b_off) * 100
                for rname, rval, feasible in [
                    ('Enrollment alone',    f'+{req_enroll_pct:.1f}%',       req_enroll_pct <= max_enroll),
                    ('On-campus rate alone', f'{req_off_pp:+.1f}pp off-campus', req_off_pp >= abs(max_off)),
                ]:
                    fc_col = C['GREEN'] if feasible else C['RED']
                    fc_txt = 'Within limit' if feasible else 'Exceeds limit'
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:8px 0;border-bottom:1px solid {C["BORDER"]};">'
                        f'<div style="font-size:12px;color:{C["IVORY"]};">{rname}</div>'
                        f'<div style="font-size:12px;font-weight:600;color:{fc_col};">'
                        f'{rval} '
                        f'<span style="font-size:10px;color:{fc_col};">({fc_txt})</span>'
                        f'</div></div>',
                        unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
# DATA TABLE
# ════════════════════════════════════════════════════════════════════════

elif page == 'AI Overview':
    st.markdown("""
    <div class="bw-page-title">Artificial Intelligence</div>
    <div class="bw-page-sub">Investment Analysis</div>
    """, unsafe_allow_html=True)

    ai_col1, ai_col2 = st.columns([2,1])
    with ai_col1:
        ai_schools = st.multiselect('Select Schools', sorted(all_school_results.keys()), default=schools[:1] if schools else [])
    with ai_col2:
        analysis_type = st.selectbox('Analysis Type', [
            'Investment Thesis',
            'Risk Assessment',
            'Market Comparison',
            'Full Investment Memo',
            'Due Diligence Checklist',
        ])

    st.markdown(f'<div style="color:{C["MUTED"]};font-size:12px;margin:8px 0 20px;">Select one or more schools and an analysis type. The AI will synthesize all model data into a professional investment assessment using only real data from the model.</div>', unsafe_allow_html=True)

    if ai_schools and st.button('GENERATE ANALYSIS', type='primary'):
        school_contexts = []
        for sch in ai_schools:
            sr = all_school_results.get(sch, {})
            ch = ch_data.get(sch, {}) if ch_data else {}
            z  = zillow_data.get(sch, {}) if zillow_data else {}
            from collegehouse_loader import get_supply_score_ch, supply_signal_ch

            lines = [
                f"SCHOOL: {sch}",
                f"Investment Signal: {sr.get('signal','N/A')} | Score: {sr.get('investment_score',0):.3f}/1.000",
                f"Off-Campus Demand: {int(sr.get('off_campus_demand') or 0):,} beds",
                f"Off-Campus Rate: {sr.get('pct_ug_off_campus',0):.1%}",
                f"Retention Rate: {sr.get('retention_rate',0):.1%}",
                f"Out-of-State Share: {sr.get('pct_oos_ug',0):.1%}",
                f"In-State Tuition: ${sr.get('tuition_instate') or 0:,}",
                f"Avg Aid Package: ${sr.get('avg_aid_package') or 0:,}",
                f"Data Source: {sr.get('data_source','CDS')} ({len(sr.get('panel',[]))} years)",
            ]
            if ch and 'error' not in ch:
                sc  = get_supply_score_ch(ch)
                sig,_ = supply_signal_ch(sc)
                sc_str = f'{sc:.3f}' if sc is not None else 'N/A'
                occ = ch.get('occupancy_rate') or 0
                pl2 = ch.get('pre_lease_rate') or 0
                lines += [
                    f"Supply Signal: {sig} (score: {sc_str})",
                    f"Occupancy: {occ:.1%}",
                    f"Pre-Lease: {pl2:.1%}",
                    f"Bed-to-Student Ratio: {ch.get('bed_to_student_ratio','N/A')}",
                    f"Market Saturation: {ch.get('market_saturation','N/A')}",
                    f"Avg Rent/Bed: ${ch.get('avg_rent_per_bed',0):,.0f}/mo",
                    f"Under Construction: {ch.get('beds_under_construction',0):,} beds",
                ]
            if z:
                latest = z.get('latest_rent') or 0
                momentum = z.get('momentum_label','N/A')
                # Get actual rent values for last 3 years only - avoid trend calculation anomalies
                annual = z.get('rent_annual_avg', {}) or {}
                recent_yrs = sorted(annual.keys())[-3:] if annual else []
                rent_history = ', '.join([f"{yr}: ${annual[yr]:,.0f}" for yr in recent_yrs])
                lines += [
                    f"Current Rent (Zillow ZORI): ${latest:,.0f}/mo",
                    f"Recent Rent History: {rent_history}",
                    f"Rent Momentum Signal: {momentum}",
                ]
            school_contexts.append("\n".join(lines))

        type_instructions = {
            'Investment Thesis':      "Write a concise investment thesis for this student housing market. Cover demand fundamentals, supply dynamics, and rent trajectory. End with a clear BUY/HOLD/AVOID recommendation with conviction level.",
            'Risk Assessment':        "Identify and assess the top 5 risks for investing in this student housing market. For each risk cite the specific data point that supports or mitigates it. Be honest about data gaps.",
            'Market Comparison':      "Compare these markets side by side. Identify which represents the strongest near-term investment opportunity and explain specifically what differentiates them.",
            'Full Investment Memo':   "Write a full investment committee memo. Sections: Executive Summary, Market Overview, Demand Analysis, Supply Analysis, Rent Analysis, Risk Factors, Recommendation. Use professional REPE language.",
            'Due Diligence Checklist':"Generate a prioritized due diligence checklist. For each item explain why it matters given the specific data shown and what you expect to find.",
        }

        prompt = (
            "You are a senior real estate private equity analyst specializing in student housing investments. "
            "You have been given proprietary market data from a quantitative investment screening model.\n\n"
            + type_instructions[analysis_type]
            + "\n\nMARKET DATA:\n\n"
            + "\n\n---\n\n".join(school_contexts)
            + "\n\nWrite in a professional but direct tone appropriate for an investment committee. "
            "Use specific numbers from the data. Do not make generic statements that could apply to any market. "
            "Format with headers where appropriate."
        )

        with st.spinner('Generating analysis...'):
            try:
                import requests as _req
                try:
                    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
                except Exception:
                    api_key = ""
                resp = _req.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
                    json={
                        "model": "claude-sonnet-4-5",
                        "max_tokens": 1500,
                        "messages": [{"role": "user", "content": prompt}]
                    },
                    timeout=45
                )
                ai_text = resp.json().get('content', [{}])[0].get('text', 'No response received.')

                st.markdown(
                    '<div style="background:#0E0E10;border:1px solid rgba(200,170,125,0.27);'
                    'border-top:2px solid #C8AA7D;padding:28px;margin-top:20px;">'
                    '<div style="font-size:9px;font-weight:700;letter-spacing:.25em;'
                    'text-transform:uppercase;color:#C8AA7D;margin-bottom:16px;'
                    'font-family:Manrope,sans-serif;">'
                    + analysis_type.upper() + ' \u2014 ' + ' + '.join(ai_schools)
                    + '</div></div>',
                    unsafe_allow_html=True
                )
                st.markdown(ai_text)
                st.download_button(
                    label='DOWNLOAD ANALYSIS',
                    data=ai_text,
                    file_name='Beechwood_' + '_'.join(ai_schools) + '_' + analysis_type.replace(' ','_') + '.txt',
                    mime='text/plain',
                )
            except Exception as e:
                st.error(f'Analysis failed: {e}. Please try again.')

    elif not ai_schools:
        st.markdown(f'<div class="ib">Select at least one school above then click Generate Analysis.</div>', unsafe_allow_html=True)


elif page == 'Data Table':
    st.markdown('''
    <div class="bw-page-title">Research Data</div>
    <div class="bw-page-sub">Panel Dataset</div>
    ''', unsafe_allow_html=True)
    sf = st.multiselect('Filter schools',schools,default=schools)
    filtered = panel[panel['school'].isin(sf)] if sf else panel
    show = [c for c in ['school','academic_year','total_undergrad','total_grad','off_campus_demand',
            'pct_ug_off_campus','pct_ug_on_campus','pct_ftfy_on_campus','retention_rate',
            'grad_rate_6yr','pct_oos_ug','pct_need_met','tuition_instate','tuition_oos',
            'on_campus_housing','off_campus_housing','rent_premium','avg_aid_package',
            'avg_need_grant','avg_need_loan','pct_any_loan','avg_debt','transfer_enrolled',
            'ftfy_enrolled','total_applicants','admission_rate','yield_rate'] if c in filtered.columns]
    st.dataframe(filtered[show],use_container_width=True,hide_index=True)
    st.download_button('⬇ Download CSV',filtered[show].to_csv(index=False),'p4_panel.csv','text/csv')

elif page == 'Market Map':
    st.markdown('''
    <div class="bw-page-title">Market Map</div>
    <div class="bw-page-sub">Geographic Intelligence</div>
    ''', unsafe_allow_html=True)

    import plotly.graph_objects as go

    # ── School metadata (conference-corrected) ─────────────────────────────
    SCHOOL_META = {
        'BostonCollege':        dict(lat=42.3355, lon=-71.1685, city='Chestnut Hill, MA', state='MA', conference='ACC',     region='Northeast'),
        'Clemson':              dict(lat=34.6834, lon=-82.8374, city='Clemson, SC',        state='SC', conference='ACC',     region='Southeast'),
        'Duke':                 dict(lat=36.0014, lon=-78.9382, city='Durham, NC',          state='NC', conference='ACC',     region='Southeast'),
        'GeorgiaTech':          dict(lat=33.7756, lon=-84.3963, city='Atlanta, GA',         state='GA', conference='ACC',     region='Southeast'),
        'Louisville':           dict(lat=38.2086, lon=-85.7585, city='Louisville, KY',      state='KY', conference='ACC',     region='Southeast'),
        'Miami':                dict(lat=25.7217, lon=-80.2684, city='Coral Gables, FL',    state='FL', conference='ACC',     region='Southeast'),
        'NCState':              dict(lat=35.7872, lon=-78.6819, city='Raleigh, NC',          state='NC', conference='ACC',     region='Southeast'),
        'Pittsburgh':           dict(lat=40.4444, lon=-79.9608, city='Pittsburgh, PA',      state='PA', conference='ACC',     region='Northeast'),
        'SMU':                  dict(lat=32.8410, lon=-96.7842, city='Dallas, TX',           state='TX', conference='ACC',     region='South'),
        'Stanford':             dict(lat=37.4275, lon=-122.1697,city='Stanford, CA',        state='CA', conference='ACC',     region='West'),
        'Syracuse':             dict(lat=43.0390, lon=-76.1350, city='Syracuse, NY',        state='NY', conference='ACC',     region='Northeast'),
        'UCBerkeley':           dict(lat=37.8724, lon=-122.2595,city='Berkeley, CA',        state='CA', conference='ACC',     region='West'),
        'UNC':                  dict(lat=35.9049, lon=-79.0469, city='Chapel Hill, NC',     state='NC', conference='ACC',     region='Southeast'),
        'UniversityOfMaryland': dict(lat=38.9869, lon=-76.9426, city='College Park, MD',   state='MD', conference='Big Ten', region='Northeast'),
        'UVA':                  dict(lat=38.0336, lon=-78.5080, city='Charlottesville, VA', state='VA', conference='ACC',     region='Southeast'),
        'VirginiaTech':         dict(lat=37.2284, lon=-80.4234, city='Blacksburg, VA',      state='VA', conference='ACC',     region='Southeast'),
        'WakeForest':           dict(lat=36.1340, lon=-80.2773, city='Winston-Salem, NC',   state='NC', conference='ACC',     region='Southeast'),
        'FloridaState':         dict(lat=30.4417, lon=-84.2979, city='Tallahassee, FL',      state='FL', conference='ACC',      region='Southeast'),
        # Big 12
        'Arizona':              dict(lat=32.2319, lon=-110.9501,city='Tucson, AZ',          state='AZ', conference='Big 12',   region='West'),
        'ArizonaState':         dict(lat=33.4255, lon=-111.9400,city='Tempe, AZ',           state='AZ', conference='Big 12',   region='West'),
        'Baylor':               dict(lat=31.5493, lon=-97.1467, city='Waco, TX',            state='TX', conference='Big 12',   region='South'),
        'BYU':                  dict(lat=40.2518, lon=-111.6493,city='Provo, UT',           state='UT', conference='Big 12',   region='West'),
        'Cincinnati':           dict(lat=39.1329, lon=-84.5150, city='Cincinnati, OH',      state='OH', conference='Big 12',   region='Midwest'),
        'ColoradoBoulder':      dict(lat=40.0076, lon=-105.2659,city='Boulder, CO',         state='CO', conference='Big 12',   region='West'),
        'Houston':              dict(lat=29.7199, lon=-95.3422, city='Houston, TX',         state='TX', conference='Big 12',   region='South'),
        'IowaState':            dict(lat=42.0267, lon=-93.6465, city='Ames, IA',            state='IA', conference='Big 12',   region='Midwest'),
        'Kansas':               dict(lat=38.9543, lon=-95.2558, city='Lawrence, KS',        state='KS', conference='Big 12',   region='Midwest'),
        'KansasState':          dict(lat=39.1836, lon=-96.5717, city='Manhattan, KS',       state='KS', conference='Big 12',   region='Midwest'),
        'OklahomaState':        dict(lat=36.1156, lon=-97.0584, city='Stillwater, OK',      state='OK', conference='Big 12',   region='South'),
        'TCU':                  dict(lat=32.7096, lon=-97.3627, city='Fort Worth, TX',      state='TX', conference='Big 12',   region='South'),
        'TexasTech':            dict(lat=33.5843, lon=-101.8749,city='Lubbock, TX',         state='TX', conference='Big 12',   region='South'),
        'UCF':                  dict(lat=28.6024, lon=-81.2001, city='Orlando, FL',         state='FL', conference='Big 12',   region='Southeast'),
        'Utah':                 dict(lat=40.7649, lon=-111.8421,city='Salt Lake City, UT',  state='UT', conference='Big 12',   region='West'),
        'WestVirginia':         dict(lat=39.6355, lon=-79.9545, city='Morgantown, WV',      state='WV', conference='Big 12',   region='Northeast'),
    }

    # ── Signal → pin color ────────────────────────────────────────────────
    def pin_color(signal):
        s = (signal or '').upper()
        if 'STRONG BUY' in s: return '#10B981'
        if 'BUY'        in s: return '#34D399'
        if 'HOLD'       in s: return '#F59E0B'
        if 'CAUTION'    in s: return '#F97316'
        if 'AVOID'      in s: return '#EF4444'
        return '#9CA3AF'

    # ── Filters ───────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)
    all_conferences = sorted(set(m['conference'] for m in SCHOOL_META.values()))
    all_regions     = sorted(set(m['region']     for m in SCHOOL_META.values()))
    all_states      = sorted(set(m['state']       for m in SCHOOL_META.values()))

    with fc1:
        sel_conf   = st.multiselect('Conference', all_conferences, default=[])
    with fc2:
        sel_region = st.multiselect('Region',     all_regions,     default=[])
    with fc3:
        sel_state  = st.multiselect('State',      all_states,      default=[])

    # ── Build filtered school list ────────────────────────────────────────
    display_schools = [
        s for s, m in SCHOOL_META.items()
        if s in school_results
        and (not sel_conf   or m['conference'] in sel_conf)
        and (not sel_region or m['region']     in sel_region)
        and (not sel_state  or m['state']      in sel_state)
    ]

    # ── Build map traces ──────────────────────────────────────────────────
    lats, lons, colors, texts, hovers = [], [], [], [], []
    for s in display_schools:
        sr = school_results[s]
        m  = SCHOOL_META[s]
        signal = sr.get('signal', 'N/A')
        score  = sr.get('investment_score', 0) or 0
        demand = int(sr.get('off_campus_demand', 0) or 0)
        label  = s if s not in ('BostonCollege','GeorgiaTech','NCState','UCBerkeley','UniversityOfMaryland','VirginiaTech','WakeForest','FloridaState') else {
            'BostonCollege':'Boston College','GeorgiaTech':'Georgia Tech',
            'NCState':'NC State','UCBerkeley':'UC Berkeley',
            'UniversityOfMaryland':'Univ. of Maryland','VirginiaTech':'Virginia Tech','WakeForest':'Wake Forest','FloridaState':'Florida State',
            'ArizonaState':'Arizona State',
            'ColoradoBoulder':'Colorado',
            'IowaState':'Iowa State',
            'KansasState':'Kansas State',
            'OklahomaState':'Oklahoma State',
            'TexasTech':'Texas Tech',
            'WestVirginia':'West Virginia'
        }[s]
        lats.append(m['lat'])
        lons.append(m['lon'])
        colors.append(pin_color(signal))
        texts.append(label)
        hovers.append(
            f"<b>{label}</b><br>"
            f"Signal: {signal}<br>"
            f"Score: {score:.3f}<br>"
            f"Off-Campus Demand: {demand:,}<br>"
            f"Conference: {m['conference']}<br>"
            f"Location: {m['city']}"
        )

    fig = go.Figure()

    # Legend color groups
    legend_groups = {
        'Buy / Strong Buy': '#10B981',
        'Hold':             '#F59E0B',
        'Caution':          '#F97316',
        'Avoid':            '#EF4444',
    }
    for label, color in legend_groups.items():
        fig.add_trace(go.Scattergeo(
            lat=[None], lon=[None],
            mode='markers',
            marker=dict(size=12, color=color),
            name=label,
            showlegend=True,
        ))

    # Main pins
    fig.add_trace(go.Scattergeo(
        lat=lats, lon=lons,
        mode='markers+text',
        text=texts,
        textposition='top center',
        textfont=dict(size=11, color='#F5F1EA', family='Manrope, sans-serif'),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hovers,
        marker=dict(
            size=14,
            color=colors,
            line=dict(width=1.5, color='#1A1A2E'),
            symbol='circle',
        ),
        showlegend=False,
    ))

    fig.update_layout(
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            showland=True,   landcolor='#1E1E2E',
            showlakes=True,  lakecolor='#0E0E18',
            showcoastlines=True, coastlinecolor='#2D2D3F',
            showsubunits=True,   subunitcolor='#2D2D3F',
            bgcolor='#0E0E18',
        ),
        paper_bgcolor='#0E0E18',
        plot_bgcolor='#0E0E18',
        font=dict(color='#F5F1EA', family='Manrope, sans-serif'),
        margin=dict(l=0, r=0, t=10, b=0),
        height=560,
        legend=dict(
            orientation='h',
            yanchor='bottom', y=0.02,
            xanchor='center', x=0.5,
            bgcolor='rgba(30,30,46,0.85)',
            bordercolor='#3A3028',
            borderwidth=1,
            font=dict(size=12, color='#F5F1EA'),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Inline Compare ────────────────────────────────────────────────────
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,rgba(200,170,125,0.20),transparent);margin-bottom:16px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sh">Compare Schools</div>', unsafe_allow_html=True)

    school_display = sorted(display_schools)
    cmp1, cmp2 = st.columns(2)
    with cmp1:
        ca = st.selectbox('School A', ['— select —'] + school_display, key='map_cmp_a')
    with cmp2:
        cb = st.selectbox('School B', ['— select —'] + school_display, key='map_cmp_b')

    if ca != '— select —' and cb != '— select —' and ca != cb:
        # ── Score cards ──────────────────────────────────────────────────
        st.markdown('<div class="sh" style="font-size:13px;margin-top:12px;">Investment Score</div>', unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        for col_, sch in [(cc1, ca), (cc2, cb)]:
            sr   = school_results[sch]
            sig  = sr['signal']; sco = sr['investment_score']; scol = sr['signal_color']
            label = sch if sch not in ('BostonCollege','GeorgiaTech','NCState','UCBerkeley','UniversityOfMaryland','VirginiaTech','WakeForest','FloridaState') else {
                'BostonCollege':'Boston College','GeorgiaTech':'Georgia Tech',
                'NCState':'NC State','UCBerkeley':'UC Berkeley',
                'UniversityOfMaryland':'Univ. of Maryland','VirginiaTech':'Virginia Tech','WakeForest':'Wake Forest','FloridaState':'Florida State',
                'ArizonaState':'Arizona State',
            'ColoradoBoulder':'Colorado',
            'IowaState':'Iowa State',
            'KansasState':'Kansas State',
            'OklahomaState':'Oklahoma State',
            'TexasTech':'Texas Tech',
            'WestVirginia':'West Virginia'}[sch]
            with col_:
                st.markdown(f"""<div class="card" style="text-align:center">
                    <div style="font-size:13px;font-weight:600;color:{C['TEXT']};margin-bottom:10px;">{label}</div>
                    <div style="background:{scol}22;border:1px solid {scol};border-radius:6px;
                                padding:6px 14px;display:inline-block;color:{scol};font-weight:700;font-size:14px;">{sig}</div>
                    <div style="margin-top:8px;font-size:26px;font-weight:700;color:{scol};">{sco:.3f}</div>
                    <div style="font-size:11px;color:{C['MUTED']};">investment score</div>
                </div>""", unsafe_allow_html=True)

        # ── Side-by-side metrics table ────────────────────────────────────
        st.markdown('<div class="sh" style="font-size:13px;margin-top:12px;">Key Metrics</div>', unsafe_allow_html=True)
        metric_rows_cmp = {
            'Total Undergrads':  ('total_undergrad',   'number'),
            'Off-Campus Demand': ('off_campus_demand', 'number'),
            'Off-Campus Rate':   ('pct_ug_off_campus', 'pct'),
            'Retention Rate':    ('retention_rate',    'pct'),
            'OOS Share':         ('pct_oos_ug',        'pct'),
            'Instate Tuition':   ('tuition_instate',   'money'),
            'Avg Aid Package':   ('avg_aid_package',   'money'),
            'Need Met %':        ('pct_need_met',      'pct'),
        }
        table_cmp = {'Metric': list(metric_rows_cmp.keys())}
        for sch in [ca, cb]:
            sr = school_results[sch]
            table_cmp[sch] = [fmt(sr.get(col), style) for _, (col, style) in metric_rows_cmp.items()]
        st.dataframe(pd.DataFrame(table_cmp), use_container_width=True, hide_index=True)

        # ── Radar chart ───────────────────────────────────────────────────
        st.markdown('<div class="sh" style="font-size:13px;margin-top:12px;">Multi-Dimensional Profile</div>', unsafe_allow_html=True)
        radar_cols_cmp   = ['pct_ug_off_campus','retention_rate','pct_oos_ug','pct_need_met','investment_score']
        radar_labels_cmp = ['Off-Campus Rate','Retention','OOS Share','Need Met','Inv. Score']
        cmp_color_map    = {ca: SCHOOL_COLORS[0], cb: SCHOOL_COLORS[1]}
        fig_r2 = go.Figure()
        for sch in [ca, cb]:
            sr = school_results[sch]
            vals = []
            for col in radar_cols_cmp:
                v = sr['investment_score'] if col == 'investment_score' else sr.get(col)
                all_v = [school_results[s]['investment_score'] if col == 'investment_score'
                         else (school_results[s].get(col) or 0) for s in schools]
                mn, mx = min(all_v), max(all_v)
                vals.append(round(((v or 0) - mn) / (mx - mn) if mx > mn else 0.5, 3))
            vals.append(vals[0])
            fig_r2.add_trace(go.Scatterpolar(
                r=vals, theta=radar_labels_cmp + [radar_labels_cmp[0]],
                fill='toself', name=sch,
                line=dict(color=cmp_color_map[sch], width=2),
                fillcolor=cmp_color_map[sch].replace('1)', '0.12)')))
        fig_r2.update_layout(
            paper_bgcolor=C['CARD'], plot_bgcolor=C['CARD'],
            font=dict(family='Segoe UI', color=C['TEXT'], size=12),
            height=360, margin=dict(l=60, r=60, t=30, b=30),
            polar=dict(
                bgcolor=C['CARD'],
                radialaxis=dict(visible=True, range=[0,1], tickfont=dict(color=C['MUTED']), gridcolor='rgba(200,170,125,0.09)'),
                angularaxis=dict(tickfont=dict(color=C['TEXT']), gridcolor='rgba(200,170,125,0.09)'),
            ),
            legend=dict(orientation='h', y=-0.12, bgcolor='rgba(0,0,0,0)', font=dict(color=C['MUTED'])),
        )
        st.plotly_chart(fig_r2, use_container_width=True)
    elif ca != '— select —' and cb != '— select —' and ca == cb:
        st.info('Select two different schools to compare.')
