"""
CDS Extraction Engine
Parses any CDS Excel file and returns a standardized dict of variables.
Handles format variations across years (2021-2026).
"""
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path


def _to_float(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(str(val).replace('$', '').replace(',', '').replace('%', '').strip())
    except:
        return None


def _search(df, keyword, val_col_offset=1, skip_keywords=None, min_val=None, max_val=None, pct=False):
    """Search dataframe for a keyword, return first numeric value found."""
    kw = keyword.lower()
    skip = [s.lower() for s in (skip_keywords or [])]
    for i, row in df.iterrows():
        for j, cell in enumerate(row):
            if not isinstance(cell, str):
                continue
            cl = cell.strip().lower()
            if kw not in cl:
                continue
            if any(s in cl for s in skip):
                continue
            for k in range(j + 1, min(j + 8, len(row))):
                v = _to_float(df.iloc[i, k])
                if v is None:
                    continue
                if pct and v > 1.0:
                    v = v / 100
                if min_val is not None and v < min_val:
                    continue
                if max_val is not None and v > max_val:
                    continue
                return v
    return None




def _is_preformatted(filepath):
    """Check if this is a pre-formatted single-sheet CDS file (like Boston College)."""
    try:
        xl = pd.ExcelFile(filepath)
        sheets = xl.sheet_names
        # Pre-formatted files have a single sheet named like CDS_YYYY-YYYY
        if len(sheets) == 1 and ('CDS_' in sheets[0] or 'CDS-' not in sheets[0]):
            df = pd.read_excel(filepath, sheet_name=sheets[0], header=None, nrows=3)
            # Check if col A has variable names and col B has values
            col_a = [str(v).lower() for v in df.iloc[:,0] if pd.notna(v)]
            if any('total_undergrad' in c or 'enrollment' in c or 'variable' in c for c in col_a):
                return True, sheets[0]
    except:
        pass
    return False, None


def extract_preformatted(filepath, school_name, year):
    """Parse pre-formatted single-sheet CDS files where col A = variable name, col B = value."""
    xl = pd.ExcelFile(filepath)
    df = pd.read_excel(filepath, sheet_name=xl.sheet_names[0], header=None)
    data = {'school': school_name, 'academic_year': year}

    mapping = {
        'total_undergrad': 'total_undergrad',
        'total_grad': 'total_grad',
        'total_headcount': 'total_headcount',
        'ftfy_enrolled': 'ftfy_enrolled',
        'ftfy_men_enrolled': '_ftfy_men',
        'ftfy_women_enrolled': '_ftfy_women',
        'bach_degrees_awarded': 'bach_degrees',
        'retention_rate': 'retention_rate',
        'grad_rate_6yr': 'grad_rate_6yr',
        'total_applicants': 'total_applicants',
        'total_admitted': 'total_admitted',
        'transfer_applicants': 'transfer_applicants',
        'transfer_admitted': 'transfer_admitted',
        'transfer_enrolled': 'transfer_enrolled',
        'pct_ftfy_on_campus': 'pct_ftfy_on_campus',
        'pct_ug_on_campus': 'pct_ug_on_campus',
        'pct_ug_off_campus': 'pct_ug_off_campus',
        'pct_oos_ftfy': 'pct_oos_ftfy',
        'pct_oos_ug': 'pct_oos_ug',
        'pct_age_25plus': 'pct_age_25plus',
        'tuition_oos (private)': 'tuition_oos',
        'tuition_instate': 'tuition_instate',
        'required_fees': 'required_fees',
        'on_campus_housing (room only)': 'on_campus_housing',
        'on_campus_fh_total': 'on_campus_fh_total',
        'pct_need_met': 'pct_need_met',
        'avg_aid_package': 'avg_aid_package',
        'avg_need_grant': 'avg_need_grant',
        'avg_need_loan (need-based only)': 'avg_need_loan',
        'avg_need_loan': 'avg_need_loan',
        'pct_any_loan (grad class)': 'pct_any_loan',
        'pct_any_loan': 'pct_any_loan',
        'avg_debt (any loan, grad class)': 'avg_debt',
        'avg_debt': 'avg_debt',
    }

    for i, row in df.iterrows():
        key = str(row.iloc[0]).strip().lower() if pd.notna(row.iloc[0]) else ''
        val = row.iloc[1] if len(row) > 1 else None
        if pd.isna(val) or str(val).strip() in ['', 'nan', 'Calculated', 'N/A', 'External data needed']:
            continue
        try:
            fval = float(str(val).replace('$','').replace(',','').replace('%','').strip())
        except:
            continue
        for k, var in mapping.items():
            if key == k or key.startswith(k + ' ') or key.startswith(k + '('):
                data[var] = fval
                break

    # Handle ftfy from men+women
    if 'ftfy_enrolled' not in data and '_ftfy_men' in data and '_ftfy_women' in data:
        data['ftfy_enrolled'] = int(data['_ftfy_men'] + data['_ftfy_women'])

    # Derived
    if 'total_applicants' in data and 'total_admitted' in data:
        data['admission_rate'] = round(data['total_admitted'] / data['total_applicants'], 4)
    if 'total_admitted' in data and 'ftfy_enrolled' in data:
        data['yield_rate'] = round(data['ftfy_enrolled'] / data['total_admitted'], 4)
    if 'total_undergrad' in data and 'pct_ug_off_campus' in data:
        data['off_campus_demand'] = int(data['total_undergrad'] * data['pct_ug_off_campus'])
    if 'pct_ug_off_campus' not in data and 'pct_ug_on_campus' in data:
        data['pct_ug_off_campus'] = round(1 - data['pct_ug_on_campus'], 4)
        if 'total_undergrad' in data:
            data['off_campus_demand'] = int(data['total_undergrad'] * data['pct_ug_off_campus'])
    if 'tuition_instate' in data and 'avg_need_grant' in data:
        data['net_price'] = data['tuition_instate'] - data['avg_need_grant']
    if 'transfer_enrolled' in data and 'total_undergrad' in data:
        data['transfer_share'] = round(data['transfer_enrolled'] / data['total_undergrad'], 4)
    if 'ftfy_enrolled' in data and 'total_undergrad' in data:
        data['ftfy_share'] = round(data['ftfy_enrolled'] / data['total_undergrad'], 4)
    if 'total_grad' in data and 'total_undergrad' in data:
        data['grad_share'] = round(data['total_grad'] / data['total_undergrad'], 4)
    if 'greek_pct_male' in data and 'greek_pct_female' in data:
        data['greek_pct_total'] = round(data['greek_pct_male'] + data['greek_pct_female'], 4)

    return {k: v for k, v in data.items() if not k.startswith('_')}

def extract_cds(filepath, school_name, year):
    # Route UC Berkeley to its specific extractor
    if 'ucberkeley' in school_name.lower().replace(' ','').replace('-',''):
        return _extract_ucb(filepath, school_name, year)
    # Auto-detect pre-formatted single-sheet files
    preformatted, _ = _is_preformatted(filepath)
    if preformatted:
        return extract_preformatted(filepath, school_name, year)
    data = {'school': school_name, 'academic_year': year}

    def sheet(name):
        try:
            return pd.read_excel(filepath, sheet_name=name, header=None)
        except:
            return pd.DataFrame()

    # ── CDS-B: Enrollment & Retention ────────────────────────────────────
    b = sheet('CDS-B')
    if not b.empty:
        # Total undergrad - handle both split male/female cols and single total col
        for i, row in b.iterrows():
            for j, cell in enumerate(row):
                if not isinstance(cell, str):
                    continue
                cl = cell.strip().lower()
                if ('total of all undergraduate students enrolled' in cl or
                        'total of all undergraduate students' in cl):
                    # This is the definitive total — always prefer this over split rows
                    for k in range(j+1, min(j+8, len(row))):
                        v = _to_float(b.iloc[i, k])
                        if v and 5000 < v < 100000:
                            data['total_undergrad'] = int(v)
                            break
                elif ('total all undergraduates' in cl or
                      'total undergraduate students' in cl) and 'total_undergrad' not in data:
                    nums = []
                    for k in range(j+1, min(j+8, len(row))):
                        v = _to_float(b.iloc[i, k])
                        if v and 1000 < v < 100000:
                            nums.append(v)
                    if nums:
                        # If multiple large numbers it is split by gender — sum them
                        if len(nums) >= 2 and sum(nums[:2]) > 20000:
                            data['total_undergrad'] = int(sum(nums[:2]))
                        else:
                            data['total_undergrad'] = int(nums[0])
                        break
                if 'grand total all students' in cl:
                    for k in range(j+1, min(j+8, len(row))):
                        v = _to_float(b.iloc[i, k])
                        if v and v > 5000:
                            data['total_headcount'] = int(v)
                            break
                if 'total all graduate' in cl:
                    for k in range(j+1, min(j+8, len(row))):
                        v = _to_float(b.iloc[i, k])
                        if v and 100 < v < 50000:
                            data['total_grad'] = int(v)
                            break
                if "bachelor's degrees" in cl and 'bach_degrees' not in data:
                    for k in range(j+1, min(j+8, len(row))):
                        v = _to_float(b.iloc[i, k])
                        if v and 100 < v < 30000:
                            data['bach_degrees'] = int(v)
                            break

        # Retention - look for percentage or raw rate
        for i, row in b.iterrows():
            for j, cell in enumerate(row):
                if not isinstance(cell, str):
                    continue
                cl = cell.strip().lower()
                if ('calculate the percentage' in cl or 'enter retention rate' in cl or
                        ('b22' in cl and 'percentage' in cl)):
                    for k in range(j+1, min(j+8, len(row))):
                        v = _to_float(b.iloc[i, k])
                        if v is None:
                            continue
                        if 0.5 < v < 1.0:
                            data['retention_rate'] = round(v, 4)
                            break
                        elif 50 < v < 100:
                            data['retention_rate'] = round(v/100, 4)
                            break
                # Also check adjacent cell directly after B22 label
                if cl == 'b22':
                    for k in range(j+1, min(j+10, len(row))):
                        v = _to_float(b.iloc[i, k])
                        if v and 0.5 < v < 1.0:
                            data['retention_rate'] = round(v, 4)
                            break
                        elif v and 50 < v < 100:
                            data['retention_rate'] = round(v/100, 4)
                            break

        # Graduation rate 6yr
        for i, row in b.iterrows():
            for j, cell in enumerate(row):
                if not isinstance(cell, str):
                    continue
                if 'six-year graduation rate' in cell.lower() and 'grad_rate_6yr' not in data:
                    for k in range(j+1, min(j+8, len(row))):
                        v = _to_float(b.iloc[i, k])
                        if v is None:
                            continue
                        if 0.3 < v < 1.0:
                            data['grad_rate_6yr'] = round(v, 4)
                            break
                        elif 30 < v < 100:
                            data['grad_rate_6yr'] = round(v/100, 4)
                            break

    # ── CDS-C: Admissions ────────────────────────────────────────────────
    c = sheet('CDS-C')
    if not c.empty:
        app_m = app_f = adm_m = adm_f = enr_m = enr_f = None
        for i, row in c.iterrows():
            for j, cell in enumerate(row):
                if not isinstance(cell, str):
                    continue
                cl = cell.strip().lower()
                if 'first-time, first-year males who applied' in cl:
                    for k in range(j+1, min(j+6, len(row))):
                        v = _to_float(c.iloc[i, k])
                        if v and v > 100:
                            app_m = int(v); break
                elif 'first-time, first-year females who applied' in cl:
                    for k in range(j+1, min(j+6, len(row))):
                        v = _to_float(c.iloc[i, k])
                        if v and v > 100:
                            app_f = int(v); break
                elif 'first-time, first-year males who were admitted' in cl:
                    for k in range(j+1, min(j+6, len(row))):
                        v = _to_float(c.iloc[i, k])
                        if v and v > 100:
                            adm_m = int(v); break
                elif 'first-time, first-year females who were admitted' in cl:
                    for k in range(j+1, min(j+6, len(row))):
                        v = _to_float(c.iloc[i, k])
                        if v and v > 100:
                            adm_f = int(v); break
                elif ('first-time, first-year males who enrolled' in cl and
                      'full' not in cl and 'part' not in cl):
                    for k in range(j+1, min(j+6, len(row))):
                        v = _to_float(c.iloc[i, k])
                        if v and v > 10:
                            enr_m = int(v); break
                elif ('first-time, first-year females who enrolled' in cl and
                      'full' not in cl and 'part' not in cl):
                    for k in range(j+1, min(j+6, len(row))):
                        v = _to_float(c.iloc[i, k])
                        if v and v > 10:
                            enr_f = int(v); break

        if app_m and app_f:
            data['total_applicants'] = app_m + app_f
        if adm_m and adm_f:
            data['total_admitted'] = adm_m + adm_f
        if enr_m and enr_f:
            data['ftfy_enrolled'] = enr_m + enr_f
        if 'total_applicants' in data and 'total_admitted' in data:
            data['admission_rate'] = round(data['total_admitted'] / data['total_applicants'], 4)
        if 'total_admitted' in data and 'ftfy_enrolled' in data:
            data['yield_rate'] = round(data['ftfy_enrolled'] / data['total_admitted'], 4)

    # ── CDS-D: Transfers ─────────────────────────────────────────────────
    d = sheet('CDS-D')
    if not d.empty:
        for i, row in d.iterrows():
            for j, cell in enumerate(row):
                if isinstance(cell, str) and 'total' in cell.strip().lower():
                    nums = []
                    for k in range(j+1, min(j+8, len(row))):
                        v = _to_float(d.iloc[i, k])
                        if v and v > 50:
                            nums.append(int(v))
                    if len(nums) >= 3:
                        data['transfer_applicants'] = nums[0]
                        data['transfer_admitted'] = nums[1]
                        data['transfer_enrolled'] = nums[2]
                        break

    # ── CDS-F: Student Life ──────────────────────────────────────────────
    f = sheet('CDS-F')
    if not f.empty:
        for i, row in f.iterrows():
            for j, cell in enumerate(row):
                if not isinstance(cell, str):
                    continue
                cl = cell.strip().lower()
                if 'live in college-owned' in cl or 'college-owned, -operated' in cl:
                    vals = []
                    for k in range(j+1, min(j+7, len(row))):
                        v = _to_float(f.iloc[i, k])
                        if v is not None and str(f.iloc[i, k]) not in ['nan', '']:
                            fv = v/100 if v > 1 else v
                            if 0 <= fv <= 1.0:  # allow 1.0 (e.g. Duke: 100% FTFY required on campus)
                                vals.append(fv)
                    if len(vals) >= 2:
                        data['pct_ftfy_on_campus'] = round(vals[0], 4)
                        data['pct_ug_on_campus'] = round(vals[1], 4)
                elif 'live off campus or commute' in cl:
                    vals = []
                    for k in range(j+1, min(j+7, len(row))):
                        v = _to_float(f.iloc[i, k])
                        if v is not None and str(f.iloc[i, k]) not in ['nan', '']:
                            fv = v/100 if v > 1 else v
                            if 0 <= fv <= 1:  # allow 0 (Duke freshmen required on campus)
                                vals.append(fv)
                    if len(vals) >= 2:
                        data['pct_ftfy_off_campus'] = round(vals[0], 4)
                        data['pct_ug_off_campus'] = round(vals[1], 4)
                    elif len(vals) == 1 and vals[0] > 0:
                        data['pct_ug_off_campus'] = round(vals[0], 4)
                elif 'from out of state' in cl:
                    vals = []
                    for k in range(j+1, min(j+7, len(row))):
                        v = _to_float(f.iloc[i, k])
                        if v is not None and str(f.iloc[i, k]) not in ['nan', '']:
                            fv = v/100 if v > 1 else v
                            if 0 < fv < 1:
                                vals.append(fv)
                    if len(vals) >= 2:
                        data['pct_oos_ftfy'] = round(vals[0], 4)
                        data['pct_oos_ug'] = round(vals[1], 4)
                elif 'males who join fraternities' in cl:
                    for k in range(j+1, min(j+6, len(row))):
                        v = _to_float(f.iloc[i, k])
                        if v is not None:
                            data['greek_pct_male'] = round(v/100 if v > 1 else v, 4)
                            break
                elif 'females who join sororities' in cl:
                    for k in range(j+1, min(j+6, len(row))):
                        v = _to_float(f.iloc[i, k])
                        if v is not None:
                            data['greek_pct_female'] = round(v/100 if v > 1 else v, 4)
                            break
                elif 'students age 25' in cl:
                    for k in range(j+1, min(j+6, len(row))):
                        v = _to_float(f.iloc[i, k])
                        if v is not None:
                            data['pct_age_25plus'] = round(v/100 if v > 1 else v, 4)
                            break

    # ── CDS-G: Annual Expenses ───────────────────────────────────────────
    g = sheet('CDS-G')
    if not g.empty:
        for i, row in g.iterrows():
            for j, cell in enumerate(row):
                if not isinstance(cell, str):
                    continue
                cl = cell.strip().lower()
                if ('tuition: in-district' in cl or 'tuition: in-state' in cl) and 'tuition_instate' not in data:
                    for k in range(j+1, min(j+6, len(row))):
                        v = _to_float(g.iloc[i, k])
                        if v and 1000 < v < 80000:
                            data['tuition_instate'] = int(v); break
                elif 'tuition: out-of-state' in cl and 'tuition_oos' not in data:
                    for k in range(j+1, min(j+6, len(row))):
                        v = _to_float(g.iloc[i, k])
                        if v and 1000 < v < 100000:
                            data['tuition_oos'] = int(v); break
                elif 'required fees' in cl and 'required_fees' not in data:
                    for k in range(j+1, min(j+6, len(row))):
                        v = _to_float(g.iloc[i, k])
                        if v and 50 < v < 15000:
                            data['required_fees'] = int(v); break
                elif 'housing only (on-campus)' in cl or ('housing only' in cl and 'on-campus' in cl):
                    for k in range(j+1, min(j+6, len(row))):
                        v = _to_float(g.iloc[i, k])
                        if v and 1000 < v < 30000:
                            data['on_campus_housing'] = int(v); break
                elif 'food and housing (on-campus)' in cl or ('food and housing' in cl and 'on-campus' in cl):
                    for k in range(j+1, min(j+6, len(row))):
                        v = _to_float(g.iloc[i, k])
                        if v and 3000 < v < 50000:
                            data['on_campus_fh_total'] = int(v); break
                elif 'housing only' in cl and 'not applicable' not in cl and 'on_campus_housing' in data:
                    # This is the off-campus housing estimate
                    for k in range(j+1, min(j+6, len(row))):
                        v = _to_float(g.iloc[i, k])
                        if v and 1000 < v < 30000 and 'off_campus_housing' not in data:
                            data['off_campus_housing'] = int(v); break

    # ── CDS-H: Financial Aid ─────────────────────────────────────────────
    h = sheet('CDS-H')
    if not h.empty:
        for i, row in h.iterrows():
            for j, cell in enumerate(row):
                if not isinstance(cell, str):
                    continue
                cl = cell.strip().lower()
                if 'percentage of need that was met' in cl:
                    for k in range(j+1, min(j+8, len(row))):
                        v = _to_float(h.iloc[i, k])
                        if v is None:
                            continue
                        fv = v/100 if v > 1 else v
                        if 0.1 < fv <= 1.0:
                            data['pct_need_met'] = round(fv, 4); break
                elif 'average financial aid package' in cl and 'avg_aid_package' not in data:
                    for k in range(j+1, min(j+8, len(row))):
                        v = _to_float(h.iloc[i, k])
                        if v and 500 < v < 100000:
                            data['avg_aid_package'] = int(v); break
                elif 'average need-based scholarship and grant award' in cl:
                    for k in range(j+1, min(j+8, len(row))):
                        v = _to_float(h.iloc[i, k])
                        if v and 500 < v < 100000:
                            data['avg_need_grant'] = int(v); break
                elif 'average need-based loan' in cl and 'avg_need_loan' not in data:
                    for k in range(j+1, min(j+8, len(row))):
                        v = _to_float(h.iloc[i, k])
                        if v and 100 < v < 50000:
                            data['avg_need_loan'] = int(v); break
                elif 'percent of the class' in cl and 'pct_any_loan' not in data:
                    for k in range(j+1, min(j+8, len(row))):
                        v = _to_float(h.iloc[i, k])
                        if v is None:
                            continue
                        fv = v/100 if v > 1 else v
                        if 0 < fv < 1:
                            data['pct_any_loan'] = round(fv, 4); break
                elif ('cumulative principal' in cl or 'cumulative principal borrowed' in cl) and 'avg_debt' not in data:
                    for k in range(j+1, min(j+8, len(row))):
                        v = _to_float(h.iloc[i, k])
                        if v and 1000 < v < 200000:
                            data['avg_debt'] = int(v); break

    # ── Derived variables ─────────────────────────────────────────────────
    if 'total_undergrad' in data and 'pct_ug_off_campus' in data:
        data['off_campus_demand'] = int(data['total_undergrad'] * data['pct_ug_off_campus'])
    if 'pct_ug_off_campus' not in data and 'pct_ug_on_campus' in data:
        data['pct_ug_off_campus'] = round(1 - data['pct_ug_on_campus'], 4)
        if 'total_undergrad' in data:
            data['off_campus_demand'] = int(data['total_undergrad'] * data['pct_ug_off_campus'])
    if 'tuition_instate' in data and 'avg_need_grant' in data:
        data['net_price'] = data['tuition_instate'] - data['avg_need_grant']
    if 'on_campus_housing' in data and 'off_campus_housing' in data:
        data['rent_premium'] = data['off_campus_housing'] - data['on_campus_housing']
    if 'transfer_enrolled' in data and 'total_undergrad' in data:
        data['transfer_share'] = round(data['transfer_enrolled'] / data['total_undergrad'], 4)
    if 'ftfy_enrolled' in data and 'total_undergrad' in data:
        data['ftfy_share'] = round(data['ftfy_enrolled'] / data['total_undergrad'], 4)
    if 'greek_pct_male' in data and 'greek_pct_female' in data:
        data['greek_pct_total'] = round(data['greek_pct_male'] + data['greek_pct_female'], 4)
    if 'total_grad' in data and 'total_undergrad' in data:
        data['grad_share'] = round(data['total_grad'] / data['total_undergrad'], 4)

    return data


def parse_filename(filename):
    """Extract school name and year from filename like 'Alabama_CDS_2022-2023.xlsx'"""
    stem = Path(filename).stem
    # Try to find year pattern YYYY-YYYY
    m = re.search(r'(\d{4})-(\d{4})', stem)
    if m:
        year = int(m.group(2))  # use end year
        school = stem[:m.start()].strip('_- ').replace('_', ' ').replace('-', ' ').strip()
        # Remove trailing 'CDS' word if present (e.g. "University Of Maryland CDS")
        school = re.sub(r'\s*\bCDS\b\s*$', '', school, flags=re.IGNORECASE).strip()
        return school, year
    return stem, None




# ── UC Berkeley specific extractor (handles 4 different CDS formats) ──────
def _extract_ucb(filepath, school_name, year):
    """UC Berkeley uses variable tuition by cohort and has 4 different sheet formats."""
    xl = pd.ExcelFile(filepath)
    sheets = xl.sheet_names
    data = {'school': school_name, 'academic_year': year}

    def get(std, alts=[]):
        for n in [std]+alts:
            if n in sheets:
                return pd.read_excel(filepath, sheet_name=n, header=None)
        return pd.DataFrame()

    def sf(v):
        try: return float(str(v).replace('$','').replace(',','').replace('%','').strip())
        except: return None

    # B: Enrollment
    b = get('CDS-B',['B. enrollment and persistence'])
    for i, row in b.iterrows():
        for j, cell in enumerate(row):
            if not isinstance(cell,str): continue
            cl=cell.strip().lower()
            if 'total of all undergraduate students enrolled' in cl:
                for k in range(j+1,min(j+5,len(row))):
                    v=b.iloc[i,k]
                    if pd.notna(v) and isinstance(v,(int,float)) and v>5000: data['total_undergrad']=int(v); break
            elif 'total all undergraduates' in cl and 'total_undergrad' not in data:
                for k in range(j+1,min(j+5,len(row))):
                    v=b.iloc[i,k]
                    if pd.notna(v) and isinstance(v,(int,float)) and v>5000: data['total_undergrad']=int(v); break
            if ('total of all graduate' in cl) and 'total_grad' not in data:
                for k in range(j+1,min(j+5,len(row))):
                    v=b.iloc[i,k]
                    if pd.notna(v) and isinstance(v,(int,float)) and v>100: data['total_grad']=int(v); break
            if 'calculate the percentage' in cl or 'enter retention rate' in cl:
                for k in range(j+1,min(j+8,len(row))):
                    v=b.iloc[i,k]
                    fv=sf(v) if pd.notna(v) else None
                    if fv:
                        if 0.5<fv<1.0: data['retention_rate']=round(fv,4); break
                        elif 50<fv<100: data['retention_rate']=round(fv/100,4); break
            if "bachelor's degrees" in cl and 'bach_degrees' not in data:
                for k in range(j+1,min(j+5,len(row))):
                    v=b.iloc[i,k]
                    if pd.notna(v) and isinstance(v,(int,float)) and 100<v<30000: data['bach_degrees']=int(v); break

    # C: Admissions
    c=get('CDS-C',['C. FTFY admissions'])
    am=af=dm=df=em=ef=None
    for i,row in c.iterrows():
        for j,cell in enumerate(row):
            if not isinstance(cell,str): continue
            cl=cell.strip().lower()
            if 'first-time, first-year males who applied' in cl:
                for k in range(j+1,min(j+6,len(row))):
                    v=c.iloc[i,k]
                    if pd.notna(v) and isinstance(v,(int,float)) and v>100: am=int(v); break
            elif 'first-time, first-year females who applied' in cl:
                for k in range(j+1,min(j+6,len(row))):
                    v=c.iloc[i,k]
                    if pd.notna(v) and isinstance(v,(int,float)) and v>100: af=int(v); break
            elif 'first-time, first-year males who were admitted' in cl:
                for k in range(j+1,min(j+6,len(row))):
                    v=c.iloc[i,k]
                    if pd.notna(v) and isinstance(v,(int,float)) and v>10: dm=int(v); break
            elif 'first-time, first-year females who were admitted' in cl:
                for k in range(j+1,min(j+6,len(row))):
                    v=c.iloc[i,k]
                    if pd.notna(v) and isinstance(v,(int,float)) and v>10: df=int(v); break
            elif 'first-time, first-year males who enrolled' in cl and 'full' not in cl and 'part' not in cl:
                for k in range(j+1,min(j+6,len(row))):
                    v=c.iloc[i,k]
                    if pd.notna(v) and isinstance(v,(int,float)) and v>10: em=int(v); break
            elif 'first-time, first-year females who enrolled' in cl and 'full' not in cl and 'part' not in cl:
                for k in range(j+1,min(j+6,len(row))):
                    v=c.iloc[i,k]
                    if pd.notna(v) and isinstance(v,(int,float)) and v>10: ef=int(v); break
    if am and af: data['total_applicants']=am+af
    if dm and df: data['total_admitted']=dm+df
    if em and ef: data['ftfy_enrolled']=em+ef
    if 'total_applicants' in data and 'total_admitted' in data:
        data['admission_rate']=round(data['total_admitted']/data['total_applicants'],4)
    if 'total_admitted' in data and 'ftfy_enrolled' in data:
        data['yield_rate']=round(data['ftfy_enrolled']/data['total_admitted'],4)

    # D: Transfers
    d=get('CDS-D',['D. transfer admissions'])
    for i,row in d.iterrows():
        for j,cell in enumerate(row):
            if isinstance(cell,str) and 'total' in cell.lower():
                nums=[int(d.iloc[i,k]) for k in range(j+1,min(j+8,len(row)))
                      if pd.notna(d.iloc[i,k]) and isinstance(d.iloc[i,k],(int,float)) and d.iloc[i,k]>50]
                if len(nums)>=3:
                    data['transfer_applicants']=nums[0]; data['transfer_admitted']=nums[1]
                    data['transfer_enrolled']=nums[2]; break

    # F: Student Life (handles both old keyword and new F1xx question code formats)
    f=get('CDS-F',['F. student life'])
    for i,row in f.iterrows():
        row_s=[str(v) for v in row.values]
        if len(row_s)>=3:
            qn=row_s[0].strip(); vs=row_s[2].strip() if len(row_s)>2 else ''
            fmap={'F104':'pct_ftfy_on_campus','F105':'pct_ftfy_off_campus','F101':'pct_oos_ftfy',
                  'F102':'greek_pct_male','F103':'greek_pct_female','F106':'pct_age_25plus',
                  'F109':'pct_oos_ug','F112':'pct_ug_on_campus','F113':'pct_ug_off_campus','F114':'pct_age_25plus'}
            if qn in fmap:
                fv=sf(vs)
                if fv is not None: data[fmap[qn]]=round(fv/100 if fv>1 else fv,4)
        for j,cell in enumerate(row):
            if not isinstance(cell,str): continue
            cl=cell.strip().lower()
            if 'live in college-owned' in cl and 'pct_ftfy_on_campus' not in data:
                vals=[]
                for k in range(j+1,min(j+7,len(row))):
                    v=f.iloc[i,k]
                    if pd.notna(v) and str(v) not in ['nan','']:
                        fv=sf(v)
                        if fv is not None:
                            fv=fv/100 if fv>1 else fv
                            if 0<=fv<=1.0: vals.append(fv)  # allow 1.0 (100% FTFY on campus)
                if len(vals)>=2: data['pct_ftfy_on_campus']=round(vals[0],4); data['pct_ug_on_campus']=round(vals[1],4)
            elif 'live off campus or commute' in cl and 'pct_ug_off_campus' not in data:
                vals=[]
                for k in range(j+1,min(j+7,len(row))):
                    v=f.iloc[i,k]
                    if pd.notna(v) and str(v) not in ['nan','']:
                        fv=sf(v)
                        if fv is not None:
                            fv=fv/100 if fv>1 else fv
                            if 0<fv<1: vals.append(fv)
                if len(vals)>=2: data['pct_ftfy_off_campus']=round(vals[0],4); data['pct_ug_off_campus']=round(vals[1],4)
            elif 'from out of state' in cl and 'pct_oos_ug' not in data:
                vals=[]
                for k in range(j+1,min(j+7,len(row))):
                    v=f.iloc[i,k]
                    if pd.notna(v) and str(v) not in ['nan','']:
                        fv=sf(v)
                        if fv is not None:
                            fv=fv/100 if fv>1 else fv
                            if 0<fv<1: vals.append(fv)
                if len(vals)>=2: data['pct_oos_ftfy']=round(vals[0],4); data['pct_oos_ug']=round(vals[1],4)

    if 'pct_ug_off_campus' not in data and 'pct_ug_on_campus' in data:
        data['pct_ug_off_campus']=round(1-data['pct_ug_on_campus'],4)

    # G: Costs
    g=get('CDS-G',['G. annual expenses'])
    for i,row in g.iterrows():
        for j,cell in enumerate(row):
            if not isinstance(cell,str): continue
            cl=cell.strip().lower()
            if ('tuition: in-state' in cl or 'tuition: in-district' in cl) and 'tuition_instate' not in data:
                for k in range(j+1,min(j+6,len(row))):
                    v=g.iloc[i,k]
                    if pd.notna(v) and isinstance(v,(int,float)) and 1000<v<80000: data['tuition_instate']=int(v); break
            elif 'tuition: out-of-state' in cl and 'tuition_oos' not in data:
                for k in range(j+1,min(j+6,len(row))):
                    v=g.iloc[i,k]
                    if pd.notna(v) and isinstance(v,(int,float)) and 1000<v<100000: data['tuition_oos']=int(v); break
            elif 'required fees' in cl and 'required_fees' not in data:
                for k in range(j+1,min(j+6,len(row))):
                    v=g.iloc[i,k]
                    if pd.notna(v) and isinstance(v,(int,float)) and 50<v<15000: data['required_fees']=int(v); break
            elif ('room and board' in cl or 'food and housing (on-campus)' in cl) and 'on_campus_fh_total' not in data:
                for k in range(j+1,min(j+6,len(row))):
                    v=g.iloc[i,k]
                    if pd.notna(v) and isinstance(v,(int,float)) and 3000<v<50000: data['on_campus_fh_total']=int(v); break
            elif ('room only' in cl or 'housing only (on-campus)' in cl) and 'on_campus_housing' not in data:
                for k in range(j+1,min(j+6,len(row))):
                    v=g.iloc[i,k]
                    if pd.notna(v) and isinstance(v,(int,float)) and 1000<v<30000: data['on_campus_housing']=int(v); break
    # UCB cohort-based tuition fallback
    if 'tuition_instate' not in data:
        data['tuition_instate']={2023:12522,2024:13146,2025:13602,2026:14052}.get(year,14052)
        data['tuition_oos']={2023:45096,2024:47346,2025:51204,2026:52962}.get(year,52962)

    # H: Financial Aid
    h=get('CDS-H',['H. financial aid'])
    for i,row in h.iterrows():
        for j,cell in enumerate(row):
            if not isinstance(cell,str): continue
            cl=cell.strip().lower()
            if 'percentage of need that was met' in cl:
                for k in range(j+1,min(j+8,len(row))):
                    v=h.iloc[i,k]
                    if pd.notna(v) and str(v) not in ['nan','']:
                        fv=sf(v)
                        if fv:
                            fv=fv/100 if fv>1 else fv
                            if 0.1<fv<=1.0: data['pct_need_met']=round(fv,4); break
            elif 'average financial aid package' in cl and 'avg_aid_package' not in data:
                for k in range(j+1,min(j+8,len(row))):
                    v=h.iloc[i,k]
                    if pd.notna(v) and isinstance(v,(int,float)) and 500<v<100000: data['avg_aid_package']=int(v); break
            elif 'average need-based scholarship and grant award' in cl:
                for k in range(j+1,min(j+8,len(row))):
                    v=h.iloc[i,k]
                    if pd.notna(v) and isinstance(v,(int,float)) and 500<v<100000: data['avg_need_grant']=int(v); break
            elif 'average need-based loan' in cl and 'avg_need_loan' not in data:
                for k in range(j+1,min(j+8,len(row))):
                    v=h.iloc[i,k]
                    if pd.notna(v) and isinstance(v,(int,float)) and 100<v<50000: data['avg_need_loan']=int(v); break
            elif 'percent of the class' in cl and 'pct_any_loan' not in data:
                for k in range(j+1,min(j+8,len(row))):
                    v=h.iloc[i,k]
                    if pd.notna(v) and str(v) not in ['nan','']:
                        fv=sf(v)
                        if fv:
                            fv=fv/100 if fv>1 else fv
                            if 0<fv<1: data['pct_any_loan']=round(fv,4); break
            elif 'cumulative principal' in cl and 'avg_debt' not in data:
                for k in range(j+1,min(j+8,len(row))):
                    v=h.iloc[i,k]
                    if pd.notna(v) and isinstance(v,(int,float)) and 1000<v<200000: data['avg_debt']=int(v); break

    # Derived
    if 'total_undergrad' in data and 'pct_ug_off_campus' in data:
        data['off_campus_demand']=int(data['total_undergrad']*data['pct_ug_off_campus'])
    if 'tuition_instate' in data and 'avg_need_grant' in data:
        data['net_price']=data['tuition_instate']-data['avg_need_grant']
    if 'transfer_enrolled' in data and 'total_undergrad' in data:
        data['transfer_share']=round(data['transfer_enrolled']/data['total_undergrad'],4)
    if 'ftfy_enrolled' in data and 'total_undergrad' in data:
        data['ftfy_share']=round(data['ftfy_enrolled']/data['total_undergrad'],4)
    if 'greek_pct_male' in data and 'greek_pct_female' in data:
        data['greek_pct_total']=round(data['greek_pct_male']+data['greek_pct_female'],4)
    return data

def load_all_cds(folder):
    """Load all CDS files from a folder into a panel DataFrame."""
    records = []
    folder = Path(folder)
    files = list(folder.glob('*.xlsx')) + list(folder.glob('*.xls'))

    for f in sorted(files):
        school, year = parse_filename(f.name)
        if year is None:
            print(f"  Skipping {f.name} — could not parse year from filename")
            continue
        try:
            rec = extract_cds(str(f), school, year)
            records.append(rec)
            print(f"  ✓ {school} {year} — {len(rec)} variables extracted")
        except Exception as e:
            print(f"  ✗ {f.name} — ERROR: {e}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values(['school', 'academic_year']).reset_index(drop=True)
    return df
