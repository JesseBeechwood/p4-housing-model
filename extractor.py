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
    # Auto-detect numbered-table format (Big Ten style)
    if _is_table_based(filepath):
        return _extract_table_based(filepath, school_name, year)
    data = {'school': school_name, 'academic_year': year}

    # Sheet name aliases — some universities use descriptive names instead of CDS-B/CDS-F
    SHEET_ALIASES = {
        'CDS-B': ['CDS-B', 'B Enrollment', 'B. Enrollment', 'B - Enrollment',
                  'B. ENROLLMENT AND PERSISTENCE', 'Enrollment',
                  'B',                                           # Auburn single-letter
                  'B. Enrollment and Persistence',               # Mississippi State
                  'B. ENROLLMENT AND PERSISTENCE',
                  'B. Enrollment and Persistence -',             # Auburn 2023-24 trailing suffix
                  'B. Enrollment and Persistence-'],
        'CDS-F': ['CDS-F', 'F Student Life', 'F. Student Life', 'F - Student Life',
                  'F. STUDENT LIFE', 'Student Life',
                  'F',                                           # Auburn single-letter
                  'F. Student Life',                             # Mississippi State
                  'F. STUDENT LIFE',
                  'F. Student Life - 2023', 'F. Student Life - 2022',  # Auburn year-suffixed
                  'F. Student Life - 2021', 'F. Student Life - 2024'],
        'CDS-G': ['CDS-G', 'G Annual Expense', 'G. Annual Expenses', 'G - Annual Expense',
                  'G. ANNUAL EXPENSES', 'Annual Expenses',
                  'G',                                           # Auburn single-letter
                  'G. Annual Expenses and Financial Aid'],
        'CDS-H': ['CDS-H', 'H Financial Aid', 'H. Financial Aid', 'H - Financial Aid',
                  'H. FINANCIAL AID', 'Financial Aid',
                  'H',                                           # Auburn single-letter
                  'H. Financial Aid'],
    }

    def sheet(name):
        # Try exact name first, then aliases
        try:
            return pd.read_excel(filepath, sheet_name=name, header=None)
        except:
            pass
        for alias in SHEET_ALIASES.get(name, []):
            try:
                return pd.read_excel(filepath, sheet_name=alias, header=None)
            except:
                pass
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
                        if len(nums) >= 2 and sum(nums[:2]) > 3000:
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
                if ('tuition: in-district' in cl or 'tuition: in-state' in cl or
                         'in-state (out-of-district) tuition' in cl or 'in-district tuition' in cl) and 'tuition_instate' not in data:
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
                if 'percentage of need that was met' in cl or 'percent whose need was met' in cl:
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


def _is_table_based(filepath):
    """Detect files using numbered sheets (Table 1, Table 2, ...) — common Big Ten format."""
    try:
        xl = pd.ExcelFile(filepath)
        sheets = xl.sheet_names
        table_sheets = [s for s in sheets if s.strip().lower().startswith('table ')]
        return len(table_sheets) >= 5
    except:
        return False


def _extract_table_based(filepath, school_name, year):
    """
    Extract CDS data from files using numbered Table sheets.
    Scans every sheet for keyword patterns and extracts values from adjacent cells.
    Handles both clean column layouts and merged-cell formats.
    """
    data = {'school': school_name, 'academic_year': year}

    try:
        xl = pd.ExcelFile(filepath)
        sheets = xl.sheet_names
    except:
        return data

    def sf(v):
        try:
            return float(str(v).replace('$', '').replace(',', '').replace('%', '').strip())
        except:
            return None

    def nums_in_row(row, df, i):
        """Get all numeric values in a row, excluding unreasonably large ones."""
        vals = []
        for j, cell in enumerate(row):
            v = sf(cell)
            if v is not None and not (v > 1_000_000):
                vals.append((j, v))
        return vals

    # Only scan tables 1-35 — data is always within first 35 tables
    # This prevents timeouts on files with 100+ tables (Indiana has 260)
    import re as _re
    relevant_sheets = []
    for sh in sheets:
        m = _re.match(r'Table\s+(\d+)$', sh, _re.IGNORECASE)
        if m and int(m.group(1)) <= 35:
            relevant_sheets.append(sh)
        elif not sh.lower().startswith('table'):
            relevant_sheets.append(sh)

    # Load all relevant sheets in a single call — much faster than per-sheet loading
    try:
        all_sheets_raw = pd.read_excel(filepath, sheet_name=relevant_sheets,
                                       header=None, dtype=object)
        all_sheets = all_sheets_raw if isinstance(all_sheets_raw, dict) else {relevant_sheets[0]: all_sheets_raw}
    except Exception:
        all_sheets = {}
        for sh in relevant_sheets:
            try:
                all_sheets[sh] = pd.read_excel(filepath, sheet_name=sh, header=None, dtype=object)
            except:
                pass

    # Track whether we've seen the F1 housing header (for unlabeled-row format like Michigan)
    _housing_header_seen = False
    _housing_header_row  = -1
    _housing_sheet       = None

    for sh, df in all_sheets.items():
        for i in range(len(df)):
            row = [df.iloc[i, j] for j in range(len(df.columns))]
            # Flatten any newline-merged cells
            row_text = ' '.join(str(c) for c in row if c is not None and str(c) != 'nan').lower()

            # ── Total undergrad enrollment ──────────────────────────────
            if 'total_undergrad' not in data:
                # Handle enrollment from racial/ethnic breakdown table (Michigan format):
                # Row label 'Total' with large numbers in columns for UG and FTFY
                if row_text.strip() == 'total' or 'total' == row_text[:5]:
                    nvs = [(j, sf(c)) for j, c in enumerate(row)
                           if sf(c) is not None and 5000 < sf(c) < 150000]
                    if nvs:
                        data['total_undergrad'] = int(max(v for _, v in nvs))

                # Handle lone large number in a row (Ohio State Table 1 format)
                if (len([c for c in row if c is not None]) == 1):
                    v = sf(row[0] if row[0] is not None else (row[1] if len(row) > 1 else None))
                    if v and 10000 < v < 100000:
                        data['total_undergrad'] = int(v)

                if 'total all undergraduates' in row_text:
                    # Prefer the largest value in the row — last column is often the grand total
                    nvs = [sf(c) for c in row if sf(c) is not None and 1000 < sf(c) < 200000]
                    if nvs:
                        data['total_undergrad'] = int(max(nvs))

                if ('total undergraduate students' in row_text or
                    'total undergraduates' in row_text) and 'total_undergrad' not in data:
                    # Sum all numeric values (men + women + other + unknown columns)
                    nvs = [sf(c) for c in row if sf(c) is not None and 100 < sf(c) < 100000]
                    if nvs and sum(nvs) > 1000:
                        data['total_undergrad'] = int(sum(nvs))
                    if 'total_undergrad' not in data:
                        # Check next row
                        if i + 1 < len(df):
                            nrow = [df.iloc[i+1, j] for j in range(len(df.columns))]
                            for cell in nrow:
                                v = sf(cell)
                                if v and 1000 < v < 100000:
                                    data['total_undergrad'] = int(v); break

                elif 'total undergraduates' in row_text and 'total_undergrad' not in data:
                    # Try summing FT men + women + PT men + women from same row
                    nvs = [(j, sf(c)) for j, c in enumerate(row) if sf(c) and 100 < sf(c) < 100000]
                    if len(nvs) >= 4:
                        total = sum(v for _, v in nvs[:4])
                        if 1000 < total < 200000:
                            data['total_undergrad'] = int(total)
                    elif len(nvs) == 1 and 1000 < nvs[0][1] < 100000:
                        data['total_undergrad'] = int(nvs[0][1])

            # ── Housing rates ────────────────────────────────────────────
            # Detect F1 housing section header (Michigan unlabeled format)
            if 'f1' in row_text and ('first-time' in row_text or 'undergraduates enrolled' in row_text) and 'student life' not in row_text:
                _housing_header_seen = True
                _housing_header_row  = i
                _housing_sheet       = sh

            # Handle unlabeled housing rows (Michigan style):
            # After the F1 header, rows of just numbers appear: on-campus row then off-campus row
            if (_housing_header_seen and sh == _housing_sheet and
                    'pct_ug_on_campus' not in data and
                    i > _housing_header_row and i <= _housing_header_row + 6):
                nvs = [(j, sf(c)) for j, c in enumerate(row)
                       if sf(c) is not None and 0.05 < sf(c) < 1.0]
                if len(nvs) >= 2 and 'live' not in row_text:
                    # Row with 2 values 0.05-1.0 and no text = on-campus rates
                    data['pct_ftfy_on_campus'] = round(nvs[0][1], 4)
                    data['pct_ug_on_campus']   = round(nvs[1][1], 4)

            if 'live in college-owned' in row_text or ('affiliated' in row_text and 'housing' in row_text and 'percent' in row_text):
                if 'pct_ug_on_campus' not in data:
                    # Try decimal format first (0.25), then percentage format (25.0)
                    nvs = [(j, sf(c)) for j, c in enumerate(row) if sf(c) is not None and 0 < sf(c) < 2.0]
                    if not nvs:
                        nvs_pct = [(j, sf(c)/100) for j, c in enumerate(row) if sf(c) is not None and 1 < sf(c) <= 100]
                        nvs = nvs_pct
                    if len(nvs) >= 2:
                        data['pct_ftfy_on_campus'] = round(nvs[0][1], 4)
                        data['pct_ug_on_campus']   = round(nvs[1][1], 4)
                    elif len(nvs) == 1 and 0 < nvs[0][1] < 1.0:
                        data['pct_ug_on_campus'] = round(nvs[0][1], 4)

            if 'live off campus or commute' in row_text:
                # Check if this is a FTFY row (F105 code) vs UG row (F113 code)
                is_ftfy_row = 'f105' in row_text or ('f10' in row_text and 'f11' not in row_text)
                is_ug_row   = 'f113' in row_text or 'f112' in row_text

                if is_ftfy_row and 'pct_ftfy_off_campus' not in data:
                    # Store FTFY separately, don't overwrite UG
                    nvs = [(j, sf(c)) for j, c in enumerate(row) if sf(c) is not None and 0 < sf(c) < 2.0]
                    if not nvs:
                        nvs = [(j, sf(c)/100) for j, c in enumerate(row) if sf(c) is not None and 1 < sf(c) <= 100]
                    if nvs:
                        data['pct_ftfy_off_campus'] = round(nvs[0][1], 4)
                elif is_ug_row or (not is_ftfy_row and 'pct_ug_off_campus' not in data):
                    nvs = [(j, sf(c)) for j, c in enumerate(row) if sf(c) is not None and 0 < sf(c) < 2.0]
                    if not nvs:
                        nvs = [(j, sf(c)/100) for j, c in enumerate(row) if sf(c) is not None and 1 < sf(c) <= 100]
                    if len(nvs) >= 2:
                        data['pct_ftfy_off_campus'] = round(nvs[0][1], 4)
                        data['pct_ug_off_campus']   = round(nvs[1][1], 4)
                    elif len(nvs) == 1 and 0 < nvs[0][1] < 1.0:
                        data['pct_ug_off_campus'] = round(nvs[0][1], 4)

            # ── OOS share ────────────────────────────────────────────────
            if 'from out of state' in row_text and 'pct_oos_ug' not in data:
                nvs = [(j, sf(c)) for j, c in enumerate(row) if sf(c) is not None and 0 < sf(c) <= 1.0]
                if len(nvs) >= 2:
                    data['pct_oos_ftfy'] = round(nvs[0][1], 4)
                    data['pct_oos_ug']   = round(nvs[1][1], 4)
                elif len(nvs) == 1:
                    data['pct_oos_ug'] = round(nvs[0][1], 4)

            # ── Retention ────────────────────────────────────────────────
            # Only look for B22 specifically (not B4-B21 graduation rate tables)
            if 'retention_rate' not in data and 'b22' in row_text:
                for lookahead in range(1, 15):
                    if i + lookahead >= len(df): break
                    nrow = [df.iloc[i + lookahead, j] for j in range(len(df.columns))]
                    nrow_text = ' '.join(str(c) for c in nrow if c is not None).lower()
                    # Stop if we hit another section
                    if any(kw in nrow_text for kw in ['c. first-time','admission','b4','b5','graduation rate']):
                        break
                    for cell in nrow:
                        v = sf(cell)
                        if v is None: continue
                        if 0.70 < v < 1.0:
                            data['retention_rate'] = round(v, 4); break
                        if 70 < v < 100:
                            data['retention_rate'] = round(v / 100, 4); break
                    if 'retention_rate' in data: break

            # ── Tuition in-state ─────────────────────────────────────────
            if 'tuition_instate' not in data and 'tuition' in row_text and 'in-state' in row_text:
                for j, cell in enumerate(row):
                    v = sf(cell)
                    if v and 1000 < v < 80000:
                        data['tuition_instate'] = int(v); break

            # ── Need met ─────────────────────────────────────────────────
            if 'pct_need_met' not in data and 'percentage of need that was met' in row_text:
                nvs = [(j, sf(c)) for j, c in enumerate(row) if sf(c) is not None and 0 < sf(c) <= 1.0]
                if nvs:
                    data['pct_need_met'] = round(nvs[0][1], 4)
                else:
                    # value might be on next row
                    if i + 1 < len(df):
                        nrow = [df.iloc[i+1, j] for j in range(len(df.columns))]
                        for cell in nrow:
                            v = sf(cell)
                            if v and 0 < v <= 1.0:
                                data['pct_need_met'] = round(v, 4); break
                            if v and 50 < v <= 100:
                                data['pct_need_met'] = round(v / 100, 4); break

            # ── Avg aid package ──────────────────────────────────────────
            if 'avg_aid_package' not in data and 'average financial aid package' in row_text:
                for j, cell in enumerate(row):
                    v = sf(cell)
                    if v and 1000 < v < 100000:
                        data['avg_aid_package'] = int(v); break
                if 'avg_aid_package' not in data and i + 1 < len(df):
                    nrow = [df.iloc[i+1, j] for j in range(len(df.columns))]
                    for cell in nrow:
                        v = sf(cell)
                        if v and 1000 < v < 100000:
                            data['avg_aid_package'] = int(v); break

    # ── Derive off-campus rate if only on-campus available ───────────────
    if 'pct_ug_off_campus' not in data and 'pct_ug_on_campus' in data:
        data['pct_ug_off_campus'] = round(1.0 - data['pct_ug_on_campus'], 4)

    # ── Compute demand ───────────────────────────────────────────────────
    if 'total_undergrad' in data and 'pct_ug_off_campus' in data:
        data['off_campus_demand'] = int(data['total_undergrad'] * data['pct_ug_off_campus'])

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
            if 'percentage of need that was met' in cl or 'percent whose need was met' in cl:
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
            continue
        try:
            rec = extract_cds(str(f), school, year)
            records.append(rec)
        except Exception:
            pass

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values(['school', 'academic_year']).reset_index(drop=True)
    return df
