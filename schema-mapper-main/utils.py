def canonical_schema():
    return {
        "Tax ID": "Tax identifier",
        "Reg No.": "Registration number",
        "VAT#": "VAT number",
        "name": "Entity name",
        "address": "Address",
        "numeric_id": "Generic numeric id",
        "country": "Country"
    }

def simple_clean_value(v):
    try:
        if v is None:
            return v
        s = str(v).strip()
        s = " ".join(s.split())
        s = s.replace("\u2013", "-").replace("\u2014", "-")
        return s
    except Exception:
        return v

def compare_tables(df_before, df_after, max_rows=20):
    rows = min(len(df_before), max_rows)
    diffs = []
    for i in range(rows):
        before = df_before.iloc[i].to_dict()
        after = df_after.iloc[i].to_dict()
        rowdiff = {}
        for k in set(before.keys()).union(after.keys()):
            if before.get(k) != after.get(k):
                rowdiff[k] = {"before": before.get(k), "after": after.get(k)}
        if rowdiff:
            diffs.append({"row": i, "changes": rowdiff})
    return diffs
