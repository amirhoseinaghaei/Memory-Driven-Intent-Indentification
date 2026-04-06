import pandas as pd
import json
from collections import defaultdict

# -------------------------
# Config
# -------------------------
EXCEL_PATH = "/home/amiraghaei/memory-driven-intention-identification-agent/src/disease_disease.csv"   # <-- your file
OUTPUT_JSON = "disease_hierarchy.json"

# Choose which column to use as disease id
PARENT_ID_COL = "x_id"   # or "x_index"
CHILD_ID_COL  = "y_id"   # or "y_index"

# -------------------------
# Helpers
# -------------------------
def norm_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def add_unique_child(lst, child_id: str, child_name: str):
    if not child_id or not child_name:
        return
    obj = {"id": child_id, "name": child_name}
    if obj not in lst:
        lst.append(obj)


# -------------------------
# Load Excel
# -------------------------
df = pd.read_csv(EXCEL_PATH)

required_cols = ["display_relation", "x_name", PARENT_ID_COL, "y_name", CHILD_ID_COL]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Keep only parent-child rows
df = df[df["display_relation"].str.lower() == "parent-child"]


# -------------------------
# Build parent -> children mapping
# -------------------------
parent_map = defaultdict(lambda: {
    "name": "",
    "id": "",
    "disease": []
})

for _, row in df.iterrows():
    parent_id   = norm_str(row[PARENT_ID_COL])
    parent_name = norm_str(row["x_name"])

    child_id    = norm_str(row[CHILD_ID_COL])
    child_name  = norm_str(row["y_name"])

    if not parent_id or not child_id:
        continue

    p = parent_map[parent_id]

    # set parent fields once
    if not p["id"]:
        p["id"] = parent_id
    if not p["name"]:
        p["name"] = parent_name

    add_unique_child(p["disease"], child_id, child_name)


# -------------------------
# Convert to JSON list
# -------------------------
json_list = [
    {"disease": payload}
    for payload in parent_map.values()
]

# -------------------------
# Save
# -------------------------
with open("../../dataset/" + OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(json_list, f, ensure_ascii=False, indent=2)

print(f"✅ Generated {len(json_list)} disease hierarchy objects")
