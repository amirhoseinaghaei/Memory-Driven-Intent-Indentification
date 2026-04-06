import pandas as pd
import json
from collections import defaultdict

# -------------------------
# Config
# -------------------------
GENE_ANATOMY_CSV = "/home/amiraghaei/memory-driven-intention-identification-agent/src/anatomy_gene.csv"  # <-- your file
KEEP_ONLY_DISPLAY_RELATION = "expression present"  

# -------------------------
# Helpers
# -------------------------
def norm_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def ensure_cols(df, required, file_label):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{file_label}: missing columns {missing}. Found: {list(df.columns)}")

def add_unique_obj(lst, name: str, _id: str):
    if not name or not _id:
        return
    obj = {"name": name, "id": _id}
    if obj not in lst:
        lst.append(obj)

# -------------------------
# Read + validate
# -------------------------
df = pd.read_csv(GENE_ANATOMY_CSV)
ensure_cols(
    df,
    ["display_relation", "x_name", "x_id", "y_name", "y_id"],
    "Gene-Anatomy CSV"
)

# Optional filter
df = df[df["display_relation"].astype(str).str.lower() == KEEP_ONLY_DISPLAY_RELATION.lower()]

# -------------------------
# Build: gene_key -> {"name","id","anatomy":[{"name","id"},...]}
# -------------------------
gene_map = defaultdict(lambda: {"name": "", "id": "", "anatomy": []})

for _, row in df.iterrows():
    gene_name = norm_str(row["x_name"])
    gene_id   = norm_str(row["x_id"])
    anat_name = norm_str(row["y_name"])
    anat_id   = norm_str(row["y_id"])

    if not gene_name or not gene_id:
        continue

    # Use gene_id as stable key
    g = gene_map[gene_id]
    if not g["name"]:
        g["name"] = gene_name
    if not g["id"]:
        g["id"] = gene_id

    add_unique_obj(g["anatomy"], anat_name, anat_id)

# -------------------------
# Convert to requested list of JSON objects
# -------------------------
json_list = [{"gene": payload} for payload in gene_map.values()]

# Save
with open("../../dataset/gene_anatomy.json", "w", encoding="utf-8") as f:
    json.dump(json_list, f, ensure_ascii=False, indent=2)

print(f"✅ Built {len(json_list)} gene objects")
