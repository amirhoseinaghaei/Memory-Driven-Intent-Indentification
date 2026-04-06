import pandas as pd
import json
from collections import defaultdict

# -------------------------
# Config: CSV file paths
# -------------------------
PHENO_CSV = "C:/Users/lilliam/Downloads/memory_project/phenotype_disease.csv"
GENE_CSV  = "C:/Users/lilliam/Downloads/memory_project/gene_disease.csv"
DRUG_CSV  = "C:/Users/lilliam/Downloads/memory_project/drug_disease.csv"
# Choose disease ID column
DISEASE_ID_COL = "y_id"     # or "y_index"


# -------------------------
# Helpers
# -------------------------
def norm_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def read_csv_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def ensure_cols(df, required, file_label):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{file_label}: missing columns {missing}. Found: {list(df.columns)}")

def add_unique_obj(lst, name: str, _id: str):
    """Store {"name":..., "id":...} uniquely."""
    if not name or not _id:
        return
    obj = {"name": name, "id": _id}
    if obj not in lst:
        lst.append(obj)


# -------------------------
# Build: disease_name -> payload
# -------------------------
disease_map = defaultdict(lambda: {"id": "", "phenotypes": [], "genes": [], "drugs": []})


# -------------------------
# 1) Phenotypes CSV
#    Keep only display_relation == "phenotype present"
# -------------------------
df_ph = read_csv_file(PHENO_CSV)
ensure_cols(df_ph, ["display_relation", "x_name", "x_id", "y_name", DISEASE_ID_COL], "Phenotype CSV")

df_ph = df_ph[df_ph["display_relation"].str.lower() == "phenotype present"]

for _, row in df_ph.iterrows():
    disease_name = norm_str(row["y_name"])
    disease_id   = norm_str(row[DISEASE_ID_COL])

    ph_name = norm_str(row["x_name"])
    ph_id   = norm_str(row["x_id"])

    if not disease_name:
        continue

    if disease_id and not disease_map[disease_name]["id"]:
        disease_map[disease_name]["id"] = disease_id

    add_unique_obj(disease_map[disease_name]["phenotypes"], ph_name, ph_id)


# -------------------------
# 2) Genes CSV (no filtering)
# -------------------------
df_ge = read_csv_file(GENE_CSV)
ensure_cols(df_ge, ["x_name", "x_id", "y_name", DISEASE_ID_COL], "Gene CSV")

for _, row in df_ge.iterrows():
    disease_name = norm_str(row["y_name"])
    disease_id   = norm_str(row[DISEASE_ID_COL])

    gene_name = norm_str(row["x_name"])
    gene_id   = norm_str(row["x_id"])

    if not disease_name:
        continue

    if disease_id and not disease_map[disease_name]["id"]:
        disease_map[disease_name]["id"] = disease_id

    add_unique_obj(disease_map[disease_name]["genes"], gene_name, gene_id)


# -------------------------
# 3) Drugs CSV
#    Keep only display_relation == "indication"
# -------------------------
df_dr = read_csv_file(DRUG_CSV)
ensure_cols(df_dr, ["display_relation", "x_name", "x_id", "y_name", DISEASE_ID_COL], "Drug CSV")

df_dr = df_dr[df_dr["display_relation"].str.lower() == "indication"]

for _, row in df_dr.iterrows():
    disease_name = norm_str(row["y_name"])
    disease_id   = norm_str(row[DISEASE_ID_COL])

    drug_name = norm_str(row["x_name"])
    drug_id   = norm_str(row["x_id"])

    if not disease_name:
        continue

    if disease_id and not disease_map[disease_name]["id"]:
        disease_map[disease_name]["id"] = disease_id

    add_unique_obj(disease_map[disease_name]["drugs"], drug_name, drug_id)


# -------------------------
# Convert to final JSON list
# Only keep diseases with >=1 phenotype AND >=1 drug
# -------------------------
json_list = []

for disease_name, payload in disease_map.items():

    if len(payload["phenotypes"]) == 0 or len(payload["drugs"]) == 0:
        continue

    json_list.append({
        "disease": {
            "name": disease_name,    # optional
            "id": payload["id"],
            "phenotypes": payload["phenotypes"],
            "genes": payload["genes"],
            "drugs": payload["drugs"],
        }
    })

# -------------------------
# Save
# -------------------------
with open("../../dataset/disease_bundle.json", "w", encoding="utf-8") as f:
    json.dump(json_list, f, ensure_ascii=False, indent=2)

print(f"✅ Final diseases kept: {len(json_list)}")
