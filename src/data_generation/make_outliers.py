import json
from pathlib import Path

input_path = Path("C:/Users/lilliam/Downloads/memory_project/dataset/final_mapping.json")
output_path = Path("C:/Users/lilliam/Downloads/memory_project/dataset/outlier_diseases.json")
with input_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

outlier_diseases = {}

for disease_id, phenotype_dict in data.items():
    # Case 1: disease has no phenotypes
    if not phenotype_dict:
        outlier_diseases[disease_id] = {
            "reason": "no_phenotypes"
        }
        continue

    # Case 2: one or more phenotypes have no anatomy
    empty_anatomy_phenotypes = [
        phenotype_id
        for phenotype_id, anatomy_list in phenotype_dict.items()
        if not anatomy_list
    ]

    if empty_anatomy_phenotypes:
        outlier_diseases[disease_id] = {
            "reason": "has_phenotype_with_no_anatomy",
            "phenotypes_with_no_anatomy": empty_anatomy_phenotypes
        }

print("Number of outlier diseases:", len(outlier_diseases))

with output_path.open("w", encoding="utf-8") as f:
    json.dump(outlier_diseases, f, indent=2)

print(f"Saved to {output_path}")