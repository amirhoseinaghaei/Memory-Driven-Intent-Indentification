
import json 
# Load dataset
with open(
    "C:/Users/lilliam/Downloads/memory_project/dataset/final_mapping.json",
    "r",
    encoding="utf-8",
) as f:
    input_mapping = json.load(f)


# Load dataset
with open(
    "C:/Users/lilliam/Downloads/memory_project/dataset/disease_catalog.json",
    "r",
    encoding="utf-8",
) as f:
    disease_id_to_name = json.load(f)


# Load dataset
with open(
    "C:/Users/lilliam/Downloads/memory_project/dataset/phenotype_catalog.json",
    "r",
    encoding="utf-8",
) as f:
    phenotype_id_to_name = json.load(f)


# Load dataset
with open(
    "C:/Users/lilliam/Downloads/memory_project/dataset/anatomy_catalog.json",
    "r",
    encoding="utf-8",
) as f:
    anatomy_id_to_name = json.load(f)


with open(
    "C:/Users/lilliam/Downloads/memory_project/dataset/outlier_diseases.json",
    "r",
    encoding="utf-8",
) as f:
    outliers = json.load(f)


outliers = list(outliers.keys())


output = {}

for disease_id, pheno_dict in input_mapping.items():
    if disease_id not in outliers:
        disease_name = disease_id_to_name.get(disease_id, disease_id)

        disease_entry = {
            "name": disease_name    }

        for phenotype_id, anatomy_ids in pheno_dict.items():
            phenotype_name = phenotype_id_to_name.get(phenotype_id, phenotype_id)
            anatomy_names = [anatomy_id_to_name.get(aid, aid) for aid in anatomy_ids]

            disease_entry[phenotype_name] = anatomy_names

        output[disease_id] = disease_entry

with open("C:/Users/lilliam/Downloads/memory_project/dataset/complete_mapping.json", "w") as f:
    json.dump(output, f, indent=2)

