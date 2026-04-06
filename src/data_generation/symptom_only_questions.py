import json
import time
from pathlib import Path
from tqdm import tqdm
import pandas as pd  # pip install pandas openpyxl
from neo4j import GraphDatabase  # pip install neo4j

from src.config.config import settings
from src.gen_ai_gateway.chat_completion import ChatCompletion


NEO4J_URI = "neo4j://10.213.22.232:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Aa@0022369945"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def get_descriptions(driver, id_label_list):
    """
    id_label_list: List of tuples -> [(id, label), ...]
    returns: dict -> {id: description}
    """
    query = """
    UNWIND $pairs AS pair
    MATCH (n)
    WHERE n.id = pair.id AND n.label = pair.label
    RETURN n.id AS id, n.description AS description, n.data AS data
    """

    pairs = [{"id": i, "label": l} for i, l in id_label_list]

    with driver.session() as session:
        result = session.run(query, pairs=pairs)
        return {record["description"].lower() :record["data"] for record in result}


import json

with open("/home/amiraghaei/memory-driven-intention-identification-agent/dataset/phenotype_catalog.json", "r") as f:
    ph_mapping = json.load(f)
ids = list(ph_mapping.keys())
new_ids = []
for id in ids: 
    new_ids.append((id,"phenotype"))
with open("phenotype_catalog3.json", "w") as f:
    json.dump(get_descriptions(driver, new_ids), f, indent=4)


# def parse_llm_json(raw: str) -> dict:
#     """
#     Safely parse model output that should be JSON.
#     Tries json.loads first; falls back to a light cleanup.
#     """
#     raw = (raw or "").strip()

#     # Common case: valid JSON
#     try:
#         return json.loads(raw)
#     except Exception:
#         pass

#     # Fallback: sometimes the model uses single quotes or wraps in ```json ... ```
#     cleaned = raw
#     cleaned = cleaned.replace("```json", "").replace("```", "").strip()

#     # If it looks like Python dict with single quotes, try a conservative fix
#     # (still not perfect, but safer than eval)
#     try:
#         cleaned2 = cleaned.replace("'", '"')
#         return json.loads(cleaned2)
#     except Exception:
#         return {}

# df = pd.read_excel("/home/amiraghaei/memory-driven-intention-identification-agent/src/data_generation/outliers.xlsx")   # default reads first sheet

# # Load dataset
# with open(
#     "/home/amiraghaei/memory-driven-intention-identification-agent/dataset/disease_bundle.json",
#     "r",
#     encoding="utf-8",
# ) as f:
#     data = json.load(f)

# chat = ChatCompletion(settings)

# rows = []

# out_path = Path("/home/amiraghaei/memory-driven-intention-identification-agent/test_data/symptom_only_questions_newapproach_using_dictionary.xlsx")
# out_path.parent.mkdir(parents=True, exist_ok=True)

# for idx, item in tqdm(enumerate(data, start=1)):
#     disease = item.get("disease", {})
#     # Drugs list
#     drugs = [d.get("name") for d in (disease.get("drugs") or []) if d.get("name")]
#     if (disease["id"]) not in list(df["disease_id"]):

#         phenotype_ids = (disease.get("phenotypes", []))
#         phenotype_ids = [(f"phenotype:{item["id"]}","phenotype") for item in phenotype_ids]
#         symptom_mapping = (get_descriptions(driver, phenotype_ids))
#         num_patients = 2
#         messages = [
#     {
#         "role": "system",
#         "content": f"""
# You are a medical dataset generator.

# Goal:
# Given a Symptom List for a single disease, generate multiple realistic patient stories.
# For each patient, generate a sequence of 5 patient-authored questions asked over time.

# Symptom List (choose ONLY from this list):
# The Symptom List is provided as a dictionary called symptom_mapping:
# {symptom_mapping}

# Definition (CRITICAL):
# - Each KEY in symptom_mapping is the exact symptom phrase that MUST appear in the patient's question text.
# - Each VALUE in symptom_mapping is the canonical symptom string that MUST be placed inside the "symptoms": [] array.

# Let N = number of unique KEYS in symptom_mapping.
# All symptom selection and progression rules apply to KEYS (not VALUES).

# Number of questions per patient:
# - ALWAYS set K = 5.

# Hard rules:
# 1) Create EXACTLY {num_patients} different patients (distinct voice, age range, context),
# BUT if N is too small to support {num_patients} patients without violating the rules below,
# you may reduce num_patients (never increase it).
# 2) For EACH patient, generate EXACTLY 5 questions: Q1, Q2, Q3, Q4, Q5

# 3) Symptoms per question:
# - Prefer EXACTLY 4 symptoms per question whenever possible.
# - If N < 4, a question may include fewer than 4 symptoms (use all available symptoms if needed).

# 4) Symptom progression across rounds (prioritize NEW symptoms, but handle small N):
# Define "unused symptoms" as KEYS not yet used in earlier questions for this patient.

# - Q1:
#     * If N >= 4: use EXACTLY 4 unique KEYS.
#     * If N < 4: use EXACTLY N KEYS (0..3).
# - Q2:
#     * If there are >= 4 unused KEYS: use EXACTLY 4 unused KEYS.
#     * If there are 1..3 unused KEYS: include all unused KEYS, then fill remaining slots
#       (up to 4 total) by reusing KEYS from Q1.
#     * If there are 0 unused KEYS (e.g., N < 4): reuse the same KEY set as Q1.
# - Q3:
#     * Same rule as Q2, using remaining unused KEYS first, then reuse if needed.
# - Q4:
#     * Same rule as Q2, using remaining unused KEYS first, then reuse if needed.
# - Q5:
#     * Same rule as Q2, using remaining unused KEYS first, then reuse if needed.

# 5) Reuse rule (your requirement):
# - If ANY KEY is reused from a previous question in the same patient (because N is too small),
#   the question text MUST describe the situation in a NEW way (different angle),
#   e.g., new context, timeline, severity change, triggers, duration, frequency, impact on daily life,
#   attempted remedies, new worry/question, etc.
# - The text must NOT feel like a duplicate paraphrase; it should be a realistically new follow-up message.

# 6) Symptom Mapping Requirement (CRITICAL):
# For every question:

# A) Selecting symptoms:
# - You MUST select symptoms using ONLY the KEYS of symptom_mapping.

# B) Question text (MUST contain KEYS exactly):
# - The "question" text MUST include each selected KEY EXACTLY as written (verbatim substring match).
# - Do NOT change capitalization, punctuation, hyphens, spacing, or wording of any selected KEY.
# - The KEY must appear as a contiguous sequence of characters in the question text.
# - You MAY add extra lay-language context around the KEY, but you must not alter the KEY itself.

# C) Symptoms array (MUST contain VALUES only):
# - For every KEY used in the question text, you MUST place the corresponding VALUE
#   from symptom_mapping[KEY] into the "symptoms": [] list.
# - The "symptoms" array MUST contain ONLY VALUES from symptom_mapping (canonical strings).
# - NEVER place a KEY inside the "symptoms" array.
# - NEVER place a VALUE inside the question text.
# - The VALUE used in "symptoms": [] MUST correspond exactly to the KEY used in the question.

# Example:
# If symptom_mapping contains:
#   "tummy pain": "Abdominal pain"
# Then correct output is:
#   question: "I've had tummy pain for three days, especially after meals."
#   symptoms: ["Abdominal pain"]
# Incorrect outputs include:
#   question using "Abdominal pain"
#   symptoms containing "tummy pain"

# 7) Patient consistency:
# - Questions must be first-person, natural, and consistent with the patient story.
# - Keep the same patient voice across Q1→Q5.

# 8) Do not invent symptoms not in symptom_mapping KEYS.

# Return STRICT JSON only. No markdown, no extra text.

# Output schema (STRICT):
# {{
#   "disease_id": "{disease.get("id", "")}",
#   "symptom_count": N,
#   "questions_per_patient": 5,
#   "patients": [
#     {{
#       "patient_id": "P1",
#       "qa_sequence": [
#         {{
#           "question_id": "P1_Q1",
#           "question": "...",
#           "symptoms": []
#         }},
#         {{
#           "question_id": "P1_Q2",
#           "question": "...",
#           "symptoms": []
#         }},
#         {{
#           "question_id": "P1_Q3",
#           "question": "...",
#           "symptoms": []
#         }},
#         {{
#           "question_id": "P1_Q4",
#           "question": "...",
#           "symptoms": []
#         }},
#         {{
#           "question_id": "P1_Q5",
#           "question": "...",
#           "symptoms": []
#         }}
#       ]
#     }}
#   ]
# }}
# """.strip(),
#     },
#     {
#         "role": "user",
#         "content": "Generate the patients and question sequences now."
#     }
# ]
#         # messages = [
#         #     {
#         #         "role": "system",
#         #         "content": f"""
#         # You are a medical dataset generator.

#         # Goal:
#         # Given a Symptom List for a single disease, generate multiple realistic patient stories.
#         # For each patient, generate a sequence of 5 patient-authored questions asked over time.
#         # Each question should mention lay-language phrasing / synonyms of the symptoms it includes.

#         # Symptom List (choose ONLY from this list):
#         # {symptom_mapping}

#         # Let N = number of unique symptoms in Symptom List.

#         # Number of questions per patient:
#         # - ALWAYS set K = 5.

#         # Hard rules:
#         # 1) Create EXACTLY {num_patients} different patients (distinct voice, age range, context),
#         # BUT if N is too small to support {num_patients} patients without violating the rules below,
#         # you may reduce num_patients (never increase it).
#         # 2) For EACH patient, generate EXACTLY 5 questions: Q1, Q2, Q3, Q4, Q5

#         # 3) Symptoms per question:
#         # - Prefer EXACTLY 4 symptoms per question whenever possible.
#         # - If N < 4, a question may include fewer than 4 symptoms (use all available symptoms if needed).

#         # 4) Symptom progression across rounds (prioritize NEW symptoms, but handle small N):
#         # Define "unused symptoms" as symptoms not yet used in earlier questions for this patient.

#         # - Q1:
#         #     * If N >= 4: use EXACTLY 4 unique symptoms.
#         #     * If N < 4: use EXACTLY N symptoms (0..3).
#         # - Q2:
#         #     * If there are >= 4 unused symptoms: use EXACTLY 4 unused symptoms.
#         #     * If there are 1..3 unused symptoms: include all unused symptoms, then fill remaining slots
#         #     (up to 4 total) by reusing symptoms from Q1.
#         #     * If there are 0 unused symptoms (e.g., N < 4): reuse the same symptom set as Q1.
#         # - Q3:
#         #     * Same rule as Q2, using remaining unused symptoms first, then reuse if needed.
#         # - Q4:
#         #     * Same rule as Q2, using remaining unused symptoms first, then reuse if needed.        
#         # - Q5:
#         #     * Same rule as Q2, using remaining unused symptoms first, then reuse if needed.

#         # 5) Reuse rule (your requirement):
#         # - If ANY symptom is reused from a previous question in the same patient (because N is too small),
#         #     the question text MUST describe the situation in a NEW way (different angle),
#         #     e.g., new context, timeline, severity change, triggers, duration, frequency, impact on daily life,
#         #     attempted remedies, new worry/question, etc.
#         # - The text must NOT feel like a duplicate paraphrase; it should be a realistically new follow-up message.

#         # 6) Exact-string requirement (your requirement):
#         #     - For each question, the "symptoms" array must contain the EXACT symptom strings as they appear in Symptom List (verbatim).
#         #     - In addition, the "question" text MUST include each selected symptom string EXACTLY as-is (verbatim substring match).
#         #     * Do NOT change capitalization, punctuation, hyphens, spacing, or wording of the symptom string.
#         #     * Do NOT wrap or alter the symptom string (keep it intact).
#         # 7) Patient consistency:
#         # - Questions must be first-person, natural, and consistent with the patient story.
#         # - Keep the same patient voice across Q1→Q5.

#         # 8) Do not invent symptoms not in the Symptom List.

#         # Return STRICT JSON only. No markdown, no extra text.

#         # Output schema (STRICT):
#         # {{
#         # "disease_id": "{disease.get("id", "")}",
#         # "symptom_count": N,
#         # "questions_per_patient": 5,
#         # "patients": [
#         #     {{
#         #     "patient_id": "P1",
#         #     "qa_sequence": [
#         #         {{
#         #         "question_id": "P1_Q1",
#         #         "question": "...",
#         #         "symptoms": []
#         #         }},
#         #         {{
#         #         "question_id": "P1_Q2",
#         #         "question": "...",
#         #         "symptoms": []
#         #         }},
#         #         {{
#         #         "question_id": "P1_Q3",
#         #         "question": "...",
#         #         "symptoms": []
#         #         }},
#         #         {{
#         #         "question_id": "P1_Q4",
#         #         "question": "...",
#         #         "symptoms": []
#         #         }},
#         #         {{
#         #         "question_id": "P1_Q5",
#         #         "question": "...",
#         #         "symptoms": []
#         #         }}
#         #     ]
#         #     }}
#         # ]
#         # }}
#         # """.strip(),
#         #     },
#         #     {
#         #         "role": "user",
#         #         "content": "Generate the patients and question sequences now."
#         #     }
#         # ]



#         response = chat.create_response(message=messages)
#         raw = (response.choices[0].message.content or "").strip()
#         parsed = parse_llm_json(raw)
#         print(parsed)

#         patients_data = parsed.get("patients", "") 
#         for patient in patients_data:
#             (sequence) = patient.get("qa_sequence", [])
#             row = {}
#             for i in range(len(sequence)):
#                 question = (sequence)[i].get("question", "")
#                 symptoms = (sequence)[i].get("symptoms", [])
#                 if isinstance(symptoms, list):
#                         symptoms_str = ", ".join([str(s) for s in symptoms])
#                 else:
#                         symptoms_str = str(symptoms)
#                 row[f"question{i+1}"] = question
#                 row[f"symptoms{i+1}"] = symptoms_str
#             row["drugs"] = ", ".join(drugs)
#             row["disease_name"] =  disease["name"]
#             row["disease_id"] =  disease["id"]
#             rows.append(row)
#             # Optional: save every 20 rows so you don't lose progress if something crashes
#             if idx % 20 == 0:
#                 pd.DataFrame(rows).to_excel(out_path, index=False)


# pd.DataFrame(rows).to_excel(out_path, index=False)
# print(f"Saved Excel to: {out_path}")
