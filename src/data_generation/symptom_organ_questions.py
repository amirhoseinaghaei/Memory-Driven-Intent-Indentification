import json
import logging
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config.config import settings
from src.gen_ai_gateway.chat_completion import ChatCompletion

# ---------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("generation_errors.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# LLM CALL CONSTANTS
# ---------------------------------------------------------
LLM_MAX_RETRIES = 3
LLM_RETRY_BASE_DELAY = 2.0   # seconds; doubles on each attempt (exponential backoff)
LLM_RETRY_MAX_DELAY = 30.0   # cap on backoff delay


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def parse_llm_json(raw: str) -> dict:
    """
    Safely parse model output that should be JSON.
    Returns an empty dict if all parsing strategies fail.
    """
    raw = (raw or "").strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    cleaned = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    try:
        cleaned2 = cleaned.replace("'", '"')
        return json.loads(cleaned2)
    except Exception:
        return {}


def call_llm_with_retry(
    chat: ChatCompletion,
    messages: list,
    disease_id: str,
    max_retries: int = LLM_MAX_RETRIES,
) -> str | None:
    """
    Call the LLM with exponential-backoff retry on transient errors.

    Returns the raw string content on success, or None if all attempts fail.
    Errors are logged at WARNING/ERROR level; nothing is raised to the caller.
    """
    delay = LLM_RETRY_BASE_DELAY

    for attempt in range(1, max_retries + 1):
        try:
            response = chat.create_response(message=messages)

            # ── Validate response structure ──────────────────────────────
            if response is None:
                raise ValueError("LLM returned None response object.")

            choices = getattr(response, "choices", None)
            if not choices:
                raise ValueError(
                    f"LLM response has no choices. Raw response: {response}"
                )

            message = getattr(choices[0], "message", None)
            if message is None:
                raise ValueError("First choice has no 'message' attribute.")

            content = getattr(message, "content", None)
            if content is None:
                raise ValueError("Message content is None.")

            raw = content.strip()
            if not raw:
                raise ValueError("LLM returned an empty response body.")

            logger.info(
                "[%s] LLM call succeeded on attempt %d/%d.",
                disease_id, attempt, max_retries,
            )
            return raw

        # ── Retriable errors (network / server / timeout) ───────────────
        except (ConnectionError, TimeoutError, OSError) as exc:
            logger.warning(
                "[%s] Network/IO error on attempt %d/%d: %s. Retrying in %.1fs …",
                disease_id, attempt, max_retries, exc, delay,
            )

        # ── Retriable: upstream API rate-limit / 5xx (often raises generic Exception) ─
        except Exception as exc:  # noqa: BLE001
            exc_str = str(exc).lower()
            is_rate_limit = any(
                kw in exc_str for kw in ("rate limit", "429", "too many requests")
            )
            is_server_err = any(
                kw in exc_str for kw in ("500", "502", "503", "504", "server error")
            )

            if is_rate_limit:
                # Back off longer for rate-limit hits
                rate_delay = min(delay * 4, LLM_RETRY_MAX_DELAY)
                logger.warning(
                    "[%s] Rate-limit hit on attempt %d/%d. Retrying in %.1fs …",
                    disease_id, attempt, max_retries, rate_delay,
                )
                time.sleep(rate_delay)
                delay = min(delay * 2, LLM_RETRY_MAX_DELAY)
                continue

            if is_server_err or attempt < max_retries:
                logger.warning(
                    "[%s] LLM error on attempt %d/%d: %s. Retrying in %.1fs …",
                    disease_id, attempt, max_retries, exc, delay,
                )
            else:
                # Final attempt failed — log at ERROR level
                logger.error(
                    "[%s] LLM call failed permanently after %d attempts: %s",
                    disease_id, max_retries, exc,
                )
                return None

        # Sleep before the next retry (skip sleep after the last attempt)
        if attempt < max_retries:
            time.sleep(min(delay, LLM_RETRY_MAX_DELAY))
            delay = min(delay * 2, LLM_RETRY_MAX_DELAY)

    logger.error(
        "[%s] LLM call exhausted all %d retries. Skipping disease.",
        disease_id, max_retries,
    )
    return None


def build_prompt_messages(disease_id: str, disease_payload: dict, num_patients: int):
    """
    disease_payload format:
    {
      "name": "nephropathic cystinosis",
      "Symptom A": ["organ1", "organ2"],
      "Symptom B": ["organ3"]
    }
    """
    disease_obj = {disease_id: disease_payload}

    return [
        {
            "role": "system",
            "content": f"""
You are a medical dataset generator.

You will receive one disease object.

The disease object contains:
- one disease id
- one disease name
- multiple symptoms
- for each symptom, a list of allowed related organs/body locations

Your task:
Generate EXACTLY {num_patients} realistic patients.
For each patient, generate EXACTLY 3 patient-authored questions in sequence:
Q1, Q2, Q3.

Input:
{json.dumps(disease_obj, ensure_ascii=False, indent=2)}

How to interpret the input:
- Ignore the outer disease id as a symptom.
- Ignore the "name" field as a symptom.
- Every other key is a symptom. These are the ONLY valid symptoms.
- The value of each symptom key is the list of organs/body locations allowed for that symptom. These are the ONLY valid organs for that symptom.

STRICT GROUNDING RULES — these override everything else:
- symptoms_used must contain ONLY symptom keys that appear in the input object.
- organs_used must contain ONLY organ/body-location strings that appear  in that symptom's value list in the input.
- You MUST NOT add, infer, rename, paraphrase, or invent any symptom or organ not present in the input.
- Before writing each question, internally check: "Is every symptom I am using a key in the input? Is every organ I am using in that symptom's list?" If not, remove it.
- The question text may use lay language to describe a symptom or organ, but the corresponding entries in symptoms_used and organs_used must still be the exact strings from the input.

Core generation rule:
Each question must be grounded in symptom + organ evidence from the input.
If you mention a body location for a symptom, it must come only from that symptom's own organ list.

Do NOT:
- invent symptoms
- invent organs for a symptom
- mix organs between symptoms
- mention the disease name
- output markdown
- output explanations

Do:
- make the questions sound like real patients
- progressively reveal evidence across Q1, Q2, Q3
- keep each patient internally consistent

Preferred question design:
- Q1: fewer/earlier concerns
- Q2: add 2 new symptoms and if no new symptoms more specific detail about previous symptoms;
- Q3: add 2 new symptoms and if no new symptoms more specific detail about previous symptoms;


Per-question target:
- only 2 symptoms per question
- Mention body parts naturally when useful
- Tie each body part mention only to the symptom it belongs to


Output STRICT JSON only:
{{
  "disease_id": "{disease_id}",
  "disease_name": "{disease_payload.get("name", "")}",
  "rounds_per_patient": 3,
  "patients": [
    {{
      "patient_id": "P1",
      "profile": "...",
      "qa_sequence": [
        {{
          "question_id": "P1_Q1",
          "question": "...",
          "symptoms_used": ["symptom1", "symptom2"],
          "organs_used": {{
            "symptom1": ["organA"],
            "symptom2": ["organB", "organC"]
          }}
        }},
        {{
          "question_id": "P1_Q2",
          "question": "...",
          "symptoms_used": [],
          "organs_used": {{}}
        }},
        {{
          "question_id": "P1_Q3",
          "question": "...",
          "symptoms_used": [],
          "organs_used": {{}}
        }}
      ]
    }}
  ]
}}

Validation rules (every rule must pass before outputting):
1. disease_id must equal the outer disease key from the input.
2. disease_name must equal the "name" field from the input.
3. Every entry in symptoms_used must be a key that exists verbatim in the input object.
4. Every key in organs_used must also appear in symptoms_used for that question.
5. Every organ value listed under a symptom in organs_used must exist verbatim in that symptom's allowed organ list in the input.
6. The question text must be consistent with symptoms_used and organs_used.
7. No symptom or organ may appear in the question text, symptoms_used, or organs_used if it is not present in the input.

Return STRICT JSON only.
""".strip(),
        },
        {
            "role": "user",
            "content": "Generate the patients and 3-round question sequences now.",
        },
    ]


def flatten_organs_used(organs_used) -> str:
    """
    Convert:
        {"Symptom A": ["kidney"], "Symptom B": ["eye", "cornea"]}
    into a readable string.
    """
    if not isinstance(organs_used, dict):
        return ""

    parts = []
    for symptom, organs in organs_used.items():
        organs_str = (
            ", ".join(str(o) for o in organs)
            if isinstance(organs, list)
            else str(organs)
        )
        parts.append(f"{symptom}: [{organs_str}]")
    return " | ".join(parts)


def validate_generation(disease_payload: dict, parsed: dict) -> bool:
    """
    Basic validator to ensure:
    - symptoms_used come from input symptoms
    - organs_used[symptom] only contains allowed organs for that symptom
    """
    valid_symptoms = {k for k in disease_payload.keys() if k != "name"}

    patients = parsed.get("patients", [])
    if not isinstance(patients, list):
        return False

    for patient in patients:
        seq = patient.get("qa_sequence", [])
        if not isinstance(seq, list):
            return False

        for q in seq:
            symptoms_used = q.get("symptoms_used", [])
            organs_used = q.get("organs_used", {})

            if not isinstance(symptoms_used, list):
                return False
            if not isinstance(organs_used, dict):
                return False

            for s in symptoms_used:
                if s not in valid_symptoms:
                    return False

            for symptom, organs in organs_used.items():
                if symptom not in valid_symptoms:
                    return False
                if symptom not in symptoms_used:
                    return False

                allowed_organs = set(disease_payload.get(symptom, []))
                if not isinstance(organs, list):
                    return False

                for organ in organs:
                    if organ not in allowed_organs:
                        return False

    return True


# ---------------------------------------------------------
# INPUT FILES
# ---------------------------------------------------------


with open("C:/Users/lilliam/Downloads/memory_project/dataset/outlier_diseases.json", "r") as f:
    data = json.load(f)

skip_disease_ids = set(list(data.keys()))


with open(
    "C:/Users/lilliam/Downloads/memory_project/dataset/complete_mapping.json",
    "r",
    encoding="utf-8",
) as f:
    disease_data = json.load(f)

chat = ChatCompletion(settings)
rows = []

out_path = Path(
    "C:/Users/lilliam/Downloads/memory_project/dataset/SYMPTOM_ORGAN_QUESTIONS3.xlsx"
)
out_path.parent.mkdir(parents=True, exist_ok=True)

num_patients = 2

# Counters for a summary log at the end
stats = {"skipped_early": 0, "llm_failed": 0, "parse_failed": 0,
         "validation_failed": 0, "success": 0}

for idx, (disease_id, disease_payload) in tqdm(
    enumerate(disease_data.items(), start=1),
    total=len(disease_data),
):
    disease_id = str(disease_id).strip()

    # ── Pre-flight checks ────────────────────────────────────────────────
    if not disease_id:
        stats["skipped_early"] += 1
        continue

    if disease_id in skip_disease_ids:
        stats["skipped_early"] += 1
        continue

    if not isinstance(disease_payload, dict):
        logger.warning("[%s] Payload is not a dict — skipping.", disease_id)
        stats["skipped_early"] += 1
        continue

    disease_name = disease_payload.get("name", "").strip()
    if not disease_name:
        logger.warning("[%s] Missing disease name — skipping.", disease_id)
        stats["skipped_early"] += 1
        continue

    symptom_keys = [k for k in disease_payload.keys() if k != "name"]
    if not symptom_keys:
        logger.warning("[%s] No symptoms found — skipping.", disease_id)
        stats["skipped_early"] += 1
        continue

    # ── Build prompt ─────────────────────────────────────────────────────
    messages = build_prompt_messages(
        disease_id=disease_id,
        disease_payload=disease_payload,
        num_patients=num_patients,
    )

    # ── LLM call (with retry + full error handling) ──────────────────────
    raw = call_llm_with_retry(chat, messages, disease_id)
    if raw is None:
        logger.error("[%s] Giving up after LLM failures.", disease_id)
        stats["llm_failed"] += 1
        continue

    # ── Parse JSON ───────────────────────────────────────────────────────
    parsed = parse_llm_json(raw)
    if not parsed:
        logger.warning(
            "[%s] Could not parse LLM output as JSON. Raw (first 300 chars): %.300s",
            disease_id, raw,
        )
        stats["parse_failed"] += 1
        continue

    # ── Domain validation ────────────────────────────────────────────────
    if not validate_generation(disease_payload, parsed):
        logger.warning("[%s] Validation failed — skipping.", disease_id)
        stats["validation_failed"] += 1
        continue

    # ── Flatten into rows ────────────────────────────────────────────────
    patients_data = parsed.get("patients", [])
    if not isinstance(patients_data, list):
        logger.warning("[%s] 'patients' field is not a list — skipping.", disease_id)
        stats["validation_failed"] += 1
        continue

    for patient in patients_data:
        sequence = patient.get("qa_sequence", [])
        if not isinstance(sequence, list):
            continue

        row = {
            "patient_id": patient.get("patient_id", ""),
            "profile": patient.get("profile", ""),
            "disease_name": disease_name,
            "disease_id": disease_id,
        }

        for i in range(3):
            q = sequence[i] if i < len(sequence) else {}

            question = q.get("question", "")
            symptoms_used = q.get("symptoms_used", [])
            organs_used = q.get("organs_used", {})

            symptoms_str = (
                ", ".join(str(s) for s in symptoms_used)
                if isinstance(symptoms_used, list)
                else str(symptoms_used)
            )
            organs_str = flatten_organs_used(organs_used)

            row[f"question{i + 1}"] = question
            row[f"symptoms_used_q{i + 1}"] = symptoms_str
            row[f"organs_used_q{i + 1}"] = organs_str

        rows.append(row)
        stats["success"] += 1

    # ── Periodic checkpoint save ─────────────────────────────────────────
    if idx % 5 == 0:
        try:
            pd.DataFrame(rows).to_excel(out_path, index=False)
            logger.info("Checkpoint saved at disease index %d (%s).", idx, disease_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to write checkpoint Excel: %s", exc)

# ── Final save ───────────────────────────────────────────────────────────
try:
    pd.DataFrame(rows).to_excel(out_path, index=False)
    logger.info("Saved Excel to: %s", out_path)
except Exception as exc:
    logger.error("Failed to write final Excel: %s", exc)

logger.info(
    "Run complete. Stats: %s",
    " | ".join(f"{k}={v}" for k, v in stats.items()),
)