from fastapi import FastAPI
from typing import Dict, Any
import re
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize(text: str) -> str:
    if not text:
        return ""

    text = text.upper()

    replacements = {
        "CNTRST": "CONTRAST",
        "CONTRST": "CONTRAST",
        "W/O": " WITHOUT ",
        "WO ": " WITHOUT ",
        "W/": " WITH ",
        "&": " AND ",
        "LT": " LEFT ",
        "RT": " RIGHT ",
        "BILAT": " BILATERAL ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_modality(desc: str) -> str:
    desc = normalize(desc)
    padded = f" {desc} "

    if "MAM" in desc or "TOMO" in desc:
        return "MAMMO"

    if desc.startswith("CTA") or " CTA " in padded:
        return "CTA"

    if desc.startswith("MRA") or " MRA " in padded:
        return "MRA"

    if desc.startswith("MRI") or desc.startswith("MR ") or " MRI " in padded:
        return "MRI"

    if desc.startswith("CT") or " CT " in padded:
        return "CT"

    if desc.startswith("XR") or "X RAY" in desc or "RADIOGRAPH" in desc:
        return "XR"

    if desc.startswith("US") or "ULTRASOUND" in desc or " US " in padded or "DOPPLER" in desc:
        return "US"

    if desc.startswith("NM") or "NUC MED" in desc or "SPECT" in desc:
        return "NM"

    if "PET" in desc:
        return "PET"

    if desc.startswith("FL") or "FLUORO" in desc:
        return "FL"

    return ""


def get_body_region(desc: str) -> str:
    desc = normalize(desc)

    region_keywords = {
        "brain_head": [
            "BRAIN", "HEAD", "SKULL", "SINUS", "ORBIT", "FACE", "FACIAL"
        ],
        "chest": [
            "CHEST", "THORAX", "LUNG", "PULMONARY", "RIB", "STERNUM"
        ],
        "abdomen": [
            "ABDOMEN", "ABD", "LIVER", "HEPATIC", "GALLBLADDER",
            "BILIARY", "PANCREAS", "SPLEEN", "KIDNEY", "RENAL", "ADRENAL"
        ],
        "pelvis": [
            "PELVIS", "PELVIC", "BLADDER", "UTERUS", "OVARY",
            "PROSTATE", "SCROTUM", "TESTICLE"
        ],
        "spine": [
            "SPINE", "CERVICAL", "THORACIC", "LUMBAR", "SACRUM", "SACRAL"
        ],
        "neck": [
            "NECK", "THYROID", "PARATHYROID", "SOFT TISSUE NECK"
        ],
        "cardiac": [
            "CARDIAC", "HEART", "CORONARY", "MYO PERF"
        ],
        "breast": [
            "BREAST", "MAMMO", "TOMO"
        ],
        "vascular": [
            "ANGIO", "CTA", "MRA", "ARTERY", "ARTERIAL", "VEIN",
            "VENOUS", "VASCULAR", "DOPPLER", "CAROTID", "AORTA"
        ],
        "extremity": [
            "SHOULDER", "ELBOW", "WRIST", "HAND", "FINGER", "THUMB",
            "HIP", "KNEE", "ANKLE", "FOOT", "TOE", "FEMUR", "TIBIA",
            "FIBULA", "HUMERUS", "FOREARM", "CLAVICLE"
        ],
    }

    matched = set()

    for region, keywords in region_keywords.items():
        if any(keyword in desc for keyword in keywords):
            matched.add(region)

    return "|".join(sorted(matched))


def token_similarity(a: str, b: str) -> float:
    a_tokens = set(normalize(a).split())
    b_tokens = set(normalize(b).split())

    stopwords = {
        "WITH", "WITHOUT", "CONTRAST", "WO", "W", "AND", "OR",
        "LIMITED", "EXAM", "STUDY", "LEFT", "RIGHT", "BILATERAL",
        "PORTABLE", "AP", "PA", "LAT", "LATERAL", "VIEW", "VIEWS",
        "MIN", "MINIMUM", "COMPLETE", "ONLY"
    }

    a_tokens -= stopwords
    b_tokens -= stopwords

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def get_laterality(desc: str) -> str:
    desc = f" {normalize(desc)} "

    if " BILATERAL " in desc:
        return "B"
    if " LEFT " in desc:
        return "L"
    if " RIGHT " in desc:
        return "R"

    return ""


def is_relevant(current: Dict[str, Any], prior: Dict[str, Any]) -> bool:
    current_desc = current.get("study_description", "")
    prior_desc = prior.get("study_description", "")

    cur_norm = normalize(current_desc)
    prior_norm = normalize(prior_desc)

    cur_modality = get_modality(cur_norm)
    prior_modality = get_modality(prior_norm)

    cur_region = get_body_region(cur_norm)
    prior_region = get_body_region(prior_norm)

    cur_regions = set(cur_region.split("|")) if cur_region else set()
    prior_regions = set(prior_region.split("|")) if prior_region else set()
    overlap = cur_regions & prior_regions

    similarity = token_similarity(cur_norm, prior_norm)

    cur_laterality = get_laterality(cur_norm)
    prior_laterality = get_laterality(prior_norm)

    # Exact same normalized study description.
    if cur_norm == prior_norm:
        return True

    # Very similar descriptions are usually relevant.
    if similarity >= 0.40:
        return True

    # For extremities, left/right mismatch is usually not relevant.
    if (
        "extremity" in cur_regions
        and "extremity" in prior_regions
        and cur_laterality
        and prior_laterality
        and cur_laterality != prior_laterality
        and "B" not in {cur_laterality, prior_laterality}
        and similarity < 0.55
    ):
        return False

    # Same modality + same body region is a strong signal.
    if overlap and cur_modality == prior_modality:
        return True

    # Chest CT/XR priors are often useful across CT and XR.
    if (
        "chest" in overlap
        and cur_modality in {"CT", "XR", "CTA"}
        and prior_modality in {"CT", "XR", "CTA"}
    ):
        return True

    # Breast priors are usually relevant to current breast studies.
    if "breast" in overlap:
        return True

    # Brain/head cross-modality can create false positives, so require similarity.
    if "brain_head" in overlap:
        if cur_modality == prior_modality:
            return True
        return similarity >= 0.30

    # Abdomen and pelvis are related, but require some text overlap across modalities.
    if ("abdomen" in overlap or "pelvis" in overlap) and similarity >= 0.25:
        return True

    # Vascular studies are often relevant when both are vascular.
    if "vascular" in overlap and similarity >= 0.20:
        return True

    # Cardiac studies are often relevant when both are cardiac.
    if "cardiac" in overlap and similarity >= 0.20:
        return True

    # Same modality with moderate textual overlap.
    if cur_modality and cur_modality == prior_modality and similarity >= 0.25:
        return True

    return False


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: Dict[str, Any]):
    cases = payload.get("cases", [])
    prior_count = sum(len(case.get("prior_studies", [])) for case in cases)

    logger.info(
        "request_id=%s case_count=%s prior_count=%s",
        payload.get("request_id", "no-request-id"),
        len(cases),
        prior_count,
    )

    predictions = []

    for case in cases:
        case_id = case.get("case_id")
        current_study = case.get("current_study", {})
        prior_studies = case.get("prior_studies", [])

        for prior in prior_studies:
            predictions.append({
                "case_id": case_id,
                "study_id": prior.get("study_id"),
                "predicted_is_relevant": bool(is_relevant(current_study, prior))
            })

    return {"predictions": predictions}