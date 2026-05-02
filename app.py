from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import re

app = FastAPI()


def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.upper()
    text = text.replace("CNTRST", "CONTRAST")
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_modality(desc: str) -> str:
    desc = normalize(desc)
    modalities = ["MRI", "CT", "XR", "US", "ULTRASOUND", "PET", "NM", "MAMMO"]
    for m in modalities:
        if desc.startswith(m + " ") or desc == m:
            if m == "ULTRASOUND":
                return "US"
            return m
    return ""


def get_body_region(desc: str) -> str:
    desc = normalize(desc)

    region_keywords = {
        "brain_head": ["BRAIN", "HEAD", "SKULL"],
        "chest": ["CHEST", "THORAX", "LUNG", "RIB"],
        "abdomen": ["ABDOMEN", "ABD", "LIVER", "KIDNEY", "RENAL", "PELVIS"],
        "spine": ["SPINE", "CERVICAL", "THORACIC", "LUMBAR"],
        "neck": ["NECK", "SOFT TISSUE NECK"],
        "cardiac": ["CARDIAC", "HEART", "CORONARY"],
        "breast": ["BREAST", "MAMMO"],
        "extremity": [
            "SHOULDER", "ELBOW", "WRIST", "HAND",
            "HIP", "KNEE", "ANKLE", "FOOT", "FEMUR", "TIBIA", "FIBULA"
        ],
        "vascular": ["ANGIO", "CTA", "MRA", "ARTERY", "VEIN", "VASCULAR"],
    }

    matched = set()
    for region, keywords in region_keywords.items():
        if any(k in desc for k in keywords):
            matched.add(region)

    return "|".join(sorted(matched))


def token_similarity(a: str, b: str) -> float:
    a_tokens = set(normalize(a).split())
    b_tokens = set(normalize(b).split())

    stopwords = {
        "WITH", "WITHOUT", "CONTRAST", "WO", "W", "LIMITED",
        "EXAM", "STUDY", "LEFT", "RIGHT"
    }

    a_tokens -= stopwords
    b_tokens -= stopwords

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def is_relevant(current: Dict[str, Any], prior: Dict[str, Any]) -> bool:
    current_desc = current.get("study_description", "")
    prior_desc = prior.get("study_description", "")

    cur_norm = normalize(current_desc)
    prior_norm = normalize(prior_desc)

    cur_modality = get_modality(cur_norm)
    prior_modality = get_modality(prior_norm)

    cur_region = get_body_region(cur_norm)
    prior_region = get_body_region(prior_norm)

    similarity = token_similarity(cur_norm, prior_norm)

    # Strong match: same/similar exam description
    if similarity >= 0.35:
        return True

    # Same body region is usually relevant
    if cur_region and prior_region:
        cur_regions = set(cur_region.split("|"))
        prior_regions = set(prior_region.split("|"))
        if cur_regions & prior_regions:
            return True

    # Special common cross-modality brain/head comparison
    if "brain_head" in cur_region and "brain_head" in prior_region:
        return True

    # Chest XR/CT often useful for comparison
    if "chest" in cur_region and "chest" in prior_region:
        return True

    # Same modality alone is not enough if body region differs
    if cur_modality and cur_modality == prior_modality and similarity >= 0.2:
        return True

    return False


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: Dict[str, Any]):
    predictions = []

    for case in payload.get("cases", []):
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