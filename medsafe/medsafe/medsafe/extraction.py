"""
The neural extraction layer.

This is the part of the system that reads messy, unstructured clinical
text (a prescription, a physician's note, a patient's own description of
what they're taking) and turns it into the structured Patient/Medication
facts the symbolic reasoner needs. Free text is exactly what LLMs are
good at and rule engines are bad at - so this is where the "neural" half
of the pipeline lives.

Two implementations are provided:

1. LLMExtractor  - calls the real Anthropic API (claude-sonnet-5) with a
                    JSON-schema-constrained prompt. This is the "real"
                    neural component. Requires ANTHROPIC_API_KEY to be set.

2. MockExtractor  - a small deterministic regex-based stand-in, used so
                    this project can be demoed/tested end-to-end without
                    needing API credentials. It is intentionally simple
                    and is NOT the neurosymbolic contribution of this
                    project - it exists purely so the notebook runs
                    offline. Swap it for LLMExtractor to get the real
                    neural behavior.

Both implementations satisfy the same interface: extract(text) -> ClinicalCase
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

from .models import ClinicalCase, Medication, Patient

EXTRACTION_SCHEMA_PROMPT = """\
You are a clinical information extraction system. Read the clinical note below \
and extract ONLY the structured facts it explicitly states or strongly implies. \
Do not infer medications, doses, or conditions that are not mentioned. \
Normalize drug names to their generic (non-brand) lowercase name where possible.

Respond with ONLY a JSON object (no markdown fences, no commentary) matching this schema:

{
  "patient": {
    "name": string or null,
    "age_years": number or null,
    "weight_kg": number or null,
    "egfr": number or null,
    "is_pregnant": boolean or null,
    "conditions": [string, ...],
    "allergies": [string, ...]
  },
  "medications": [
    {
      "name": string,
      "dose_value": number or null,
      "dose_unit": string or null,
      "frequency": string or null,
      "route": string or null
    }
  ]
}

Clinical note:
---
{note}
---

JSON:"""


class Extractor(Protocol):
    def extract(self, text: str) -> ClinicalCase: ...


class LLMExtractor:
    """Real neural extractor, backed by the Anthropic API (claude-sonnet-5)."""

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def extract(self, text: str) -> ClinicalCase:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "The 'anthropic' package is required for LLMExtractor. "
                "Install with: pip install anthropic"
            ) from e

        if not self.api_key:
            raise RuntimeError(
                "No API key found. Set the ANTHROPIC_API_KEY environment variable, "
                "or use MockExtractor for an offline demo."
            )

        client = anthropic.Anthropic(api_key=self.api_key)
        prompt = EXTRACTION_SCHEMA_PROMPT.format(note=text)
        response = client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(block.text for block in response.content if block.type == "text")
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        return _case_from_dict(data, source_text=text)


class MockExtractor:
    """
    Deterministic, regex/keyword-based extractor for offline demo/testing.

    This is NOT meant to demonstrate NLP sophistication - it exists so the
    notebook can run without an API key. It recognizes a small vocabulary
    of drug names and simple patterns like 'metformin 500mg BID'.
    """

    _KNOWN_DRUGS = [
        "warfarin", "aspirin", "ibuprofen", "naproxen", "diclofenac", "amiodarone",
        "sertraline", "fluoxetine", "paroxetine", "citalopram", "escitalopram",
        "tramadol", "linezolid", "sildenafil", "tadalafil", "vardenafil",
        "nitroglycerin", "isosorbide", "simvastatin", "atorvastatin", "lovastatin",
        "clarithromycin", "erythromycin", "itraconazole", "ketoconazole",
        "lisinopril", "enalapril", "ramipril", "captopril", "losartan", "valsartan",
        "spironolactone", "eplerenone", "amiloride", "hydrochlorothiazide",
        "furosemide", "bumetanide", "clopidogrel", "omeprazole", "esomeprazole",
        "lithium", "digoxin", "alprazolam", "diazepam", "lorazepam", "clonazepam",
        "oxycodone", "morphine", "hydrocodone", "fentanyl", "methotrexate",
        "metformin", "propranolol", "atenolol", "metoprolol", "carvedilol",
        "acetaminophen", "paracetamol", "gentamicin",
    ]

    _DOSE_PATTERN = re.compile(
        r"\b(" + "|".join(_KNOWN_DRUGS) + r")\b(?:\s+(\d+(?:\.\d+)?)\s*(mg|mcg))?"
        r"(?:\s+((?:once|twice|three times|four times)\s+daily|bid|tid|qid|od|qd|q\d+h))?",
        re.IGNORECASE,
    )
    _AGE_PATTERN = re.compile(r"\b(\d{1,3})[\s-]*(?:year|yo|y/o|years old)", re.IGNORECASE)
    _WEIGHT_PATTERN = re.compile(r"\b(\d{1,3}(?:\.\d+)?)\s*kg\b", re.IGNORECASE)
    _EGFR_PATTERN = re.compile(r"eGFR[:\s]*(\d{1,3}(?:\.\d+)?)", re.IGNORECASE)
    _CONDITION_KEYWORDS = [
        "asthma", "copd", "pregnan", "liver", "hepatic", "cirrhosis", "alcohol",
        "viral", "influenza", "chickenpox", "varicella", "diabetes", "hypertension",
        "atrial fibrillation", "ckd", "chronic kidney disease",
    ]

    def extract(self, text: str) -> ClinicalCase:
        lower = text.lower()

        age_match = self._AGE_PATTERN.search(text)
        weight_match = self._WEIGHT_PATTERN.search(text)
        egfr_match = self._EGFR_PATTERN.search(text)

        conditions = [kw for kw in self._CONDITION_KEYWORDS if kw in lower]
        is_pregnant = "pregnan" in lower

        patient = Patient(
            age_years=float(age_match.group(1)) if age_match else None,
            weight_kg=float(weight_match.group(1)) if weight_match else None,
            egfr=float(egfr_match.group(1)) if egfr_match else None,
            is_pregnant=is_pregnant if is_pregnant else None,
            conditions=conditions,
        )

        meds = []
        seen = set()
        for m in self._DOSE_PATTERN.finditer(text):
            name = m.group(1).lower()
            if name in ("paracetamol",):
                name = "acetaminophen"
            if name in seen:
                continue
            seen.add(name)
            meds.append(Medication(
                name=name,
                dose_value=float(m.group(2)) if m.group(2) else None,
                dose_unit=m.group(3).lower() if m.group(3) else None,
                frequency=m.group(4).lower() if m.group(4) else None,
            ))

        return ClinicalCase(patient=patient, medications=meds, source_text=text)


def _case_from_dict(data: dict, source_text: str | None = None) -> ClinicalCase:
    p = data.get("patient", {}) or {}
    patient = Patient(
        name=p.get("name"),
        age_years=p.get("age_years"),
        weight_kg=p.get("weight_kg"),
        egfr=p.get("egfr"),
        is_pregnant=p.get("is_pregnant"),
        conditions=[c.lower() for c in (p.get("conditions") or [])],
        allergies=[a.lower() for a in (p.get("allergies") or [])],
    )
    meds = []
    for m in data.get("medications", []) or []:
        meds.append(Medication(
            name=m["name"].lower(),
            dose_value=m.get("dose_value"),
            dose_unit=m.get("dose_unit"),
            frequency=m.get("frequency"),
            route=m.get("route"),
        ))
    return ClinicalCase(patient=patient, medications=meds, source_text=source_text)
