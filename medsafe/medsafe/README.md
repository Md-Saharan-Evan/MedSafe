# MedSafe — A Neurosymbolic Medication Safety Checker

A prototype demonstrating **neurosymbolic AI** applied to a real, high-stakes problem:
catching dangerous drug-drug interactions, contraindications, and dosage errors before
they reach a patient.

## Why this problem, why this architecture

Medication errors are a leading cause of preventable patient harm. Two facts make this
a great fit for neurosymbolic AI rather than a pure LLM or a pure rule engine:

1. **Clinical input is messy natural language** (notes, prescriptions, patient
   descriptions) — exactly what LLMs are good at parsing and rule engines are bad at.
2. **The safety-critical logic must be deterministic and auditable** — a clinician (or
   a regulator) needs to know *exactly* why a system flagged something, and the same
   patient facts must always produce the same answer. This is exactly what you don't
   want to leave to an LLM's probabilistic judgment.

So the system splits cleanly in two:

```
free-text clinical note
        │
        ▼
┌─────────────────────┐
│   NEURAL LAYER       │  LLM (Claude) extracts patient + medication facts
│   extraction.py       │  from unstructured text
└──────────┬───────────┘
           ▼
   structured facts (Patient, Medication)
           │
           ▼
┌─────────────────────┐
│  SYMBOLIC LAYER       │  deterministic forward-chaining rule engine
│  knowledge_base.py,   │  checks facts against a curated clinical KB
│  reasoner.py          │  (drug interactions, contraindications, dosage limits)
└──────────┬───────────┘
           ▼
     SafetyReport (severity + explanation + recommendation, per rule)
```

The neural and symbolic layers are fully decoupled: you can swap in a better LLM
without touching the safety logic, and you can extend/audit the safety logic without
ever running a model.

## Project structure

```
medsafe/
├── medsafe/
│   ├── models.py         # Patient, Medication, Alert, SafetyReport data models
│   ├── knowledge_base.py # ~20 symbolic rules: interactions, contraindications, dosage limits
│   ├── reasoner.py        # deterministic forward-chaining rule engine
│   ├── extraction.py      # LLMExtractor (real Claude API) + MockExtractor (offline demo)
│   └── pipeline.py        # wires extraction + reasoning together
├── tests/
│   └── test_rules.py      # unit tests for the symbolic layer (11 tests)
└── notebooks/
    └── medsafe_demo.ipynb  # end-to-end walkthrough with realistic clinical notes
```

## Quickstart

```bash
pip install anthropic   # only needed if you want real LLM extraction
cd medsafe
python -m pytest tests/ -v          # verify the symbolic layer
jupyter notebook notebooks/medsafe_demo.ipynb
```

```python
from medsafe import NeurosymbolicPipeline

note = """
Patient is a 78-year-old male, weight 82kg, eGFR 28. History of atrial fibrillation
and hypertension. Currently prescribed warfarin 5mg once daily, amiodarone 200mg
once daily, and was just started on ibuprofen 400mg tid for knee pain.
"""

report = NeurosymbolicPipeline().run(note)   # uses offline MockExtractor by default
print(report.summary())
for alert in report.alerts:
    print(alert)
```

To use real Claude-based extraction instead of the offline regex mock:

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-..."

from medsafe import NeurosymbolicPipeline, LLMExtractor
report = NeurosymbolicPipeline(extractor=LLMExtractor()).run(note)
```

## What's in the knowledge base

~20 rules across three categories, covering well-known textbook cases:

- **Drug-drug interactions**: warfarin+NSAIDs, warfarin+amiodarone, serotonin syndrome
  combinations, PDE5 inhibitors+nitrates, statins+CYP3A4 inhibitors, the "triple
  whammy" (ACEI/ARB+diuretic+NSAID), clopidogrel+PPI, lithium toxicity risk factors,
  digoxin+amiodarone, benzodiazepine+opioid, methotrexate+NSAIDs, and more.
- **Contraindications**: ACE inhibitors in pregnancy, beta blockers in asthma/COPD,
  metformin in renal impairment, NSAIDs in CKD, aspirin in pediatric viral illness.
- **Dosage limits**: acetaminophen daily maximum (with hepatic-risk adjustment),
  age-adjusted warfarin dosing caution, weight-based dosing sanity checks.

## Honest limitations

- This is a **research/education prototype**, not a certified medical device. The
  knowledge base is illustrative (~20 rules), not exhaustive — a real deployment would
  need a licensed clinical database (e.g. Lexicomp, Micromedex) and regulatory review.
- `MockExtractor` is a deliberately simple regex/keyword matcher included so the
  notebook runs without API credentials — it is *not* the neural contribution of this
  project. Swap in `LLMExtractor` for real natural-language understanding.
- The reasoner uses simple substring matching for drug names; a production system
  would need proper drug ontology normalization (RxNorm, brand↔generic mapping, etc.).
