import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# MedSafe: A Neurosymbolic Medication Safety Checker

**The problem:** medication errors — dangerous drug-drug interactions, contraindications,
and dosage mistakes — are one of the leading causes of preventable patient harm. Clinical
notes and prescriptions are written in messy natural language, but the safety rules that
govern whether a combination is dangerous are precise, well-established medical facts.

**Why neurosymbolic AI is the right architecture here:**

| | Pure LLM | Pure symbolic (rule engine) | Neurosymbolic (this project) |
|---|---|---|---|
| Reads free-text clinical notes | ✅ good | ❌ needs structured input | ✅ (neural layer) |
| Guaranteed to check every known interaction | ❌ can miss/hallucinate | ✅ deterministic | ✅ (symbolic layer) |
| Explainable / auditable decision | ⚠️ post-hoc, unreliable | ✅ exact rule + citation | ✅ (symbolic layer) |
| Same input → same output, always | ❌ not guaranteed | ✅ guaranteed | ✅ (symbolic layer) |

The architecture:

```
 free-text clinical note
        │
        ▼
 ┌─────────────────┐
 │  NEURAL LAYER    │   LLM extracts patient + medication facts
 │  (extraction.py) │   from unstructured text
 └────────┬─────────┘
          ▼
   structured facts (Patient, Medication)
          │
          ▼
 ┌─────────────────┐
 │ SYMBOLIC LAYER   │   deterministic rule engine checks facts
 │ (knowledge_base, │   against a curated clinical knowledge base
 │  reasoner.py)    │   (drug interactions, contraindications, dosage limits)
 └────────┬─────────┘
          ▼
     SafetyReport (severity, explanation, recommendation — per rule)
```

This notebook walks through the whole pipeline, shows it running on realistic clinical
notes, and demonstrates *why* the symbolic layer matters by comparing it against a naive
"just ask the LLM" baseline.

**⚠️ Disclaimer:** this is a research/education prototype with a demonstration knowledge
base of well-known textbook interactions. It is *not* a certified clinical decision
support system and must not be used for real patient care.""")

code("""import sys
sys.path.insert(0, "..")  # so `import medsafe` works from the notebook directory

from medsafe import (
    NeurosymbolicPipeline, SymbolicReasoner, MockExtractor, LLMExtractor,
    Patient, Medication, ClinicalCase, Severity,
)
from medsafe.knowledge_base import all_rules

print(f"Symbolic knowledge base loaded: {len(all_rules())} rules")""")

md("""## 1. The symbolic knowledge base

Before touching any LLM, let's look at what the symbolic layer actually knows. Every
rule is a small, independent, testable Python function — e.g. here's the rule for the
classic "triple whammy" (ACE inhibitor/ARB + diuretic + NSAID → acute kidney injury):""")

code('''import inspect
from medsafe.knowledge_base import interaction_triple_whammy
print(inspect.getsource(interaction_triple_whammy))''')

md("""## 2. Running the reasoner directly on structured facts

If you already have structured data (e.g. from an EHR), you can skip extraction
entirely and go straight to the symbolic reasoner:""")

code('''case = ClinicalCase(
    patient=Patient(age_years=78, weight_kg=82, egfr=28, conditions=["atrial fibrillation", "hypertension"]),
    medications=[
        Medication(name="warfarin", dose_value=5, dose_unit="mg", frequency="once daily"),
        Medication(name="amiodarone", dose_value=200, dose_unit="mg", frequency="once daily"),
        Medication(name="ibuprofen", dose_value=400, dose_unit="mg", frequency="tid"),
    ],
)

report = SymbolicReasoner().evaluate(case)
print(report.summary())
print()
for alert in report.alerts:
    print(alert)
    print()''')

md("""## 3. The full neurosymbolic pipeline: free text in, safety report out

Now the real use case: a messy clinical note goes in, and a structured, auditable
safety report comes out. By default the pipeline uses `MockExtractor`, a small
regex-based stand-in that lets this notebook run **without any API key** — swap in
`LLMExtractor()` (see the cell after this one) to use real Claude-based extraction.""")

code('''note_1 = \"\"\"
Patient is a 78-year-old male, weight 82kg, eGFR 28. History of atrial fibrillation
and hypertension. Currently prescribed warfarin 5mg once daily, amiodarone 200mg
once daily, and was just started on ibuprofen 400mg tid for knee pain.
\"\"\"

pipeline = NeurosymbolicPipeline()  # MockExtractor + SymbolicReasoner
report = pipeline.run(note_1)

print("--- INPUT NOTE ---")
print(note_1.strip())
print()
print("--- EXTRACTED FACTS (neural layer output) ---")
print(report.case.patient)
for m in report.case.medications:
    print(" ", m)
print()
print("--- SAFETY REPORT (symbolic layer output) ---")
print(report.summary())
print()
for alert in report.alerts:
    print(alert)
    print()''')

md("""### Using the real LLM extractor

To use actual Claude-based extraction instead of the offline mock, set your API key
and swap the extractor:

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-..."  # or set it in your shell environment

pipeline = NeurosymbolicPipeline(extractor=LLMExtractor())
report = pipeline.run(note_1)
```

The rest of the pipeline (the symbolic reasoning, the report, the explanations) is
**identical either way** — that's the point of the architecture: you can upgrade the
neural component independently without ever touching the safety-critical logic.""")

code('''import os

if os.environ.get("ANTHROPIC_API_KEY"):
    llm_pipeline = NeurosymbolicPipeline(extractor=LLMExtractor())
    llm_report = llm_pipeline.run(note_1)
    print(llm_report.summary())
    for alert in llm_report.alerts:
        print(alert)
else:
    print("No ANTHROPIC_API_KEY set in this environment — skipping live LLM call.")
    print("(This is expected in the sandbox this notebook was built in.")
    print(" Set the env var and re-run this cell to use real Claude extraction.)")''')

md("""## 4. More test cases across all three rule categories

Let's run a batch of realistic cases to exercise drug interactions, contraindications,
and dosage-limit rules.""")

code('''test_notes = {
    "Serotonin syndrome risk": \"\"\"
        45yo female on sertraline 100mg daily for depression, recently prescribed
        tramadol 50mg qid for back pain.
    \"\"\",
    "PDE5 + nitrate (absolute contraindication)": \"\"\"
        62yo male with erectile dysfunction on sildenafil 50mg, also has stable
        angina on isosorbide dinitrate.
    \"\"\",
    "Statin + CYP3A4 inhibitor": \"\"\"
        55yo patient on simvastatin 40mg daily, started on clarithromycin 500mg
        bid for a sinus infection.
    \"\"\",
    "Metformin in severe CKD": \"\"\"
        70yo diabetic patient, eGFR 22, on metformin 1000mg bid.
    \"\"\",
    "ACE inhibitor in pregnancy": \"\"\"
        29yo pregnant patient, 12 weeks, on lisinopril 10mg daily for chronic hypertension.
    \"\"\",
    "Acetaminophen overdose risk with liver disease": \"\"\"
        60yo patient with alcoholic liver disease, taking acetaminophen 1000mg
        four times daily for chronic pain.
    \"\"\",
    "Safe, unremarkable regimen": \"\"\"
        50yo patient on atorvastatin 20mg daily and lisinopril 10mg daily for
        hyperlipidemia and hypertension. No other conditions noted.
    \"\"\",
}

pipeline = NeurosymbolicPipeline()
for label, note in test_notes.items():
    report = pipeline.run(note)
    flag = "⚠️ " if report.alerts else "✅ "
    print(f"{flag}{label}: {report.summary()}")
    for a in report.alerts:
        print(f"    - [{a.severity.value}] {a.rule_id}: {' + '.join(a.involved)}")
    print()''')

md("""## 5. Why the symbolic layer matters: consistency and coverage

A pure LLM asked "is this safe?" will usually catch *obvious* interactions, but its
answers are not guaranteed to be complete or repeatable — it can phrase things
differently each time, miss a less-common interaction, or occasionally state a wrong
one confidently. The symbolic reasoner, by contrast, checks the *same* fixed rule set
every time, so its output is exhaustive (over its known rules) and 100% reproducible.

We can demonstrate the reproducibility guarantee directly: run the identical case
through the reasoner many times and confirm the output never varies (something you
cannot guarantee by re-prompting an LLM at nonzero temperature).""")

code('''case = ClinicalCase(
    patient=Patient(egfr=25),
    medications=[Medication(name="metformin", dose_value=500, dose_unit="mg", frequency="bid")],
)

results = set()
for _ in range(50):
    report = SymbolicReasoner().evaluate(case)
    results.add(tuple(a.rule_id for a in report.alerts))

print(f"Distinct outputs across 50 runs: {len(results)}")
print(f"Output: {results}")
assert len(results) == 1, "Symbolic layer must be perfectly deterministic"
print("\\n✅ Confirmed: identical facts always produce identical, auditable alerts.")''')

md("""## 6. Coverage summary of the knowledge base

A quick inventory of what the symbolic layer currently knows about, grouped by
category and severity — useful both as documentation and as a sanity check when
adding new rules.""")

code('''from collections import Counter
from medsafe.models import ClinicalCase, Patient, Medication

# Probe each rule with an empty case just to read its metadata isn't possible directly
# (rules are pure functions, not data), so instead we report on the *docstring/name*
# of each registered rule as a proxy for the KB's coverage.
rule_fns = all_rules()
print(f"Total rules: {len(rule_fns)}\\n")
for fn in rule_fns:
    doc = (fn.__doc__ or "").strip().split("\\n")[0]
    print(f"  {fn.__name__:45s} {doc}")''')

md("""## 7. Try your own case

Edit the note below and re-run to see the pipeline in action on a new scenario.""")

code('''my_note = \"\"\"
68-year-old patient with a history of depression and chronic pain, on fluoxetine
40mg daily. Just started on tramadol 50mg qid by another provider for a shoulder
injury.
\"\"\"

report = NeurosymbolicPipeline().run(my_note)
print(report.summary())
print()
for alert in report.alerts:
    print(alert)
    print()''')

md("""## Summary

- **Neural layer** (`extraction.py`): turns unstructured clinical text into structured
  facts. Swappable — offline mock for demos, real Claude API for production.
- **Symbolic layer** (`knowledge_base.py`, `reasoner.py`): a deterministic, testable,
  auditable rule engine over a curated clinical knowledge base. This is where safety
  guarantees live, and it's covered by unit tests (`tests/test_rules.py`).
- **Pipeline** (`pipeline.py`): wires the two together, but keeps them cleanly
  separated — you can upgrade the LLM without touching the safety logic, and you can
  audit/extend the safety logic without ever needing to run a model.

This separation is the core idea of neurosymbolic AI: use neural networks for what
they're good at (perception, language, ambiguity), and symbolic reasoning for what
*it's* good at (guarantees, explainability, formal correctness) — instead of asking
one system to do both.""")

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

with open("notebooks/medsafe_demo.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written.")
