"""
The symbolic knowledge base.

This is the part of the system that must be *correct*, *auditable*, and
*never left to a language model's judgment*. Each rule is a small, pure
Python function: (patient, medications) -> list[Alert]. Rules are
independent, declarative, and testable in isolation - this is the
"symbolic" half of the neurosymbolic pipeline.

NOTE ON SCOPE: this is a demonstration knowledge base covering well-known,
textbook drug interactions and contraindications. It is NOT a substitute
for a licensed clinical decision support system (e.g. Micromedex, Lexicomp)
and should not be used for real patient care. The point of this project is
to demonstrate the *architecture*, not to ship a certified medical device.
"""

from __future__ import annotations

from typing import Callable

from .models import Alert, AlertCategory, ClinicalCase, Severity

Rule = Callable[[ClinicalCase], list[Alert]]

_RULES: list[Rule] = []


def rule(func: Rule) -> Rule:
    """Decorator that registers a rule function into the global rule set."""
    _RULES.append(func)
    return func


def all_rules() -> list[Rule]:
    return list(_RULES)


def _has(meds, *names: str) -> list[str]:
    """Return the medication names present that match any of `names` (substring match)."""
    found = []
    for m in meds:
        for n in names:
            if n in m.name.lower():
                found.append(m.name)
                break
    return found


# ---------------------------------------------------------------------------
# DRUG-DRUG INTERACTION RULES
# ---------------------------------------------------------------------------

@rule
def interaction_warfarin_bleeding_risk(case: ClinicalCase) -> list[Alert]:
    meds = case.medications
    if _has(meds, "warfarin"):
        culprits = _has(meds, "aspirin", "ibuprofen", "naproxen", "diclofenac")
        if culprits:
            return [Alert(
                rule_id="INT-001",
                severity=Severity.MAJOR,
                category=AlertCategory.DRUG_INTERACTION,
                involved=["warfarin"] + culprits,
                explanation="NSAIDs/aspirin combined with warfarin significantly increase "
                            "bleeding risk (additive antiplatelet + anticoagulant effect, "
                            "plus GI mucosal irritation).",
                recommendation="Avoid combination if possible; if unavoidable, use lowest "
                                "effective dose, add GI protection, and monitor INR closely.",
            )]
    return []


@rule
def interaction_warfarin_amiodarone(case: ClinicalCase) -> list[Alert]:
    meds = case.medications
    if _has(meds, "warfarin") and _has(meds, "amiodarone"):
        return [Alert(
            rule_id="INT-002",
            severity=Severity.MAJOR,
            category=AlertCategory.DRUG_INTERACTION,
            involved=["warfarin", "amiodarone"],
            explanation="Amiodarone inhibits warfarin metabolism (CYP2C9), often raising "
                        "INR substantially within days to weeks.",
            recommendation="Reduce warfarin dose empirically (commonly 30-50%) and monitor "
                            "INR frequently after starting/stopping amiodarone.",
        )]
    return []


@rule
def interaction_serotonin_syndrome(case: ClinicalCase) -> list[Alert]:
    meds = case.medications
    ssris = _has(meds, "sertraline", "fluoxetine", "paroxetine", "citalopram", "escitalopram")
    risky = _has(meds, "tramadol", "maoi", "phenelzine", "tranylcypromine", "linezolid")
    if ssris and risky:
        return [Alert(
            rule_id="INT-003",
            severity=Severity.MAJOR,
            category=AlertCategory.DRUG_INTERACTION,
            involved=ssris + risky,
            explanation="Combining an SSRI with another serotonergic agent (tramadol, "
                        "MAOI, linezolid) increases risk of serotonin syndrome "
                        "(hyperthermia, autonomic instability, altered mental status).",
            recommendation="Avoid combination. If both are clinically necessary, use the "
                            "lowest doses, ensure adequate washout when switching, and "
                            "counsel patient on serotonin syndrome symptoms.",
        )]
    return []


@rule
def interaction_sildenafil_nitrates(case: ClinicalCase) -> list[Alert]:
    meds = case.medications
    pde5 = _has(meds, "sildenafil", "tadalafil", "vardenafil")
    nitrates = _has(meds, "nitroglycerin", "isosorbide")
    if pde5 and nitrates:
        return [Alert(
            rule_id="INT-004",
            severity=Severity.MAJOR,
            category=AlertCategory.DRUG_INTERACTION,
            involved=pde5 + nitrates,
            explanation="PDE5 inhibitors potentiate nitrate-induced vasodilation, causing "
                        "severe, potentially fatal hypotension.",
            recommendation="Absolute contraindication. Do not co-administer under any "
                            "circumstances.",
        )]
    return []


@rule
def interaction_statin_cyp3a4_inhibitor(case: ClinicalCase) -> list[Alert]:
    meds = case.medications
    statins = _has(meds, "simvastatin", "atorvastatin", "lovastatin")
    inhibitors = _has(meds, "clarithromycin", "erythromycin", "itraconazole", "ketoconazole")
    if statins and inhibitors:
        return [Alert(
            rule_id="INT-005",
            severity=Severity.MAJOR,
            category=AlertCategory.DRUG_INTERACTION,
            involved=statins + inhibitors,
            explanation="Strong CYP3A4 inhibitors raise statin plasma levels substantially, "
                        "increasing risk of myopathy and rhabdomyolysis.",
            recommendation="Hold the statin during the course of the interacting drug, or "
                            "switch to a statin not metabolized by CYP3A4 (e.g. rosuvastatin, "
                            "pravastatin).",
        )]
    return []


@rule
def interaction_ace_potassium_sparing(case: ClinicalCase) -> list[Alert]:
    meds = case.medications
    acei = _has(meds, "lisinopril", "enalapril", "ramipril", "captopril")
    ksparing = _has(meds, "spironolactone", "eplerenone", "amiloride")
    if acei and ksparing:
        return [Alert(
            rule_id="INT-006",
            severity=Severity.MODERATE,
            category=AlertCategory.DRUG_INTERACTION,
            involved=acei + ksparing,
            explanation="ACE inhibitors combined with potassium-sparing diuretics increase "
                        "risk of hyperkalemia, especially with renal impairment.",
            recommendation="Monitor serum potassium and renal function within 1-2 weeks of "
                            "starting the combination.",
        )]
    return []


@rule
def interaction_triple_whammy(case: ClinicalCase) -> list[Alert]:
    """ACEI/ARB + diuretic + NSAID = classic 'triple whammy' acute kidney injury risk."""
    meds = case.medications
    acei_arb = _has(meds, "lisinopril", "enalapril", "ramipril", "losartan", "valsartan")
    diuretic = _has(meds, "hydrochlorothiazide", "furosemide", "bumetanide")
    nsaid = _has(meds, "ibuprofen", "naproxen", "diclofenac")
    if acei_arb and diuretic and nsaid:
        return [Alert(
            rule_id="INT-007",
            severity=Severity.MAJOR,
            category=AlertCategory.DRUG_INTERACTION,
            involved=acei_arb + diuretic + nsaid,
            explanation="The 'triple whammy': ACEI/ARB + diuretic + NSAID together "
                        "significantly increase risk of acute kidney injury, especially "
                        "in the elderly or dehydrated patients.",
            recommendation="Avoid the three-way combination; if an NSAID is truly needed, "
                            "use short-term, lowest dose, and check renal function before "
                            "and during use.",
        )]
    return []


@rule
def interaction_clopidogrel_omeprazole(case: ClinicalCase) -> list[Alert]:
    meds = case.medications
    if _has(meds, "clopidogrel") and _has(meds, "omeprazole", "esomeprazole"):
        return [Alert(
            rule_id="INT-008",
            severity=Severity.MODERATE,
            category=AlertCategory.DRUG_INTERACTION,
            involved=["clopidogrel", "omeprazole"],
            explanation="Omeprazole/esomeprazole strongly inhibit CYP2C19, reducing "
                        "conversion of clopidogrel to its active metabolite and its "
                        "antiplatelet effect.",
            recommendation="Consider pantoprazole or an H2-blocker instead, particularly "
                            "in patients with recent stent placement or ACS.",
        )]
    return []


@rule
def interaction_lithium_toxicity(case: ClinicalCase) -> list[Alert]:
    meds = case.medications
    if _has(meds, "lithium"):
        culprits = _has(meds, "ibuprofen", "naproxen", "lisinopril", "enalapril",
                         "hydrochlorothiazide")
        if culprits:
            return [Alert(
                rule_id="INT-009",
                severity=Severity.MODERATE,
                category=AlertCategory.DRUG_INTERACTION,
                involved=["lithium"] + culprits,
                explanation="NSAIDs, ACE inhibitors, and thiazide diuretics all reduce "
                            "lithium renal clearance, risking lithium toxicity "
                            "(tremor, confusion, arrhythmia).",
                recommendation="Monitor lithium levels closely (within a week) after "
                                "starting/stopping any of these agents.",
            )]
    return []


@rule
def interaction_digoxin_amiodarone(case: ClinicalCase) -> list[Alert]:
    meds = case.medications
    if _has(meds, "digoxin") and _has(meds, "amiodarone"):
        return [Alert(
            rule_id="INT-010",
            severity=Severity.MODERATE,
            category=AlertCategory.DRUG_INTERACTION,
            involved=["digoxin", "amiodarone"],
            explanation="Amiodarone inhibits P-glycoprotein, roughly doubling digoxin "
                        "plasma levels and increasing risk of digoxin toxicity.",
            recommendation="Reduce digoxin dose by ~50% when starting amiodarone and "
                            "monitor digoxin levels/ECG.",
        )]
    return []


@rule
def interaction_benzo_opioid(case: ClinicalCase) -> list[Alert]:
    meds = case.medications
    benzo = _has(meds, "alprazolam", "diazepam", "lorazepam", "clonazepam")
    opioid = _has(meds, "oxycodone", "morphine", "hydrocodone", "fentanyl", "tramadol")
    if benzo and opioid:
        return [Alert(
            rule_id="INT-011",
            severity=Severity.MAJOR,
            category=AlertCategory.DRUG_INTERACTION,
            involved=benzo + opioid,
            explanation="Co-administration of benzodiazepines and opioids causes additive "
                        "CNS and respiratory depression; a leading cause of overdose deaths.",
            recommendation="Avoid combination when possible; if necessary, use lowest "
                            "effective doses and prescribe naloxone.",
        )]
    return []


@rule
def interaction_methotrexate_nsaid(case: ClinicalCase) -> list[Alert]:
    meds = case.medications
    if _has(meds, "methotrexate"):
        culprits = _has(meds, "ibuprofen", "naproxen", "diclofenac")
        if culprits:
            return [Alert(
                rule_id="INT-012",
                severity=Severity.MODERATE,
                category=AlertCategory.DRUG_INTERACTION,
                involved=["methotrexate"] + culprits,
                explanation="NSAIDs reduce renal clearance of methotrexate, increasing "
                            "risk of methotrexate toxicity (myelosuppression, mucositis) - "
                            "particularly at higher (oncologic) methotrexate doses.",
                recommendation="Use with caution; avoid at high-dose methotrexate regimens, "
                                "monitor CBC and renal function at low doses.",
            )]
    return []


# ---------------------------------------------------------------------------
# CONTRAINDICATION RULES (patient condition / state vs. drug)
# ---------------------------------------------------------------------------

@rule
def contraindication_ace_pregnancy(case: ClinicalCase) -> list[Alert]:
    patient = case.patient
    culprits = _has(case.medications, "lisinopril", "enalapril", "ramipril", "captopril",
                     "losartan", "valsartan")
    if patient.is_pregnant and culprits:
        return [Alert(
            rule_id="CI-001",
            severity=Severity.MAJOR,
            category=AlertCategory.CONTRAINDICATION,
            involved=culprits + ["pregnancy"],
            explanation="ACE inhibitors and ARBs are teratogenic (fetal renal damage, "
                        "oligohydramnios) and contraindicated in pregnancy.",
            recommendation="Discontinue immediately and switch to a pregnancy-safe "
                            "antihypertensive (e.g. labetalol, nifedipine, methyldopa).",
        )]
    return []


@rule
def contraindication_beta_blocker_asthma(case: ClinicalCase) -> list[Alert]:
    patient = case.patient
    culprits = _has(case.medications, "propranolol", "atenolol", "metoprolol", "carvedilol")
    if patient.has_condition("asthma", "copd", "bronchospasm") and culprits:
        return [Alert(
            rule_id="CI-002",
            severity=Severity.MODERATE,
            category=AlertCategory.CONTRAINDICATION,
            involved=culprits + ["asthma/COPD"],
            explanation="Non-selective (and to a lesser extent cardioselective) beta "
                        "blockers can trigger bronchospasm in patients with reactive "
                        "airway disease.",
            recommendation="If a beta blocker is necessary, prefer a cardioselective agent "
                            "(e.g. metoprolol) at a low starting dose, and monitor for "
                            "wheeze/dyspnea.",
        )]
    return []


@rule
def contraindication_metformin_renal(case: ClinicalCase) -> list[Alert]:
    patient = case.patient
    culprits = _has(case.medications, "metformin")
    if culprits and patient.egfr is not None:
        if patient.egfr < 30:
            return [Alert(
                rule_id="CI-003",
                severity=Severity.MAJOR,
                category=AlertCategory.CONTRAINDICATION,
                involved=culprits + [f"eGFR {patient.egfr}"],
                explanation="Metformin is contraindicated at eGFR < 30 mL/min/1.73m^2 due "
                            "to risk of lactic acidosis from drug accumulation.",
                recommendation="Discontinue metformin; consider alternative agents "
                                "(e.g. DPP-4 inhibitor, insulin) with dosing appropriate "
                                "for renal function.",
            )]
        elif patient.egfr < 45:
            return [Alert(
                rule_id="CI-003b",
                severity=Severity.MODERATE,
                category=AlertCategory.CONTRAINDICATION,
                involved=culprits + [f"eGFR {patient.egfr}"],
                explanation="Metformin requires dose reduction and increased monitoring "
                            "at eGFR 30-45 mL/min/1.73m^2.",
                recommendation="Reduce dose (commonly to max 1000mg/day), monitor renal "
                                "function every 3-6 months.",
            )]
    return []


@rule
def contraindication_nsaid_renal(case: ClinicalCase) -> list[Alert]:
    patient = case.patient
    culprits = _has(case.medications, "ibuprofen", "naproxen", "diclofenac")
    if culprits and patient.egfr is not None and patient.egfr < 30:
        return [Alert(
            rule_id="CI-004",
            severity=Severity.MODERATE,
            category=AlertCategory.CONTRAINDICATION,
            involved=culprits + [f"eGFR {patient.egfr}"],
            explanation="NSAIDs reduce renal prostaglandin-mediated blood flow and can "
                        "worsen renal function in patients with significant CKD.",
            recommendation="Avoid NSAIDs; use acetaminophen or a non-nephrotoxic "
                            "alternative for pain/inflammation.",
        )]
    return []


@rule
def contraindication_aspirin_pediatric_viral(case: ClinicalCase) -> list[Alert]:
    patient = case.patient
    culprits = _has(case.medications, "aspirin")
    if culprits and patient.age_years is not None and patient.age_years < 18 \
            and patient.has_condition("viral", "influenza", "chickenpox", "varicella"):
        return [Alert(
            rule_id="CI-005",
            severity=Severity.MAJOR,
            category=AlertCategory.CONTRAINDICATION,
            involved=culprits + ["pediatric viral illness"],
            explanation="Aspirin in children/teens with a viral illness carries risk of "
                        "Reye's syndrome (acute encephalopathy + liver failure).",
            recommendation="Avoid aspirin; use acetaminophen or ibuprofen instead "
                            "(per age-appropriate dosing).",
        )]
    return []


# ---------------------------------------------------------------------------
# DOSAGE LIMIT RULES
# ---------------------------------------------------------------------------

@rule
def dosage_acetaminophen_max(case: ClinicalCase) -> list[Alert]:
    patient = case.patient
    has_liver_risk = patient.has_condition("liver", "hepatic", "cirrhosis", "alcohol")
    limit = 2000 if has_liver_risk else 4000
    for med in case.medications:
        if "acetaminophen" in med.name.lower() or "paracetamol" in med.name.lower():
            daily = med.daily_dose_mg()
            if daily is not None and daily > limit:
                return [Alert(
                    rule_id="DOSE-001",
                    severity=Severity.MAJOR if daily > 6000 else Severity.MODERATE,
                    category=AlertCategory.DOSAGE_LIMIT,
                    involved=[med.name],
                    explanation=f"Prescribed acetaminophen totals ~{daily:.0f}mg/day, "
                                f"exceeding the {'reduced ' if has_liver_risk else ''}"
                                f"safe limit of {limit}mg/day"
                                + (" (patient has hepatic risk factors)." if has_liver_risk
                                   else " for a patient without hepatic risk factors."),
                    recommendation=f"Reduce total daily acetaminophen (from all sources, "
                                    f"including combination products) to <={limit}mg/day.",
                )]
    return []


@rule
def dosage_warfarin_elderly_caution(case: ClinicalCase) -> list[Alert]:
    patient = case.patient
    for med in case.medications:
        if "warfarin" in med.name.lower() and patient.age_years is not None \
                and patient.age_years >= 75 and med.dose_value is not None \
                and med.dose_value > 5:
            return [Alert(
                rule_id="DOSE-002",
                severity=Severity.MINOR,
                category=AlertCategory.DOSAGE_LIMIT,
                involved=[med.name],
                explanation="Elderly patients (75+) are more sensitive to warfarin and "
                            "typically require lower starting doses than younger adults.",
                recommendation="Consider a lower starting dose (e.g. 2-4mg/day) with "
                                "closer INR monitoring in elderly patients.",
            )]
    return []


@rule
def dosage_max_weight_based_check(case: ClinicalCase) -> list[Alert]:
    """Generic example: some weight-based dosing sanity check (e.g. for a weight-dosed drug)."""
    patient = case.patient
    if patient.weight_kg is None:
        return []
    for med in case.medications:
        # Example: gentamicin dosed at 5-7mg/kg/day; flag if wildly out of range.
        if "gentamicin" in med.name.lower():
            daily = med.daily_dose_mg()
            if daily is not None:
                mg_per_kg = daily / patient.weight_kg
                if mg_per_kg > 7:
                    return [Alert(
                        rule_id="DOSE-003",
                        severity=Severity.MODERATE,
                        category=AlertCategory.DOSAGE_LIMIT,
                        involved=[med.name],
                        explanation=f"Gentamicin dose is ~{mg_per_kg:.1f}mg/kg/day, above "
                                    f"the typical 5-7mg/kg/day range, raising "
                                    f"nephrotoxicity/ototoxicity risk.",
                        recommendation="Recalculate weight-based dose and check trough "
                                        "levels/renal function.",
                    )]
    return []
