"""
Unit tests for the symbolic layer. These are the tests that matter most in
this project: the neural layer can be fuzzy, but the symbolic layer must be
provably correct on known cases.

Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medsafe.models import ClinicalCase, Medication, Patient, Severity
from medsafe.reasoner import SymbolicReasoner


def med(name, dose_value=None, dose_unit=None, frequency=None):
    return Medication(name=name, dose_value=dose_value, dose_unit=dose_unit, frequency=frequency)


def test_warfarin_aspirin_interaction_fires():
    case = ClinicalCase(patient=Patient(), medications=[med("warfarin"), med("aspirin")])
    report = SymbolicReasoner().evaluate(case)
    ids = [a.rule_id for a in report.alerts]
    assert "INT-001" in ids


def test_no_alert_for_safe_regimen():
    case = ClinicalCase(patient=Patient(age_years=40), medications=[med("metformin", 500, "mg", "bid")])
    report = SymbolicReasoner().evaluate(case)
    assert report.alerts == []


def test_sildenafil_nitrate_major_absolute():
    case = ClinicalCase(patient=Patient(), medications=[med("sildenafil"), med("nitroglycerin")])
    report = SymbolicReasoner().evaluate(case)
    assert any(a.rule_id == "INT-004" and a.severity == Severity.MAJOR for a in report.alerts)


def test_metformin_contraindicated_severe_renal_impairment():
    case = ClinicalCase(patient=Patient(egfr=20), medications=[med("metformin", 1000, "mg", "bid")])
    report = SymbolicReasoner().evaluate(case)
    ids = [a.rule_id for a in report.alerts]
    assert "CI-003" in ids


def test_metformin_moderate_caution_mild_renal_impairment():
    case = ClinicalCase(patient=Patient(egfr=40), medications=[med("metformin", 1000, "mg", "bid")])
    report = SymbolicReasoner().evaluate(case)
    ids = [a.rule_id for a in report.alerts]
    assert "CI-003b" in ids
    assert "CI-003" not in ids


def test_ace_inhibitor_contraindicated_in_pregnancy():
    case = ClinicalCase(patient=Patient(is_pregnant=True), medications=[med("lisinopril")])
    report = SymbolicReasoner().evaluate(case)
    ids = [a.rule_id for a in report.alerts]
    assert "CI-001" in ids


def test_acetaminophen_overdose_flagged():
    case = ClinicalCase(patient=Patient(), medications=[med("acetaminophen", 1200, "mg", "qid")])
    report = SymbolicReasoner().evaluate(case)
    ids = [a.rule_id for a in report.alerts]
    assert "DOSE-001" in ids


def test_acetaminophen_lower_limit_with_liver_disease():
    case = ClinicalCase(
        patient=Patient(conditions=["liver disease"]),
        medications=[med("acetaminophen", 650, "mg", "tid")],  # 1950mg/day: fine normally
    )
    report = SymbolicReasoner().evaluate(case)
    # 1950mg/day is under the normal 4000mg limit but should still be fine even
    # with the reduced 2000mg limit - let's push it over instead.
    case2 = ClinicalCase(
        patient=Patient(conditions=["liver disease"]),
        medications=[med("acetaminophen", 1000, "mg", "tid")],  # 3000mg/day
    )
    report2 = SymbolicReasoner().evaluate(case2)
    ids2 = [a.rule_id for a in report2.alerts]
    assert "DOSE-001" in ids2  # exceeds the reduced 2000mg/day limit


def test_triple_whammy():
    case = ClinicalCase(
        patient=Patient(),
        medications=[med("lisinopril"), med("hydrochlorothiazide"), med("ibuprofen")],
    )
    report = SymbolicReasoner().evaluate(case)
    ids = [a.rule_id for a in report.alerts]
    assert "INT-007" in ids


def test_benzo_opioid_major():
    case = ClinicalCase(patient=Patient(), medications=[med("alprazolam"), med("oxycodone")])
    report = SymbolicReasoner().evaluate(case)
    assert any(a.rule_id == "INT-011" and a.severity == Severity.MAJOR for a in report.alerts)


def test_determinism_same_input_same_output():
    """Core neurosymbolic guarantee: identical facts always produce identical alerts."""
    case = ClinicalCase(patient=Patient(egfr=25), medications=[med("metformin", 500, "mg", "bid")])
    r1 = SymbolicReasoner().evaluate(case)
    r2 = SymbolicReasoner().evaluate(case)
    assert [a.rule_id for a in r1.alerts] == [a.rule_id for a in r2.alerts]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
