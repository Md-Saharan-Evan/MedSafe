"""
Core data models for the neurosymbolic medication safety checker.

These are the "facts" that flow from the neural extraction layer into the
symbolic reasoning layer. Keeping them as plain, typed dataclasses means the
symbolic engine can pattern-match against them deterministically, regardless
of how the facts were produced (LLM extraction, manual entry, EHR import...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Severity(str, Enum):
    MAJOR = "major"        # potentially life-threatening / do not co-administer
    MODERATE = "moderate"  # requires monitoring or dose adjustment
    MINOR = "minor"        # low clinical significance, be aware


class AlertCategory(str, Enum):
    DRUG_INTERACTION = "drug_interaction"
    CONTRAINDICATION = "contraindication"
    DOSAGE_LIMIT = "dosage_limit"


@dataclass
class Medication:
    name: str                      # normalized generic name, lowercase
    dose_value: Optional[float] = None
    dose_unit: Optional[str] = None   # "mg", "mcg", etc.
    frequency: Optional[str] = None   # e.g. "once daily", "BID", "q6h"
    route: Optional[str] = None       # "oral", "IV", ...

    def daily_dose_mg(self, doses_per_day_hint: Optional[float] = None) -> Optional[float]:
        """Best-effort estimate of total daily dose in mg, used by dosage rules."""
        if self.dose_value is None or self.dose_unit is None:
            return None
        value = self.dose_value
        if self.dose_unit.lower() in ("mcg", "ug"):
            value = value / 1000.0
        elif self.dose_unit.lower() != "mg":
            return None  # unit we don't know how to convert (e.g. mL of a suspension)

        freq_map = {
            "once daily": 1, "od": 1, "qd": 1, "daily": 1,
            "twice daily": 2, "bid": 2,
            "three times daily": 3, "tid": 3,
            "four times daily": 4, "qid": 4,
            "q6h": 4, "q8h": 3, "q12h": 2,
        }
        times = None
        if self.frequency:
            times = freq_map.get(self.frequency.strip().lower())
        if times is None:
            times = doses_per_day_hint
        if times is None:
            return None
        return value * times


@dataclass
class Patient:
    name: Optional[str] = None
    age_years: Optional[float] = None
    weight_kg: Optional[float] = None
    egfr: Optional[float] = None          # mL/min/1.73m^2, renal function
    is_pregnant: Optional[bool] = None
    conditions: list[str] = field(default_factory=list)   # normalized lowercase
    allergies: list[str] = field(default_factory=list)    # normalized lowercase

    def has_condition(self, *keywords: str) -> bool:
        conds = [c.lower() for c in self.conditions]
        return any(any(kw in c for kw in keywords) for c in conds)


@dataclass
class ClinicalCase:
    patient: Patient
    medications: list[Medication]
    source_text: Optional[str] = None   # the raw note this was extracted from


@dataclass
class Alert:
    rule_id: str
    severity: Severity
    category: AlertCategory
    involved: list[str]          # drug names / condition names implicated
    explanation: str
    recommendation: str

    def __str__(self) -> str:
        involved = " + ".join(self.involved)
        return (f"[{self.severity.value.upper()}] ({self.category.value}) {involved}\n"
                f"    Why: {self.explanation}\n"
                f"    Action: {self.recommendation}")


@dataclass
class SafetyReport:
    case: ClinicalCase
    alerts: list[Alert]
    rules_evaluated: int

    @property
    def has_major_alert(self) -> bool:
        return any(a.severity == Severity.MAJOR for a in self.alerts)

    def summary(self) -> str:
        if not self.alerts:
            return f"No safety issues found ({self.rules_evaluated} rules evaluated)."
        by_sev = {"major": 0, "moderate": 0, "minor": 0}
        for a in self.alerts:
            by_sev[a.severity.value] += 1
        return (f"{len(self.alerts)} alert(s) found "
                f"({by_sev['major']} major, {by_sev['moderate']} moderate, {by_sev['minor']} minor) "
                f"out of {self.rules_evaluated} rules evaluated.")
