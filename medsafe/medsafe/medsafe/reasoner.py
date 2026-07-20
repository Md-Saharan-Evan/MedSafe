"""
The symbolic reasoning engine.

Deliberately simple: a forward-chaining rule evaluator. Every rule is
evaluated against the case's facts (patient + medications); any rule whose
conditions match fires and contributes an Alert. There is no learning, no
probability, no hallucination possible here - a rule either fires or it
doesn't, and the same case always produces the same alerts. This
determinism is the whole point: it's the part of the system a clinician
can audit and trust.
"""

from __future__ import annotations

from .knowledge_base import all_rules
from .models import ClinicalCase, SafetyReport, Severity

_SEVERITY_ORDER = {Severity.MAJOR: 0, Severity.MODERATE: 1, Severity.MINOR: 2}


class SymbolicReasoner:
    def __init__(self, rules=None):
        self.rules = rules if rules is not None else all_rules()

    def evaluate(self, case: ClinicalCase) -> SafetyReport:
        alerts = []
        for rule_fn in self.rules:
            fired = rule_fn(case)
            alerts.extend(fired)
        alerts.sort(key=lambda a: _SEVERITY_ORDER[a.severity])
        return SafetyReport(case=case, alerts=alerts, rules_evaluated=len(self.rules))

    def explain_rule(self, rule_id: str) -> str:
        for rule_fn in self.rules:
            if rule_fn.__name__.upper().startswith(rule_id.split("-")[0].lower()):
                return rule_fn.__doc__ or rule_fn.__name__
        return "Rule not found."
