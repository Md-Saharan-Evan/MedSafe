"""
The end-to-end neurosymbolic pipeline: free text -> structured facts -> report.

    raw clinical text
          |
          v
   [ NEURAL: Extractor ]      <- handles ambiguity, natural language, typos
          |
          v
    ClinicalCase (facts)
          |
          v
  [ SYMBOLIC: SymbolicReasoner ]   <- deterministic, explainable, auditable
          |
          v
     SafetyReport
"""

from __future__ import annotations

from .extraction import Extractor, MockExtractor
from .models import ClinicalCase, SafetyReport
from .reasoner import SymbolicReasoner


class NeurosymbolicPipeline:
    def __init__(self, extractor: Extractor | None = None, reasoner: SymbolicReasoner | None = None):
        self.extractor = extractor or MockExtractor()
        self.reasoner = reasoner or SymbolicReasoner()

    def run(self, clinical_text: str) -> SafetyReport:
        case: ClinicalCase = self.extractor.extract(clinical_text)
        return self.reasoner.evaluate(case)

    def run_on_case(self, case: ClinicalCase) -> SafetyReport:
        """Skip extraction if you already have structured facts (e.g. from an EHR)."""
        return self.reasoner.evaluate(case)
