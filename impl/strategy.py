from abc import ABC, abstractmethod

from models.causal_result import AllCausalResults, CausalResult
from models.copes_test_case import COPESTestCase


class Strategy(ABC):

    def run_all(self, test_case: COPESTestCase) -> AllCausalResults:
        raise NotImplementedError()

    def run(self, test_case: COPESTestCase) -> CausalResult:
        raise NotImplementedError()
