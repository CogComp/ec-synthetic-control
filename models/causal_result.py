from dataclasses import dataclass
from typing import List, Literal, Optional

from models.copes_test_case import COPESTestCase
from models.kept_sample import KeptSample


@dataclass
class CausalResult:
    is_causal: bool | Literal["indeterminate"]
    anchor_similarity: bool
    inverted_effect: str
    anonymized_original_effect: str
    inverted_anchor: str
    samples: List[KeptSample]
    test_case: COPESTestCase
    reasoning: str
    anchor_similarity_reasoning: str

    def __repr__(self) -> str:
        return f"""
CausalResult(
    {self.is_causal=}
    {self.anchor_similarity=}
    {self.inverted_effect=}
    {self.anonymized_original_effect=}
    {self.inverted_anchor=}
    {self.test_case.effect()=}
    {self.reasoning=}
    {self.anchor_similarity_reasoning=}
    {self.samples=}
)
        """


@dataclass
class AllCausalResults:
    test_cases: List[COPESTestCase]
    results: List[CausalResult]

    def actual(self):
        return self.test_cases[0].cause_idx

    def pred(self):
        return [
            r.test_case.event_id_to_test for r in self.results if r.is_causal == True
        ]

    def indeterminate(self):
        return [
            r.test_case.event_id_to_test
            for r in self.results
            if r.is_causal == "indeterminate"
        ]

    def stats(self):
        actual_idx = set(self.test_cases[0].cause_idx) - set(self.indeterminate())
        all_idx = set(
            r.test_case.event_id_to_test
            for r in self.results
            if r.is_causal != "indeterminate"
        )
        res_idx = set(self.pred())

        tp = len(actual_idx.intersection(res_idx))
        tn = len(all_idx - res_idx - actual_idx)
        fp = len(res_idx - actual_idx)
        fn = len(actual_idx - res_idx)

        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    def as_dict(self):
        return {
            "test_case": self.test_cases[0].id,
            "actual": self.actual(),
            "predicted": self.pred(),
            "indeterminate": self.indeterminate(),
            **self.stats(),
        }

    def __repr__(self) -> str:
        return f"""
AllCausalResults(
    test_case={self.test_cases[0].id}
    actual={self.actual()}
    predicted={self.pred()}
    indeterminate={self.indeterminate()}
)
        """
