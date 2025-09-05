from copy import deepcopy
from typing import List

from pydantic import BaseModel

from impl.strategy import Strategy
from models.causal_result import AllCausalResults, CausalResult
from models.copes_test_case import COPESTestCase
from utils.openai import (
    Model,
    PromptLabel,
    call_openai_structured,
    events_to_event_dict,
)

_PROMPT = """
You will be given a sequence of events, numbered from 1 to 4, and a final consequence. Your job is to try and understand the temporal sequencing of events, and return
the indices of events which are causes of the consequence in the key `causes` and your reasoning as a string in the key `reasoning` in a JSON object.

Events:
1: {event0}
2: {event1}
3: {event2}
4: {event3}

Consequence:
5. {event4}
"""


class GPT4Result(BaseModel):
    causes: List[int]
    reasoning: str


class GPT4Baseline(Strategy):

    def run(self, test_case: COPESTestCase) -> CausalResult:
        idx = test_case.event_id_to_test
        all_results = self.run_all(test_case)
        return all_results.results[idx]

    def run_all(self, test_case: COPESTestCase) -> AllCausalResults:
        events = test_case.events
        events_input = events_to_event_dict(events)
        prompt = _PROMPT.format_map(events_input)

        result = call_openai_structured(
            PromptLabel.GPT4_ZS, struct=GPT4Result, prompt=prompt, model=Model.GPT4
        )

        causal_results: List[CausalResult] = []
        test_cases = []
        for i in range(0, 4):
            tc = deepcopy(test_case)
            tc.event_id_to_test = i
            test_cases.append(tc)
            r = CausalResult(
                is_causal=(i + 1) in result.causes,
                anchor_similarity=True,
                inverted_effect="",
                anonymized_original_effect="",
                inverted_anchor="",
                samples=[],
                test_case=tc,
                reasoning=result.reasoning,
                anchor_similarity_reasoning="",
            )
            causal_results.append(r)

        return AllCausalResults(test_cases, causal_results)
