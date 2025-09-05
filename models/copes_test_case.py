from functools import cached_property
from typing import List
from collections.abc import Callable


class COPESTestCase:

    def __init__(
        self,
        copes_dict: dict,
        event_id_to_test: int,
        context_augmenter: Callable[[List[str]], str] = lambda s: "",
    ):
        self.events = copes_dict["story"]
        self.id = copes_dict["id"]
        self.cause_idx: List[int] = copes_dict["cause_idx"]
        self.event_id_to_test = event_id_to_test
        self.context_augmenter = context_augmenter

    def __repr__(self):
        return f"""
TestCase {self.id}

events: {self.events}
event_id_to_test: {self.event_id_to_test}
test_event: {self.test_event}
augmented_context: {self.augmented_context}
anchor: {self.anchor_text()}
cause_id: {self.cause_idx}
cause: {[event for i, event in enumerate(self.events) if i in self.cause_idx ] }
effect: {self.effect()}
        """

    def anchor_text(self):
        return [
            s
            for s in [self.augmented_context, *self.events[: self.event_id_to_test]]
            if len(s) > 0
        ]

    def anchor_text_s(self):
        return " ".join(self.anchor_text())

    def effect(self):
        return self.events[-1]

    @cached_property
    def augmented_context(self):
        if self.event_id_to_test > 0:
            return ""
        return self.context_augmenter(self.events)

    @property
    def test_event(self):
        return self.events[self.event_id_to_test]
