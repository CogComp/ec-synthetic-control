from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from typing import List, Optional


class SimilarityResponseWithReasoning(BaseModel):
    is_similar: bool
    reasoning: str


@dataclass
class KeptSample:
    corpus_id: int
    anchor: List[str]
    anon_anchor: str
    anchor_embed: List[float]
    anchor_cos: float
    treatment: str
    anon_treatment: str
    treatment_embed: List[float]
    treatment_cos: float
    effect: List[str]
    anon_effect: str
    effect_embed: List[float]
    llm_original_treatment_in_anchor: Optional[SimilarityResponseWithReasoning]
    llm_treatment_event_similar: Optional[SimilarityResponseWithReasoning]
    llm_original_treatment_in_effect: Optional[SimilarityResponseWithReasoning]

    def __repr__(self) -> str:
        return f"""
KeptSample(
    {self.corpus_id=}
    {self.anchor_cos=}
    {self.treatment_cos=}
    {self.anchor=}
    {self.anon_anchor=}
    {self.treatment=}
    {self.anon_treatment=}
    {self.effect=}
    {self.anon_effect=}
    {self.llm_treatment_event_similar=}
    {self.llm_original_treatment_in_anchor=}
    {self.llm_original_treatment_in_effect=}
)
        """
