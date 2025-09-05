import logging
import typing
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers.util import cos_sim
from sklearn import linear_model

from impl.corpus import Corpus, RetrievedCorpusDoc
from impl.inversion import invert_embeddings
from impl.strategy import Strategy
from models.causal_result import AllCausalResults, CausalResult
from models.copes_test_case import COPESTestCase
from models.kept_sample import KeptSample, SimilarityResponseWithReasoning
from utils.fts_utils import clean_raw_text_from_corpus
from utils.globals import get_executor
from utils.openai import (
    BasicArrayResultResponse,
    BasicResultResponse,
    Model,
    PromptLabel,
    call_openai_embeddings,
    call_openai_structured,
)


@dataclass
class ScHyperParams:
    corpus_retrieval_size: int = 100
    kept_samples_count: int = 5
    min_samples_count: int = 2
    anchor_cos_threshold: float = 0.8
    l2_alpha: Optional[float] = 1.0
    inversion_num_steps: int = 10
    inversion_beam_width: int = 4


class TransformedEmbeddings(typing.NamedTuple):
    transformed_effect: np.ndarray
    transformed_anchor: np.ndarray
    weights: np.ndarray
    intercept: float


_SUMMARIZE_TEXT_PROMPT = """        
You will be given a short story. Please help to summarize the key events in the text to 5 or fewer sentences of less than 15 words each.
The events should be in chronological order, and the events should capture the key actors, location, causes, and effects of the event being described.
Return your answer in JSON as a array of strings in the key `result`.

Here is an example:
Text:
```
Once upon a time, there was an ugly frog. The ugly frog lived in a small pond. The frog liked to get things. He would get things from the bottom of the pond. One day, he saw a shiny weight.
The ugly frog wanted the shiny weight. He tried to get it, but it was too heavy. He tried and tried, but he could not get it. The ugly frog was sad. He wanted the shiny weight so much.
Then, a big fish came. The big fish saw the ugly frog and the shiny weight. The big fish wanted to help. The big fish and the ugly frog worked together to get the shiny weight. They were happy to have the shiny weight. They became good friends.
```

Answer:
```
{{
  "result": [
    "An ugly frog who liked to get things lived in a small pond.",
    "One day, the ugly frog saw a shiny weight, and wanted to get it, but could not.",
    "A big fish came, and the fish wanted to help the ugly frog get the shiny weight.",
    "The big fish worked together with the ugly frog to get the shiny weight.",
    "The fish and the frog were happy to get the weight and became good friends."
  ]
}}
```

Now your turn:
Text:
"{text}"
"""


_EVENT_QUESTIONS_PROMPT = """
Given two separate events:

------
Event A: "{event}"
------
Event B: "{test_event}"
------

Ignoring the specific characters, {question}? Provide your answer in JSON with the keys `is_similar` and `reasoning`.
"""


_ANONYMIZE_PROMPT = """
You will be given a story. Your job is to anonymize the names of persons, and replace them with a generic term. If there is nothing to anonymize, return the story as is.

For example, "Mary" should be replaced by "a girl", and "Tim" should be replaced by "a boy". 

Return your result as a string in the key `result` of a JSON object.

Now your turn:
Story: {event}
"""


def _llm_summarize_text_short(model: Model, text: str) -> List[str]:
    prompt = _SUMMARIZE_TEXT_PROMPT.format(text=text)
    return call_openai_structured(
        label=PromptLabel.SUMMARIZE_TEXT,
        prompt=prompt,
        struct=BasicArrayResultResponse,
        model=model,
    ).result


def _llm_anonymize_text(model: Model, text: str) -> str:
    prompt = _ANONYMIZE_PROMPT.format(event=text)
    return call_openai_structured(
        label=PromptLabel.ANONYMIZE_TEXT,
        prompt=prompt,
        struct=BasicResultResponse,
        model=model,
    ).result


def _llm_check_event_contains_or_similar(
    model: Model, event, test_event
) -> SimilarityResponseWithReasoning:
    questions = [
        "does a similar event to event B take place in event A",
        "is event B a subset of event A",
    ]

    responses = list(
        get_executor().map(
            lambda r: call_openai_structured(
                label=PromptLabel.CONTAINS_OR_SIMILAR,
                prompt=_EVENT_QUESTIONS_PROMPT.format(
                    event=event, test_event=test_event, question=r
                ),
                struct=SimilarityResponseWithReasoning,
                model=model,
            ),
            questions,
        )
    )

    reasoning = []
    count_true = 0
    for q, r in zip(questions, responses):
        if r.is_similar:
            count_true += 1
        reasoning.append(f"[{q}]: {r.is_similar} - {r.reasoning}")

    return SimilarityResponseWithReasoning(
        is_similar=count_true >= (len(questions) / 2), reasoning="\n".join(reasoning)
    )


def _check_sample_similarity(
    model: Model, test_case: COPESTestCase, sample: KeptSample
) -> KeptSample:
    executor = get_executor()

    llm_original_treatment_in_anchor = executor.submit(
        _llm_check_event_contains_or_similar,
        model,
        "\n".join(sample.anchor),
        test_case.test_event,
    )
    llm_original_treatment_in_effect = executor.submit(
        _llm_check_event_contains_or_similar,
        model,
        "\n".join(sample.effect),
        test_case.test_event,
    )
    llm_treatment_event_similar = executor.submit(
        _llm_check_event_contains_or_similar,
        model,
        sample.treatment,
        test_case.test_event,
    )
    _sample = deepcopy(sample)
    _sample.llm_original_treatment_in_anchor = llm_original_treatment_in_anchor.result()
    _sample.llm_original_treatment_in_effect = llm_original_treatment_in_effect.result()
    _sample.llm_treatment_event_similar = llm_treatment_event_similar.result()
    return _sample


def _samples_to_transformed_embeddings(
    model: Model,
    test_case: COPESTestCase,
    samples: List[KeptSample],
    l2_alpha: Optional[float],
) -> TransformedEmbeddings:
    all_kept_samples_anchor_matrix = [ks.anchor_embed for ks in samples]
    anonymized_original_anchor_text = _llm_anonymize_text(
        model, test_case.anchor_text_s()
    )
    anonymized_original_anchor_embeddings = call_openai_embeddings(
        [anonymized_original_anchor_text]
    )[0]

    X = np.transpose(all_kept_samples_anchor_matrix)
    y = anonymized_original_anchor_embeddings

    if l2_alpha is None:
        regressor = linear_model.LinearRegression()
    else:
        regressor = linear_model.Ridge(alpha=l2_alpha)

    lm = regressor.fit(X, y)
    coefs = lm.coef_
    intercept = lm.intercept_
    transformed_effect = np.array(
        [np.dot(coefs, [ks.effect_embed for ks in samples]) + intercept]
    ).astype(np.float32)
    transformed_anchor = np.array(
        [np.dot(coefs, [ks.anchor_embed for ks in samples]) + intercept]
    ).astype(np.float32)

    # effect_normalizing_factor = 1 / np.sqrt(np.sum(np.square(transformed_effect)))
    # anchor_normalizing_factor = 1 / np.sqrt(np.sum(np.square(transformed_anchor)))

    # normalized_transformed_effect = effect_normalizing_factor * transformed_effect
    # normalized_transformed_anchor = anchor_normalizing_factor * transformed_anchor

    return TransformedEmbeddings(
        transformed_effect, transformed_anchor, coefs, intercept
    )


class SyntheticControl(Strategy):
    def __init__(
        self, hyperparams: ScHyperParams, output_dir: str, model: Model = Model.GPT3
    ):
        self.model = model
        self.hyperparams = hyperparams
        self.corpus = Corpus(retrieval_size=hyperparams.corpus_retrieval_size)
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)

    def _retrieve_samples(
        self, test_case: COPESTestCase, logger: logging.Logger | List[logging.Logger]
    ) -> List[RetrievedCorpusDoc]:
        retrieval_query = " ".join(test_case.events)
        anonymized_retrieval_query = _llm_anonymize_text(self.model, retrieval_query)
        if isinstance(logger, logging.Logger):
            logger = [logger]

        for l in logger:
            l.info("Retrieval using query %s", anonymized_retrieval_query)

        retrieved = self.corpus.retrieve_similar(anonymized_retrieval_query)
        return retrieved

    def _get_samples(
        self,
        test_case: COPESTestCase,
        retrieved: List[RetrievedCorpusDoc],
        logger: logging.Logger,
    ) -> List[KeptSample]:
        executor = get_executor()
        chunk_size = 4
        retrieved_chunked = [
            retrieved[i : i + chunk_size] for i in range(0, len(retrieved), chunk_size)
        ]

        logger.info(
            "Retrieved %d, total chunks: %d", len(retrieved), len(retrieved_chunked)
        )
        r_idx = 0
        all_kept_samples: List[KeptSample] = []

        logger.info("_____ GETTING SAMPLES ______")
        while (
            r_idx < len(retrieved_chunked)
            and len(all_kept_samples) < self.hyperparams.kept_samples_count
        ):
            chunk = retrieved_chunked[r_idx]
            logger.info("_____ PROCESSING CORPUS IDs (%s) _____", [r.id for r in chunk])
            kept_samples = executor.map(
                self._process_retrieved_doc_for_samples,
                chunk,
                repeat(test_case),
                repeat(logger),
            )
            for ks in kept_samples:
                all_kept_samples.extend(ks)
            logger.info("_____ ALL KEPT SAMPLES _____")
            logger.info(all_kept_samples)
            r_idx += 1

        final_samples = all_kept_samples[: self.hyperparams.kept_samples_count]
        logger.info("_____ FINISHED SAMPLING ______")

        return final_samples

    def _async_embed_texts(self, texts: List[str]) -> Future[List[List[float]]]:
        fut = get_executor().submit(call_openai_embeddings, texts)
        return fut

    def _process_retrieved_doc_for_samples(
        self,
        retrieved: RetrievedCorpusDoc,
        test_case: COPESTestCase,
        logger: logging.Logger,
    ) -> List[KeptSample]:
        try:
            executor = get_executor()
            cleaned = clean_raw_text_from_corpus(retrieved.doc)

            summarized_retrieved_text = _llm_summarize_text_short(self.model, cleaned)

            anchor, treatment, effect = zip(
                *(
                    (
                        summarized_retrieved_text[:prefix_i],
                        summarized_retrieved_text[prefix_i],
                        summarized_retrieved_text[prefix_i + 1 :],
                    )
                    for prefix_i in range(1, len(summarized_retrieved_text) - 1)
                )
            )

            anchor_text = ["\n".join(a) for a in anchor]
            effect_text = ["\n".join(e) for e in effect]
            anonymized_anchor_text = executor.map(
                _llm_anonymize_text, repeat(self.model), anchor_text
            )
            anonymized_treatment_text = executor.map(
                _llm_anonymize_text, repeat(self.model), treatment
            )
            anonymized_effect_text = executor.map(
                _llm_anonymize_text, repeat(self.model), effect_text
            )

            anonymized_original_anchor_text = executor.submit(
                _llm_anonymize_text, self.model, test_case.anchor_text_s()
            )
            anonymized_original_treatment = executor.submit(
                _llm_anonymize_text, self.model, test_case.test_event
            )

            original_anchor_embeddings = self._async_embed_texts(
                [anonymized_original_anchor_text.result()]
            )
            original_treatment_embeddings = self._async_embed_texts(
                [anonymized_original_treatment.result()]
            )

            anonymized_anchor_text = list(anonymized_anchor_text)
            anonymized_treatment_text = list(anonymized_treatment_text)
            anonymized_effect_text = list(anonymized_effect_text)
            embedded_anchors = self._async_embed_texts(anonymized_anchor_text)
            embedded_treatments = self._async_embed_texts(anonymized_treatment_text)
            embedded_effects = self._async_embed_texts(anonymized_effect_text)

            treatment_cosine_scores = cos_sim(
                original_treatment_embeddings.result()[0], embedded_treatments.result()
            )
            treatment_cosine_scores = np.array(treatment_cosine_scores).reshape(-1)
            anchor_cosine_scores = cos_sim(
                original_anchor_embeddings.result()[0], embedded_anchors.result()
            )
            anchor_cosine_scores = np.array(anchor_cosine_scores).reshape(-1)

            all_samples = [
                KeptSample(
                    retrieved.id,
                    anchor[i],
                    anonymized_anchor_text[i],
                    embedded_anchors.result()[i],
                    anchor_cosine_scores[i],
                    treatment[i],
                    anonymized_treatment_text[i],
                    embedded_treatments.result()[i],
                    treatment_cosine_scores[i],
                    effect[i],
                    anonymized_effect_text[i],
                    embedded_effects.result()[i],
                    None,
                    None,
                    None,
                )
                for i in range(len(anchor))
            ]

            kept_samples = [
                ks
                for ks in all_samples
                if ks.anchor_cos > self.hyperparams.anchor_cos_threshold
            ]

            kept_samples: List[KeptSample] = list(
                get_executor().map(
                    _check_sample_similarity,
                    repeat(self.model),
                    repeat(test_case),
                    kept_samples,
                )
            )

            logger.info(
                "[%d] All Cosine Filtered Samples \n%s", retrieved.id, kept_samples
            )

            kept_samples = sorted(
                [
                    ks
                    for ks in kept_samples
                    if not ks.llm_original_treatment_in_anchor.is_similar
                    and not ks.llm_treatment_event_similar.is_similar
                    and not ks.llm_original_treatment_in_effect.is_similar
                ],
                key=lambda x: x.anchor_cos,
                reverse=True,
            )

            return kept_samples
        except Exception as e:
            logger.exception(
                "[%d] Error while processing sample, skipped", retrieved.id
            )
            return []

    def _create_logger_for_run(self, test_case: COPESTestCase) -> logging.Logger:
        logger_path = (
            self.output_dir
            + f"test_case_{test_case.id}_{test_case.event_id_to_test}.log"
        )
        return self._create_logger(
            logger_path, f"test_case_{test_case.id}_{test_case.event_id_to_test}"
        )

    def _create_logger(self, logger_path, logger_name):
        logger = logging.getLogger(logger_name)
        logger.propagate = False
        logger.setLevel(logging.INFO)
        ch = logging.FileHandler(logger_path, mode="w")
        ch.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(ch)
        return logger

    def _determine_causal(
        self,
        test_case: COPESTestCase,
        inverted_effect_result: str,
        inverted_anchor_result: str,
        samples: List[KeptSample],
    ) -> CausalResult:
        # anchor_similarity = _llm_check_event_contains_or_similar(
        #     self.model, inverted_anchor_result, test_case.anchor_text_s()
        # )

        effect_similarity = _llm_check_event_contains_or_similar(
            self.model, inverted_effect_result, test_case.effect()
        )

        # if anchor_similarity.is_similar:
        causal = not effect_similarity.is_similar
        # else:
        #   causal = "indeterminate"

        return CausalResult(
            causal,
            True,  # anchor_similarity.is_similar,
            inverted_effect_result,
            "",
            inverted_anchor_result,
            samples,
            test_case,
            effect_similarity.reasoning,
            "",  # anchor_similarity.reasoning,
        )

    def run(self, test_case: COPESTestCase) -> CausalResult:
        logger = self._create_logger_for_run(test_case)
        logger.info(test_case)

        retrieved = self._retrieve_samples(test_case, logger)
        samples = self._get_samples(test_case, retrieved, logger)

        if len(samples) < self.hyperparams.kept_samples_count:
            logger.info("_____ FAILED TO GET SUFFICIENT SAMPLES  ______")
            result = CausalResult(
                "indeterminate", False, "", "", "", samples, test_case, "", ""
            )
            logger.info(result)
            return result

        logger.info("_____ INVERTING ______")
        transformed_embeddings = _samples_to_transformed_embeddings(
            self.model, test_case, samples, self.hyperparams.l2_alpha
        )
        logger.info(
            "Inverted using weights: intercept=%.4f weights=%s",
            transformed_embeddings.intercept,
            transformed_embeddings.weights,
        )
        inverted_effect_result = invert_embeddings(
            transformed_embeddings.transformed_effect,
            num_steps=self.hyperparams.inversion_num_steps,
            sequence_beam_width=self.hyperparams.inversion_beam_width,
        )[0]
        inverted_anchor_result = invert_embeddings(
            transformed_embeddings.transformed_anchor,
            num_steps=self.hyperparams.inversion_num_steps,
            sequence_beam_width=self.hyperparams.inversion_beam_width,
        )[0]

        logger.info("_____ COMPARING ______")
        result = self._determine_causal(
            test_case,
            inverted_effect_result,
            inverted_anchor_result,
            samples,
        )
        logger.info(result)
        return result

    def run_all(self, test_case: COPESTestCase) -> AllCausalResults:
        test_cases: List[COPESTestCase] = []
        loggers: List[logging.Logger] = []
        for i in range(0, 4):
            tc = deepcopy(test_case)
            tc.event_id_to_test = i
            logger = self._create_logger_for_run(tc)
            logger.info(tc)
            test_cases.append(tc)
            loggers.append(logger)

        retrieved = self._retrieve_samples(test_case, loggers)

        all_samples = list(
            get_executor().map(
                self._get_samples, test_cases, repeat(retrieved), loggers
            )
        )

        filtered_samples: List[List[KeptSample]] = []
        filtered_test_cases: List[COPESTestCase] = []
        filtered_loggers: List[logging.Logger] = []
        indeterminate: List[CausalResult] = []
        for logger, samples, test_case in zip(loggers, all_samples, test_cases):
            if len(samples) < self.hyperparams.min_samples_count:
                logger.info("_____ FAILED TO GET SUFFICIENT SAMPLES  ______")
                result = CausalResult(
                    "indeterminate", False, "", "", "", samples, test_case, "", ""
                )
                logger.info(result)
                indeterminate.append(result)
            else:
                filtered_samples.append(samples)
                filtered_test_cases.append(test_case)
                filtered_loggers.append(logger)
                logger.info("_____ INVERTING ______")

        if len(filtered_test_cases) > 0:
            transformed_embeddings = list(
                map(
                    _samples_to_transformed_embeddings,
                    repeat(self.model),
                    filtered_test_cases,
                    filtered_samples,
                    repeat(self.hyperparams.l2_alpha),
                )
            )

            for logger, transformed in zip(filtered_loggers, transformed_embeddings):
                logger.info(
                    "Inverted using weights: intercept=%.4f weights=%s",
                    transformed.intercept,
                    transformed.weights,
                )

            effect_embeddings = np.array(
                list(
                    map(
                        lambda x: x.transformed_effect.ravel(),
                        transformed_embeddings,
                    )
                )
            )

            anchor_embeddings = np.array(
                list(
                    map(
                        lambda x: x.transformed_anchor.ravel(),
                        transformed_embeddings,
                    )
                )
            )

            inverted_effect = invert_embeddings(
                effect_embeddings,
                num_steps=self.hyperparams.inversion_num_steps,
                sequence_beam_width=self.hyperparams.inversion_beam_width,
            )

            inverted_anchor = invert_embeddings(
                anchor_embeddings,
                num_steps=self.hyperparams.inversion_num_steps,
                sequence_beam_width=self.hyperparams.inversion_beam_width,
            )
        else:
            inverted_effect = []
            inverted_anchor = []

        for logger in filtered_loggers:
            logger.info("_____ COMPARING ______")

        all_causal_results = list(
            get_executor().map(
                self._determine_causal,
                filtered_test_cases,
                inverted_effect,
                inverted_anchor,
                filtered_samples,
            )
        )

        for logger, result in zip(filtered_loggers, all_causal_results):
            logger.info(result)

        return AllCausalResults(filtered_test_cases, all_causal_results + indeterminate)
