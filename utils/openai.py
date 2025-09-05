from enum import StrEnum
from typing import List, Sequence, TypeVar

from openai import NOT_GIVEN
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

from utils.globals import cache, get_openai_client

T = TypeVar("T", bound=BaseModel)


class BasicResultResponse(BaseModel):
    result: str


class BasicArrayResultResponse(BaseModel):
    result: List[str]


class PromptLabel(StrEnum):
    GPT4_ZS = "gpt4-zeroshot"
    AUG_CONTEXT = "aug-context"
    SUMMARIZE_TEXT = "summarize-text"
    ANONYMIZE_TEXT = "anonymize-text"
    CONTAINS_OR_SIMILAR = "contains-or-similar"
    EMBEDDINGS = "embeddings"


class Model(StrEnum):
    GPT3 = "gpt-3.5-turbo-0125"
    GPT4 = "gpt-4-turbo-2024-04-09"


def events_to_event_dict(events: List[str]) -> dict:
    events_input = {}
    for i, e in enumerate(events):
        events_input[f"event{i}"] = e
    return events_input


def call_openai_embeddings(
    documents: List[str], model: str = "text-embedding-ada-002"
) -> List[List[float]]:
    return _call_embeddings_impl(documents, model)


def call_openai_structured(
    label: PromptLabel,
    prompt: str,
    struct: type[T],
    follow_up_prompts: List[str] = [],
    cache_results: bool = True,
    model: Model = Model.GPT3,
) -> T:
    method = _call_impl
    if cache_results:
        method = cache.memoize(typed=True, tag=label)(method)

    messages: List[ChatCompletionMessageParam] = [
        {"role": "user", "content": prompt},
    ]

    result = method(messages, model, json_mode=True).choices[0]

    for follow_up_prompt in follow_up_prompts:
        messages.append({"role": "assistant", "content": result.message.content})
        messages.append({"role": "user", "content": follow_up_prompt})
        result = method(messages, model, json_mode=True).choices[0]

    return struct.model_validate_json(result.message.content or "")


@cache.memoize(tag=PromptLabel.EMBEDDINGS)
def _call_embeddings_impl(documents: List[str], model: str) -> List[List[float]]:
    return list(
        map(
            lambda d: d.embedding,
            get_openai_client().embeddings.create(input=documents, model=model).data,
        )
    )


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def _call_impl(
    messages: Sequence[ChatCompletionMessageParam],
    model: Model,
    json_mode: bool,
    temperature: float = 0.0,
) -> ChatCompletion:
    _msgs: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant designed to output JSON."
                if json_mode
                else "You are a helpful assistant."
            ),
        }
    ]

    _msgs.extend(messages)

    return get_openai_client().chat.completions.create(
        model=model,
        response_format={"type": "json_object"} if json_mode else NOT_GIVEN,
        messages=_msgs,
        temperature=temperature,
    )
