from collections.abc import Callable
from typing import List

from utils.openai import (
    BasicResultResponse,
    PromptLabel,
    call_openai_structured,
    events_to_event_dict,
)

_AUGMENT_CONTEXT_PROMPT = """        
You will be given a sequence of events, numbered from 1 to 5. Your job is to try and understand the temporal sequencing of events, and generate a new sentence that came before this sequence of events.
Return this sentence with the key "result"

The new sentence should help to provide background or context on the events. Do not explain your answer, just return the sentence.

Here is an example:

1: Clarice really wanted a pet dog of her own.
2: She tried to convince her parents to buy her a dog.
3: Her parents refused. 
4: Clarice decided to move out and live on her own. 
5. Clarice bought herself a dog.

Answer: Clarice loved dogs since she was a kid.

Now your turn:

1: {event0}
2: {event1}
3: {event2}
4: {event3}
5: {event4}
"""


def default_context_augmentation(events: List[str]) -> str:
    events_input = events_to_event_dict(events)
    prompt = _AUGMENT_CONTEXT_PROMPT.format_map(events_input)
    result = call_openai_structured(
        PromptLabel.AUG_CONTEXT, struct=BasicResultResponse, prompt=prompt
    )
    return result.result
