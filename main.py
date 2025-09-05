from itertools import repeat
import json
import os
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from pathlib import Path
from typing import List

from rich import print
from rich.console import Console
from rich.status import Status
from typer import Typer

from impl.common import default_context_augmentation
from impl.gpt4_baseline import GPT4Baseline
from impl.strategy import Strategy
from impl.synthetic_control import ScHyperParams, SyntheticControl
from models.causal_result import AllCausalResults
from models.copes_test_case import COPESTestCase
from utils.fts_utils import convert_tiny_stories_parquet, setup_fts_tiny_stories
from utils.globals import cache, get_copes_dataset
from utils.openai import Model, PromptLabel
from diskcache import Index

console = Console()
_running_idx = Index()


class StrategyType(StrEnum):
    GPT4_ZEROSHOT = "gpt4"
    SYNTHETIC_CONTROL = "sc"
    SYNTHETIC_CONTROL_GPT4 = "sc4"


app = Typer()


def _preflight():
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY not found")


def _st_to_strategy(st: StrategyType, output_dir: str) -> Strategy:
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    match st:
        case StrategyType.GPT4_ZEROSHOT:
            return GPT4Baseline()
        case StrategyType.SYNTHETIC_CONTROL:
            return SyntheticControl(output_dir=output_dir, hyperparams=ScHyperParams())
        case StrategyType.SYNTHETIC_CONTROL_GPT4:
            return SyntheticControl(
                output_dir=output_dir, hyperparams=ScHyperParams(), model=Model.GPT4
            )
        case _:
            raise NotImplementedError()


def _run_all_with_strategy_with_status(
    status: Status, st: Strategy, test_case: COPESTestCase
):
    _running_idx[test_case.id] = True
    status.update(_get_curr_status())
    result = st.run_all(test_case)
    del _running_idx[test_case.id]
    status.update(_get_curr_status())
    return result


def _get_curr_status() -> str:
    return f"Running {list(_running_idx)}"


@app.command()
def clear_prompt_cache(prompt: PromptLabel):
    cache.evict(prompt)


@app.command()
def run_testcase_event(test_case: int, event_to_test: int, strategy: StrategyType):
    _preflight()
    dataset = get_copes_dataset()
    test = COPESTestCase(
        dataset[test_case],
        event_to_test,
        context_augmenter=default_context_augmentation,
    )

    console.rule()
    print(test)
    console.rule()

    output_dir = f"output/{strategy}/"
    st = _st_to_strategy(strategy, output_dir=output_dir)
    with console.status(f"Running {strategy}") as status:
        result = st.run(test)
        print(result)


@app.command()
def run_one(test_case: int, strategy: StrategyType):
    _preflight()
    dataset = get_copes_dataset()
    test = COPESTestCase(
        dataset[test_case],
        0,
        context_augmenter=default_context_augmentation,
    )

    console.rule()
    print(test)
    console.rule()

    output_dir = f"output/{strategy}/"
    st = _st_to_strategy(strategy, output_dir=output_dir)
    with console.status(f"Running {strategy}") as status:
        result = st.run(test)
        print(result)


@app.command()
def run_from_list(path: str, strategy: StrategyType, parallelism: int = 4):
    _preflight()
    dataset = get_copes_dataset()

    indices: List[int] = []
    with open(path, "r") as f:
        indices = json.load(f)

    console.rule()
    print(f"Total test cases: {len(indices)}")
    print(indices)
    console.rule()

    output_dir = f"output/{strategy}/"
    st = _st_to_strategy(strategy, output_dir=output_dir)

    with (
        open(output_dir + path + ".output", "w") as output_f,
        ThreadPoolExecutor(max_workers=parallelism) as pool,
        console.status(_get_curr_status()) as status,
    ):
        tests = [
            COPESTestCase(
                dataset[i],
                0,
                context_augmenter=default_context_augmentation,
            )
            for i in indices
        ]
        results = list(
            map(
                pool.submit,
                repeat(_run_all_with_strategy_with_status),
                repeat(status),
                repeat(st),
                tests,
            )
        )
        for r in results:
            result = r.result()
            print(result)
            output_f.write(json.dumps(result.as_dict()) + "\n")
            output_f.flush()


@app.command()
def print_testcases(path: str):
    _preflight()
    dataset = get_copes_dataset()

    indices: List[int] = []
    with open(path, "r") as f:
        indices = json.load(f)

    for i in indices:
        tc = COPESTestCase(
            dataset[i],
            0,
            context_augmenter=default_context_augmentation,
        )

        for cause_id in tc.cause_idx:
            console.rule(f"Index {i} - {cause_id}")
            print(
                COPESTestCase(
                    dataset[i],
                    cause_id,
                    context_augmenter=default_context_augmentation,
                )
            )


@app.command()
def setup_tiny_stories_parquet(output="data/tiny_stories_v2_gpt4.parquet"):
    if not os.path.exists("data/TinyStoriesV2-GPT4-train.txt"):
        print("data/TinyStoriesV2-GPT4-train.txt is not found")
        return
    with console.status(
        f"Converting data/TinyStoriesV2-GPT4-train.txt to {output}"
    ) as status:
        convert_tiny_stories_parquet(output=output)


@app.command()
def setup_tiny_stories_corpus():
    if not os.path.exists("data/tiny_stories_v2_gpt4.parquet"):
        print("data/tiny_stories_v2_gpt4.parquet is not found")
        return
    with console.status(f"Setting up TinyStories corpus") as status:
        setup_fts_tiny_stories()


if __name__ == "__main__":
    app()
