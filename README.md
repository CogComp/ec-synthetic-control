# EC Synthetic Control

This repository contains the implementation of the approach discussed in [Event causality identification with synthetic control](https://aclanthology.org/2024.emnlp-main.103/).

The paper was presented at EMNLP 2024.

### Prereqs

- Create a `.env` file in the root of the directory with the `OPENAI_API_KEY` environment variable.
```
OPENAI_API_KEY="<your openai key>"
```
- Optionally, add Langfuse API keys to `.env` to enable tracing for OpenAI calls.
```
LANGFUSE_SECRET_KEY="<langfuse secret key>"
LANGFUSE_PUBLIC_KEY="<langfuse public key>"
LANGFUSE_HOST="<langfuse host>"
```
- Download the [COPES dataset](https://github.com/HKUST-KnowComp/COLA/blob/master/COPES_data/COPES.json) to `data/COPES.json`.
    - `curl -o data/COPES.json -LJ https://github.com/HKUST-KnowComp/COLA/raw/refs/heads/master/COPES_data/COPES.json`
- Download the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStoriesV2-GPT4-train.txt) to `data/TinyStoriesV2-GPT4-train.txt`.
    - `curl -o data/TinyStoriesV2-GPT4-train.txt -L https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt`
- `conda` needs to be installed.

### Setup
1. Create the virtual environment using `conda env create -f environment.yml`
2. Convert `TinyStoriesV2-GPT4-train.txt` to parquet, by running `python main.py setup-tiny-stories-parquet`
3. Create the BM25 index by running `python main.py setup-tiny-stories-corpus`


### Running

_Note that all indices are 0-indexed_

| Strategy | Description                                  |
|----------|----------------------------------------------|
| `gpt4`   | (Baseline) GPT4 Zeroshot Inference           |
| `sc`     | (Synthetic Control) GPT3.5 Synthetic Control |
| `sc4`    | (Synthetic Control) GPT4 Synthetic Control   |

Run outputs are logged in `output/<strategy>/`

`<test_case_id>` are all IDs from COPES.

#### Testing causality of a single event from a test case
`python main.py run-testcase-event <test_case_id> <event_id> <strategy>`

e.g. `python main.py run-testcase-event 0 0 sc`

#### Testing a full test case
`python main.py run-one <test_case_id> <strategy>`

#### Testing from a list of test case indices
`python main.py run_from_list <path_to_json> <strategy`

`path_to_json` must be a file containing a single JSON array of indexes (e.g. `[1,2,3,4]`)

#### Printing a list of test case indices
`python main.py print-testcases <path_to_json>`

### Known Issues
- Deadlocks have been observed to occasionally occur within DuckDB (or the Python DuckDB driver), causing corpus retrieval to fail.