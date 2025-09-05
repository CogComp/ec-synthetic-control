import os
import threading
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import List

import dotenv
import duckdb
import langfuse
import langfuse.openai

# https://duckdb.org/docs/api/python/known_issues.html#numpy-import-multithreading
import numpy.core.multiarray
from diskcache import FanoutCache
from joblib import Memory
from openai import OpenAI

from utils.dataset_utils import read_copes_dataset_dict

dotenv.load_dotenv()

inversion_embeddings_memory = Memory(
    location="./.cache/embeddings", mmap_mode="r", verbose=1
)
cache = FanoutCache(directory="./.cache/dc", tag_index=True)
_db_ro_con = None
_openai = None
_copes_dataset = None
_executor = None
_glock = threading.Lock()
_tlocal = threading.local()


def get_copes_dataset() -> List[dict]:
    global _copes_dataset, _glock
    if _copes_dataset is not None:
        return _copes_dataset
    with _glock:
        if _copes_dataset is None:
            _copes_dataset = read_copes_dataset_dict()
    return _copes_dataset


def get_duckdb_con(path="data/corpus.db") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(path)
    return con


def get_duckdb_ro_con(path="data/corpus.db") -> duckdb.DuckDBPyConnection:
    global _db_ro_con, _glock
    if _db_ro_con is not None:
        if "con" not in _tlocal.__dict__:
            print(f"[{threading.get_ident()}] Connecting to duckdb")
            _tlocal.con = _db_ro_con.cursor()
        return _tlocal.con
    with _glock:
        if _db_ro_con is None:
            print("Initializing duckdb")
            con = duckdb.connect(
                path, read_only=True, config={"access_mode": "READ_ONLY"}
            )
            con.execute(
                """
                SET autoinstall_known_extensions=false;
                SET autoload_known_extensions=false;
                INSTALL fts;
                LOAD fts;                
                """
            )
            _db_ro_con = con
        print(f"[{threading.get_ident()}] Connecting to duckdb")
        _tlocal.con = _db_ro_con.cursor()
    return _tlocal.con


def get_openai_client() -> OpenAI:
    global _openai, _glock
    if _openai is not None:
        return _openai
    with _glock:
        if _openai is None:
            if "LANGFUSE_SECRET_KEY" in os.environ:
                print("Using Langfuse")
                _openai = langfuse.openai.OpenAI()
            else:
                _openai = OpenAI()
    return _openai


def get_executor() -> Executor:
    global _executor, _glock
    if _executor is not None:
        return _executor

    with _glock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=2000
            )  # arbitrary high number to make sure we don't deadlock
    return _executor
