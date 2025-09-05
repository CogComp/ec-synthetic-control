from dataclasses import dataclass
from typing import List, Optional

from utils import globals
from utils.globals import cache


@dataclass
class RetrievedCorpusDoc:
    id: int
    doc: str
    score: Optional[float] = None


@cache.memoize(tag="tinystories-corpus")
def _retrieve_similar(index, table, retrieval_size, text: str):
    results = (
        globals.get_duckdb_ro_con()
        .execute(
            f"""
    SELECT id, corpus_entry, score
    FROM (
        SELECT *, {index}.match_bm25(
            id,
            ?,
            fields := 'corpus_entry'
        ) AS score
        FROM {table}
    ) sq
    WHERE score IS NOT NULL
    ORDER BY score DESC LIMIT ?;
    """,
            [text, retrieval_size],
        )
        .fetchall()
    )

    return [RetrievedCorpusDoc(*r) for r in results]


class Corpus:
    def __init__(self, retrieval_size=100):
        self.retrieval_size = 100
        self.index = "fts_main_tiny_stories"
        self.table = "tiny_stories"

    def retrieve_by_id(self, id: int):
        result = (
            globals.get_duckdb_ro_con()
            .execute(
                f"""
            SELECT id, corpus_entry FROM {self.table} WHERE id = ?
            """,
                [id],
            )
            .fetchone()
        )
        return RetrievedCorpusDoc(*result)

    def retrieve_similar(self, text: str) -> List[RetrievedCorpusDoc]:
        return _retrieve_similar(self.index, self.table, self.retrieval_size, text)
