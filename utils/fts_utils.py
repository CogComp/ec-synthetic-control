import pandas as pd
from .globals import get_duckdb_con


def clean_raw_text_from_corpus(text, max=6):
    replaced = text.replace("\\n", "\n\n").replace('\\"', '"')
    split = replaced.split("\n\n")

    if len(split) <= 6:
        return replaced

    # last sentence is some random info
    if len(split[-1]) < 25:
        split = split[:-1]

    res = []
    res.extend(split[: max // 2])
    res.append("...")
    res.extend(split[-max // 2 :])
    return "\n\n".join(res)


def setup_fts_nyt():
    """
    Generate the corpus txt with `jq '.corpus | .[]' nyt_corpus_dict.json > nyt_corpus_txt.csv`
    """
    tables = get_duckdb_con().sql("SHOW TABLES").fetchall()
    found = False
    for r in tables:
        if "nyt_corpus_txt" in r:
            found = True
            break

    if not found:
        get_duckdb_con().sql(
            """
        create or replace table nyt_corpus_txt as select row_number() over () as id, trim(corpus_entry, '"') as corpus_entry from read_csv('nyt_corpus_txt.csv',
            header = false,
            columns = {
                'corpus_entry': 'VARCHAR'
            });
        """
        )

        get_duckdb_con().sql(
            """
            pragma create_fts_index(
                'nyt_corpus_txt', 'id', 'corpus_entry'
            );
        """
        )


def setup_fts_simple_wikipedia():
    """
    https://huggingface.co/datasets/wikipedia/tree/main/data/20220301.simple
    """
    tables = get_duckdb_con().sql("SHOW TABLES").fetchall()
    found = False
    for r in tables:
        if "wikipedia_en_simple" in r:
            found = True
            break

    if not found:
        get_duckdb_con().sql(
            """
        create or replace table wikipedia_en_simple as select cast(id as int) as id, trim(text, '"') as corpus_entry from 'wikipedia_data_20220301_simple_train-00000-of-00001.parquet';
        """
        )

        get_duckdb_con().sql(
            """
            pragma create_fts_index(
                'wikipedia_en_simple', 'id', 'corpus_entry'
            );
        """
        )


def convert_tiny_stories_parquet(output):
    with open("data/TinyStoriesV2-GPT4-train.txt") as f:
        text = f.read()

    df = pd.DataFrame({"text": text.split("<|endoftext|>")})
    df.to_parquet(output, compression="zstd")


def setup_fts_tiny_stories():
    """
    https://huggingface.co/datasets/roneneldan/TinyStories
    """
    con = get_duckdb_con()
    tables = con.sql("SHOW TABLES").fetchall()
    found = False
    for r in tables:
        if "tiny_stories" in r:
            found = True
            break

    if not found:
        con.sql(
            """
        create or replace table tiny_stories as select row_number() over () as id, trim(text, '"') as corpus_entry from 'data/tiny_stories_v2_gpt4.parquet';
        """
        )

        con.sql(
            """
            pragma create_fts_index(
                'tiny_stories', 'id', 'corpus_entry'
            );
        """
        )

        con.sql(
            """
            CHECKPOINT;
            """
        )

        con.close()
