import json
from pathlib import Path

import lancedb
from lancedb.rerankers import ColbertReranker

from dreamai.lance_utils import add_to_lance_table, search_lancedb
from dreamai.md_utils import data_to_md

lance_db = lancedb.connect("example_rag")
file = "loan1.pdf"
table_name = "loan_table"

md_data = data_to_md(data=file, chunk_size=800, chunk_overlap=200, min_chunk_size=100)[0]
table = add_to_lance_table(
    db=lance_db,
    table_name=table_name,
    data=md_data.chunks,
    ems_model="hkunlp/instructor-base",
    ems_model_device="cuda",
)
res = search_lancedb(
    db=lance_db,
    table_name=table_name,
    query=["Financial Covenant"],
    reranker=ColbertReranker(model_name="answerdotai/answerai-colbert-small-v1"),
    max_search_results=10,
)
Path("fin_cov_res.json").write_text(json.dumps(res))
