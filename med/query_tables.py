import json
from pathlib import Path

import lancedb
from lancedb.rerankers import ColbertReranker

from dreamai.lance_utils import search_lancedb

lance_db = lancedb.connect("med_rag")
reranker = ColbertReranker(model_name="answerdotai/answerai-colbert-small-v1")

file = Path("charts/chart1.json")
kws = json.loads(file.read_text())
res = []
for table_name in lance_db.table_names():
    res.append(
        search_lancedb(
            db=lance_db,
            table_name=table_name,
            query=kws,
            reranker=reranker,
            max_search_results=10,
        )
    )
file.with_stem(file.stem + "_retrieved").write_text(json.dumps(res))
