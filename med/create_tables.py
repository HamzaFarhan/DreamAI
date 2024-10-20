import json
from pathlib import Path

import lancedb

from dreamai.lance_utils import add_to_lance_table
from dreamai.md_utils import MarkdownChunk

lance_db = lancedb.connect("med_rag")
ems_model = "hkunlp/instructor-base"

med_dir = Path("/media/hamza/data2/med_codes")
for file in med_dir.glob("*.json"):
    data = [
        MarkdownChunk(
            name=code["code"],
            index=i,
            text=f"code: {code['code']}\ndescriprion: {code['desc']}",
        )
        for i, code in enumerate(json.loads(Path(file).read_text()))
    ]
    table = add_to_lance_table(
        db=lance_db,
        table_name=file.stem,
        data=[
            MarkdownChunk(
                name=code["code"],
                index=i,
                text=f"code: {code['code']}\ndescriprion: {code['desc']}",
            )
            for i, code in enumerate(json.loads(Path(file).read_text()))
        ],
        ems_model=ems_model,
    )
