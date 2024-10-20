import json
import re
import tempfile
from itertools import chain
from pathlib import Path
from typing import Iterable

import bm25s
import numpy as np
import pymupdf4llm
from loguru import logger
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from dreamai.ai import ModelName, create_creator
from dreamai.dialog import Dialog, user_message
from dreamai.utils import insert_xml_tag

MIN_INDEX_GAP = 5
MAX_NUM_INDEXES = 15
MIN_CHUNK_SIZE = 600

guidelines_file = "guidelines.json"
all_guidelines = json.loads(Path(guidelines_file).read_text())


class SectionIndexes(BaseModel):
    indexes: list[int] = Field(default_factory=list)

    @field_validator("indexes")
    @classmethod
    def validate_indexes(cls, indexes: list[int], info: ValidationInfo) -> list[int]:
        logger.info(f"Initial Chunks: {indexes}\nCount: {len(indexes)}")

        if len(indexes) == 0:
            return []
        min_index_gap = (
            info.context.get("min_index_gap", MIN_INDEX_GAP) if info.context else MIN_INDEX_GAP
        )
        context_indexes = set(
            info.context.get("indexes", indexes) if info.context else indexes
        )
        sorted_indexes = sorted(set(indexes) & context_indexes)
        filled_indexes = []
        for i, current in enumerate(sorted_indexes):
            if not filled_indexes or current - filled_indexes[-1] > min_index_gap:
                filled_indexes.append(current)
            else:
                filled_indexes.extend(range(filled_indexes[-1] + 1, current + 1))
        result = []
        group_start = 0
        for i in range(1, len(filled_indexes)):
            if filled_indexes[i] - filled_indexes[i - 1] > 1:
                if i - group_start > 1:
                    result.extend(filled_indexes[group_start:i])
                group_start = i
        if len(filled_indexes) - group_start > 1:
            result.extend(filled_indexes[group_start:])

        logger.info(f"Final Chunks: {result}\nCount: {len(result)}")
        return result


class KeyWords(BaseModel):
    keywords: list[str] = Field(min_length=10)

    @field_validator("keywords")
    @classmethod
    def validate_keywords(cls, keywords: list[str]) -> list[str]:
        return list(set(keywords))


def flatten(o: Iterable):
    for item in o:
        if isinstance(item, str):
            yield item
            continue
        try:
            yield from flatten(item)
        except TypeError:
            yield item


def resolve_data_path(
    data_path: list[str | Path] | str | Path, file_extension: str | None = None
) -> chain:
    if not isinstance(data_path, list):
        data_path = [data_path]
    paths = []
    for dp in flatten(data_path):
        if isinstance(dp, (str, Path)):
            dp = Path(dp)
            if not dp.exists():
                raise Exception(f"Path {dp} does not exist.")
            if dp.is_dir():
                if file_extension:
                    paths.append(dp.glob(f"*.{file_extension}"))
                else:
                    paths.append(dp.iterdir())
            else:
                if file_extension is None or dp.suffix == f".{file_extension}":
                    paths.append([dp])
    return chain(*paths)


def docs_to_md(docs: list[str | Path] | str | Path) -> list[str]:
    docs_md = []
    for doc in resolve_data_path(data_path=docs):
        doc = Path(doc)
        if not doc.exists():
            md = str(doc)
        elif doc.suffix in [".md", ".txt"]:
            md = doc.read_text()
        elif doc.suffix == ".pdf":
            with tempfile.TemporaryDirectory() as image_folder:
                md = pymupdf4llm.to_markdown(
                    doc=str(doc),
                    write_images=True,
                    image_path=image_folder,
                    table_strategy="lines",
                )
        else:
            md = str(doc)
        docs_md.append(md)
    return docs_md


def split_text(text: str, separators: list[str] | None = None) -> tuple[list[str], str]:
    separators = separators or [r"#{1,6}\s+.+", r"\*\*.*?\*\*", r"---{3,}"]
    pattern = f'({"|".join(separators)})'
    return [chunk.strip() for chunk in re.split(pattern, text) if chunk.strip()], pattern


def chunk_md(
    md: str, separators: list[str] | None = None, min_chunk_size: int = MIN_CHUNK_SIZE
) -> dict[int, dict]:
    splits, pattern = split_text(text=md, separators=separators)
    if len(splits) == 1:
        splits, _ = split_text(text=md, separators=[r"\n"])
        pattern = ""
    logger.info(f"Num Splits: {len(splits)}")
    chunks = []
    current_chunk = ""
    start = 0
    for split in splits:
        if current_chunk:
            if re.match(pattern, split) or not pattern:
                end = min(len(md), start + len(current_chunk))
                chunks.append({"start": start, "end": end, "text": current_chunk})
                current_chunk = ""
                start = end
        current_chunk += split
    if current_chunk:
        end = min(len(md), start + len(current_chunk))
        chunks.append({"start": start, "end": end, "text": current_chunk})
    final_chunks: dict[int, dict] = {0: chunks[0]}
    for chunk in chunks[1:]:
        if len(chunk["text"]) > min_chunk_size:
            final_chunks[len(final_chunks)] = chunk
        else:
            final_chunks[len(final_chunks) - 1]["end"] = chunk["end"]
            final_chunks[len(final_chunks) - 1]["text"] += " " + chunk["text"]
    return final_chunks


def chunk_doc(doc_file: str | Path):
    doc_file = Path(doc_file)
    doc_name = doc_file.stem
    md_path = Path(f"{doc_name}_md.json")
    chunks_path = Path(f"{doc_name}_chunks.json")
    if md_path.exists() and chunks_path.exists():
        return
    md = docs_to_md(docs=doc_file)[0]
    chunks = chunk_md(md=md)
    lens = [len(chunk["text"]) for chunk in chunks.values()]
    logger.info(f"Length of markdown: {len(md)}")
    logger.info(f"Number of chunks: {len(chunks)}")
    logger.info(f"Maximum chunk length: {max(lens)}")
    logger.info(f"Minimum chunk length: {min(lens)}")
    logger.info(f"Average chunk length: {np.mean(lens):.2f}")
    Path(md_path).write_text(json.dumps(md))
    chunks_path.write_text(json.dumps(chunks))


def find_section(doc_name: str, section_guidelines: dict, model: ModelName) -> str:
    section_name = list(section_guidelines.keys())[0]
    logger.info(f"Finding Section: {section_name}")
    chunks = json.loads(Path(f"{doc_name}_chunks.json").read_text())
    dialog = Dialog(
        task="find_section_task.txt",
        chat_history=[user_message(content=f"<chunks>\n{json.dumps(chunks)}\n</chunks>")],
    )
    creator = create_creator(model=model)
    kwargs = dialog.gemini_kwargs(user=json.dumps(section_guidelines))
    logger.info(kwargs["messages"][-1]["content"][-100:])
    indexes = creator.create(response_model=list[int], max_retries=5, **kwargs)
    # indexes = SectionIndexes.model_validate(
    #     {"indexes": indexes},
    #     context={"indexes": [int(key) for key in chunks.keys()]},
    # ).indexes
    logger.success(f"Found: {len(indexes)} chunks")  # type:ignore
    res_path = f"found_{section_name}.json"
    Path(res_path).write_text(json.dumps(indexes))
    return res_path


def extract_keywords(all_guidelines: dict, model: ModelName) -> str:
    logger.info("Extracting Keywords")
    creator = create_creator(model=model)
    dialog = Dialog(task="section_kw_task.txt")
    res_path = Path("keywords.json")
    keywords = json.loads(res_path.read_text()) if res_path.exists() else {}
    for section_name, guidelines in all_guidelines.items():
        if section_name in keywords:
            continue
        logger.info(f"Extracting Keywords for: {section_name}")
        kwargs = dialog.gemini_kwargs(
            user=f"<section_name>\n{section_name}\n</section_name>\n<guidelines>\n{guidelines}\n</guidelines>"
        )
        keywords[section_name] = creator.create(response_model=KeyWords, **kwargs).keywords  # type:ignore
        res_path.write_text(json.dumps(keywords))
    logger.success("Extracted Keywords")
    return str(res_path)


def bm25_search(doc_name: str, section_name: str, k: int = 5) -> str:
    logger.info(f"Running BM25 for: {section_name}")
    chunks = json.loads(Path(f"{doc_name}_chunks.json").read_text())
    corpus = [chunks[str(i)]["text"] for i in range(len(chunks))]
    queries = json.loads(Path("keywords.json").read_text())[section_name]
    # stemmer = Stemmer.Stemmer("english")
    stemmer = None
    corpus_tokens = bm25s.tokenize(corpus, stemmer=stemmer)  # type:ignore
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    query_tokens = bm25s.tokenize(queries, stemmer=stemmer)  # type:ignore
    res = retriever.retrieve(query_tokens, k=k, return_as="documents")
    res_path = f"{section_name}_bm25_results.json"
    Path(res_path).write_text(json.dumps(res.tolist()))  # type:ignore
    logger.success(f"BM25 Results:\n{res}")
    return res_path


def combine_results(doc_name: str, section_name: str) -> str:
    chunks = json.loads(Path(f"{doc_name}_chunks.json").read_text())
    found_section = json.loads(Path(f"found_{section_name}.json").read_text())
    bm25_results = json.loads(Path(f"{section_name}_bm25_results.json").read_text())
    section_indexes = np.unique(np.concatenate([np.unique(bm25_results), found_section]))
    section_indexes = SectionIndexes.model_validate(
        {"indexes": section_indexes},
        context={"indexes": [int(key) for key in chunks.keys()]},
    ).indexes
    logger.success(f"Combined Results for: {section_name}")
    logger.info((len(bm25_results), np.array(bm25_results).shape))
    logger.info((len(section_indexes), len(chunks)))
    logger.info(section_indexes)
    res_path = f"{section_name}_indexes.json"
    Path(res_path).write_text(json.dumps(section_indexes))
    return res_path


def highlight_doc(doc_name: str, section_name: str) -> str:
    md = Path(f"{doc_name}_md.json").read_text()
    chunks = json.loads(Path(f"{doc_name}_chunks.json").read_text())
    indexes = json.loads(Path(f"{section_name}_indexes.json").read_text())
    html_path = Path(f"{section_name}.html")
    html_path.unlink(missing_ok=True)
    for index in indexes:
        # logger.info(f"start: {chunks[str(index)]['start']}, end: {chunks[str(index)]['end']}")
        md = (
            insert_xml_tag(
                text=md,
                tag="mark",
                start=chunks[str(index)]["start"],
                end=chunks[str(index)]["end"],
            )
            .replace("\\n", "<br>")
            .replace("\\u201c", '"')
            .replace("\\u201d", '"')
        )
        html_path.write_text(f'<center style="font-size: 18px;">{md}</center>')
    logger.success(f"Highlighted Document for: {section_name}")
    return str(html_path)


if __name__ == "__main__":
    doc_file = "hp.md"
    doc_name = Path(doc_file).stem
    section_name = "deal_general_information"
    extract_keywords(all_guidelines=all_guidelines, model=ModelName.GEMINI_FLASH)
    chunk_doc(doc_file=doc_file)
    section_guidelines = {section_name: all_guidelines[section_name]}
    found_section_file = find_section(
        doc_name=doc_name,
        section_guidelines=section_guidelines,
        model=ModelName.GEMINI_FLASH,
    )
    bm25_results_file = bm25_search(doc_name=doc_name, section_name=section_name)
    section_indexes_file = combine_results(doc_name=doc_name, section_name=section_name)
    html_path = highlight_doc(doc_name=doc_name, section_name=section_name)
