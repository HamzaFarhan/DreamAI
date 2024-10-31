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
ATTEMPTS = 6


def get_md_file(doc_file: str | Path) -> Path:
    return Path(doc_file).with_suffix(".md")


def get_chunks_file(doc_file: str | Path) -> Path:
    doc_file = Path(doc_file)
    return doc_file.with_name(doc_file.stem + "_chunks.json")


def get_bm25_file(doc_file: str | Path) -> Path:
    doc_file = Path(doc_file)
    return doc_file.with_name(doc_file.stem + "_bm25.json")


def get_section_llm_file(doc_file: str | Path, section_name: str) -> Path:
    doc_file = Path(doc_file)
    return doc_file.with_name(doc_file.stem + f"_{section_name}_llm.json")


def get_section_indexes_file(doc_file: str | Path, section_name: str) -> Path:
    doc_file = Path(doc_file)
    return doc_file.with_name(doc_file.stem + f"_{section_name}_indexes.json")


def get_section_html_file(doc_file: str | Path, section_name: str) -> Path:
    doc_file = Path(doc_file)
    return doc_file.with_name(doc_file.stem + f"_{section_name}.html")


def get_keywords_file(guidelines_file: str | Path) -> Path:
    guidelines_file = Path(guidelines_file)
    return guidelines_file.with_stem(guidelines_file.stem + "_keywords")


class SectionIndexes(BaseModel):
    indexes: list[int] = Field(default_factory=list)

    @field_validator("indexes")
    @classmethod
    def validate_indexes(cls, indexes: list[int], info: ValidationInfo) -> list[int]:
        logger.info(f"Initial Chunks: {indexes}\nCount: {len(indexes)}")

        if len(indexes) == 0:
            return []
        min_index_gap = info.context.get("min_index_gap", MIN_INDEX_GAP) if info.context else MIN_INDEX_GAP
        context_indexes = set(info.context.get("indexes", indexes) if info.context else indexes)
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


class Keywords(BaseModel):
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


def resolve_data_path(data_path: list[str | Path] | str | Path, file_extension: str | None = None) -> chain:
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


def chunk_doc(doc_file: str | Path, overwrite: bool = False):
    doc_file = Path(doc_file)
    md_file = get_md_file(doc_file=doc_file)
    chunks_file = get_chunks_file(doc_file=doc_file)
    if md_file.exists():
        md = md_file.read_text()
    else:
        md = docs_to_md(docs=doc_file)[0]
        md_file.write_text(md)
    should_create_chunks = not chunks_file.exists() or (chunks_file.exists() and overwrite)
    if should_create_chunks:
        chunks = chunk_md(md=md)
        chunks_file.write_text(json.dumps(chunks))
        if overwrite:
            logger.warning(f"Overwriting existing chunks file: {chunks_file.name}")
    else:
        logger.warning(f"{chunks_file.name} already exists and overwrite is False")
        chunks = json.loads(chunks_file.read_text())
    lens = [len(chunk["text"]) for chunk in chunks.values()]
    logger.info(f"Length of markdown: {len(md)}")
    logger.info(f"Number of chunks: {len(chunks)}")
    logger.info(f"Maximum chunk length: {max(lens)}")
    logger.info(f"Minimum chunk length: {min(lens)}")
    logger.info(f"Average chunk length: {np.mean(lens):.2f}")


def find_section(
    doc_file: str | Path,
    guidelines_file: str | Path,
    section_name: str,
    model: ModelName,
    kg_file: str | Path | None = None,
):
    doc_file = Path(doc_file)
    chunks_file = get_chunks_file(doc_file=doc_file)
    if not chunks_file.exists():
        chunk_doc(doc_file=doc_file)
    chunks = {int(k): v for k, v in json.loads(chunks_file.read_text()).items()}
    guidelines = json.loads(Path(guidelines_file).read_text())
    section_guidelines = {section_name: guidelines[section_name]}
    logger.info(f"Finding Section: {section_name}")
    chat_history = [user_message(content=f"<chunks>\n{json.dumps(chunks)}\n</chunks>")]
    if kg_file:
        kg_rels = json.loads(Path(kg_file).read_text())[section_name]
        chat_history.append(
            user_message(content=f"<kg_relationships>\n{json.dumps(kg_rels)}\n</kg_relationships>")
        )
    dialog = Dialog(task="find_section_task.txt", chat_history=chat_history)
    creator = create_creator(model=model)
    kwargs = dialog.gemini_kwargs(
        user=f"<section_guidelines>\n{json.dumps(section_guidelines)}\n</section_guidelines>"
    )
    logger.info(kwargs["messages"][-1]["content"][-100:])
    try:
        indexes = creator.create(response_model=list[int], max_retries=ATTEMPTS, **kwargs)
    except Exception:
        indexes = list(range(135, 165))
    indexes = SectionIndexes.model_validate(
        {"indexes": indexes},  # type:ignore
        context={"indexes": list(chunks.keys())},
    ).indexes
    logger.success(f"Found: {len(indexes)} chunks")  # type:ignore
    section_file = get_section_llm_file(doc_file=doc_file, section_name=section_name)
    section_file.write_text(json.dumps(indexes))


def extract_guideline_keywords(guidelines_file: str | Path, model: ModelName, overwrite: bool = False):
    logger.info("Extracting Guideline Keywords")
    guidelines_file = Path(guidelines_file)
    keywords_file = get_keywords_file(guidelines_file=guidelines_file)
    if keywords_file.exists() and not overwrite:
        logger.warning(f"{keywords_file.name} already exists and overwrite is False")
        return
    keywords = {}
    creator = create_creator(model=model)
    dialog = Dialog(task="section_kw_task.txt")
    for section_name, section_guidelines in (json.loads(guidelines_file.read_text())).items():
        if section_name in keywords and not overwrite:
            logger.warning(f"{section_name} already exists and overwrite is False")
            continue
        logger.info(f"Extracting Keywords for: {section_name}")
        kwargs = dialog.gemini_kwargs(
            user=f"<section_name>\n{section_name}\n</section_name>\n<guidelines>\n{section_guidelines}\n</guidelines>"
        )
        keywords[section_name] = creator.create(response_model=Keywords, **kwargs).keywords  # type:ignore
        keywords_file.write_text(json.dumps(keywords))
    logger.success("Extracted Keywords")


def bm25_search(doc_file: str | Path, guidelines_file: str | Path, k: int = 5, overwrite: bool = False):
    logger.info("Running BM25 Search")
    guidelines_file = Path(guidelines_file)
    keywords_file = get_keywords_file(guidelines_file=guidelines_file)
    doc_file = Path(doc_file)
    chunks_file = get_chunks_file(doc_file=doc_file)
    bm25_file = get_bm25_file(doc_file=doc_file)
    if bm25_file.exists() and not overwrite:
        logger.warning(f"{bm25_file} already exists. Skipping BM25 search.")
        return
    if not chunks_file.exists():
        chunk_doc(doc_file=doc_file)
    chunks = json.loads(chunks_file.read_text())
    corpus = [chunks[str(i)]["text"] for i in range(len(chunks))]
    keywords = json.loads(keywords_file.read_text())
    bm25_results = json.loads(bm25_file.read_text()) if bm25_file.exists() else {}
    for section_name, queries in keywords.items():
        if section_name in bm25_results and not overwrite:
            logger.warning(f"{section_name} already has BM25 results. Skipping.")
            continue
        logger.info(f"Running BM25 for section: {section_name}")
        # stemmer = Stemmer.Stemmer("english")
        stemmer = None
        corpus_tokens = bm25s.tokenize(corpus, stemmer=stemmer)  # type:ignore
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        query_tokens = bm25s.tokenize(queries, stemmer=stemmer)  # type:ignore
        res = retriever.retrieve(query_tokens, k=k, return_as="documents")
        bm25_results[section_name] = res.tolist()  # type:ignore
        bm25_file.write_text(json.dumps(bm25_results))
    logger.success(f"BM25 Results saved to: {bm25_file}")


def combine_results(doc_file: str | Path, section_name: str):
    doc_file = Path(doc_file)
    chunks = json.loads(get_chunks_file(doc_file=doc_file).read_text())
    section_llm = json.loads(get_section_llm_file(doc_file=doc_file, section_name=section_name).read_text())
    bm25_results = np.array(json.loads(get_bm25_file(doc_file=doc_file).read_text())[section_name])
    section_indexes = np.unique(np.concatenate([bm25_results.ravel(), section_llm]))
    section_indexes = SectionIndexes.model_validate(
        {"indexes": section_indexes},
        context={"indexes": [int(key) for key in chunks.keys()]},
    ).indexes
    logger.success(f"Combined Results for: {section_name}")
    logger.info((len(bm25_results), np.array(bm25_results).shape))
    logger.info((len(section_indexes), len(chunks)))
    logger.info(section_indexes)
    get_section_indexes_file(doc_file=doc_file, section_name=section_name).write_text(json.dumps(section_indexes))


def highlight_doc(doc_file: str | Path, section_name: str, overwrite: bool = True):
    doc_file = Path(doc_file)
    md = get_md_file(doc_file=doc_file).read_text()
    chunks = json.loads(get_chunks_file(doc_file=doc_file).read_text())
    indexes = json.loads(get_section_indexes_file(doc_file=doc_file, section_name=section_name).read_text())
    html_path = get_section_html_file(doc_file=doc_file, section_name=section_name)
    if not overwrite:
        logger.warning(f"{html_path.name} already exists and overwrite is False")
        return
    html_path.unlink(missing_ok=True)
    for index in indexes:
        # logger.info(f"start: {chunks[str(index)]['start']}, end: {chunks[str(index)]['end']}")
        md = re.sub(
            r"\n",
            "<br>",
            re.sub(
                r"\u201d",
                '"',
                re.sub(
                    r"\u201c",
                    '"',
                    insert_xml_tag(
                        text=md, tag="mark", start=chunks[str(index)]["start"], end=chunks[str(index)]["end"]
                    ),
                ),
            ),
        )
        html_path.write_text(f'<center style="font-size: 18px;">{md}</center>')
    logger.success(f"Highlighted Document for: {section_name}")


if __name__ == "__main__":
    guidelines_file = Path("guidelines.json")
    doc_file = Path("data/hp.md")
    section_name = "financial_covenants"
    extract_guideline_keywords(guidelines_file=guidelines_file, model=ModelName.GEMINI_FLASH)
    chunk_doc(doc_file=doc_file)
    bm25_search(doc_file=doc_file, guidelines_file=guidelines_file)
    find_section(
        doc_file=doc_file,
        guidelines_file=guidelines_file,
        section_name=section_name,
        model=ModelName.GEMINI_FLASH,
        kg_file="hp_res.json",
    )
    combine_results(doc_file=doc_file, section_name=section_name)
    highlight_doc(doc_file=doc_file, section_name=section_name)
