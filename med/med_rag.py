import json
import re
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import lancedb
from lancedb.db import DBConnection as LanceDBConnection
from lancedb.rerankers import ColbertReranker, Reranker
from loguru import logger
from pydantic import AfterValidator, BaseModel, Field

from dreamai.ai import ModelName
from dreamai.dialog import Dialog
from dreamai.lance_utils import add_to_lance_table, search_lancedb
from dreamai.md_utils import MarkdownChunk

MODEL = ModelName.SONNET
MAX_SEARCH_RESULTS = 100


date_type = Annotated[date, AfterValidator(lambda d: d.strftime("%Y-%m-%d"))]
reasoning = Annotated[
    str,
    Field(
        ...,
        description="Explain why you chose this code. The explanation should mention the code's description and what text in the claim matched with that description.",
    ),
]


class Patient(BaseModel):
    medical_record_number: str = Field(
        ...,
        description="Any identifier you care to send that uniquely identifies a particular patient",
    )
    birth: date_type = Field(..., description="The patient's birthdate")
    gender: Literal["M", "F"]


class ModifierItem(BaseModel):
    modifier_code: str = Field(..., description="A 2-character modifier code")
    modifier_code_reasoning: reasoning


class Modifier(BaseModel):
    modifier: ModifierItem


class DiagnosisItem(BaseModel):
    diagnosis_code: str = Field(..., description="An ICD code")
    diagnosis_code_reasoning: reasoning


class Diagnosis(BaseModel):
    diagnosis: DiagnosisItem


class ProcedureItem(BaseModel):
    procedure_code: str = Field(..., description="A CPT code")
    procedure_code_reasoning: reasoning
    from_date: date_type = Field(
        ...,
        description="The starting date on which the procedure was performed. This field can be the same as the date-of-service",
    )
    to_date: date_type = Field(
        ...,
        description="The ending date on which the procedure was performed. This field can be the same as the date-of-service",
    )
    modifier_list: list[Modifier] = Field(
        default_factory=list,
        description="List of modifiers. May contain 0 to 4 modifiers",
        min_length=0,
        max_length=4,
    )
    diagnosis_list: list[Diagnosis]
    place_of_service: str | None = Field(
        None,
        description="A 2-character place of service overriding the default-place-of-service",
    )
    units: int = Field(..., description="Number of procedure units")
    unitstime: int | None = Field(
        None, description="Number of minutes spent doing the procedure (optional)"
    )


class Procedure(BaseModel):
    procedure: ProcedureItem


class Claim(BaseModel):
    carrier_code: str = Field(
        ..., description="The standard code identifying the payer (eg MC051 is Medi-Cal)"
    )
    state: str = Field(
        ...,
        description="The standard 2-character state abbreviation code (eg. VT for Vermont)",
    )
    practice: str = Field(
        ...,
        description="Any identifier you care to send that uniquely identifies the practice",
    )
    provider: str = Field(..., description="The provider's NPI number")
    default_place_of_service: str = Field(
        ..., description="A 2-character place of service code"
    )
    date_of_service: date_type
    patient: Patient
    procedure_list: list[Procedure] = Field(
        ...,
        description="List of procedures. May contain one or more procedure elements",
        min_length=1,
    )


class ChartKeywords(BaseModel):
    keywords_phrases: list[str]


def create_med_db(
    lance_db: str | LanceDBConnection,
    med_dir: str | Path,
    ems_model="hkunlp/instructor-base",
    ems_model_device: str = "cuda",
    overwrite: bool = False,
):
    logger.info(f"Creating medical database from directory: {med_dir}")
    if isinstance(lance_db, str):
        lance_db = lancedb.connect(lance_db)
        logger.info(f"Connected to LanceDB at: {lance_db}")
    for file in Path(med_dir).glob("[icd|cpt]*.json"):
        table_name = file.stem
        logger.info(f"Processing file: {file.name}")
        if table_name in lance_db.table_names():
            if overwrite:
                logger.warning(f"Overwriting existing table: {table_name}")
                lance_db.drop_table(table_name)
            else:
                logger.warning(
                    f"Skipping existing table: {table_name} as it already exists and overwrite is False"
                )
                continue
        logger.info(f"Reading data from file: {file}")
        data = [
            MarkdownChunk(
                name=code["code"],
                index=i,
                text=f"code: {code['code']}\ndescriprion: {code['desc']}",
            )
            for i, code in enumerate(json.loads(file.read_text()))
        ]
        logger.info(f"Adding {len(data)} entries to table: {table_name}")
        add_to_lance_table(
            db=lance_db,
            table_name=table_name,
            data=data,
            ems_model=ems_model,
            ems_model_device=ems_model_device,
        )
    logger.success("Medical database creation completed")


def extract_keywords(charts_dir: str | Path, overwrite: bool = False):
    for chart in Path(charts_dir).glob("*.md"):
        output_file = chart.with_name(chart.stem + "_kw.json")
        if not overwrite and output_file.exists():
            logger.warning(
                f"Skipping {output_file} as it already exists and overwrite is False"
            )
            continue
        logger.info(f"Extracting keywords for: {chart.stem}")
        chart_dialog = Dialog(
            task="kw_task.txt", template="<patient_chart>\n{chart}\n</patient_chart>"
        )
        creator, kwargs = chart_dialog.creator_with_kwargs(
            model=MODEL, template_data={"chart": chart.read_text()}
        )
        kw = creator.create(response_model=ChartKeywords, **kwargs).keywords_phrases  # type: ignore
        output_file.write_text(json.dumps(kw))
        logger.success(f"Extracted keywords for: {chart.stem}")


def query_med_db(
    lance_db: LanceDBConnection,
    charts_dir: str | Path,
    reranker: str | Reranker | None = "answerdotai/answerai-colbert-small-v1",
    max_search_results: int = MAX_SEARCH_RESULTS,
    overwrite: bool = False,
):
    for file in Path(charts_dir).glob("*_kw.json"):
        output_file = file.with_stem(file.stem + "_retrieved")
        if not overwrite and output_file.exists():
            logger.warning(
                f"Skipping {output_file} as it already exists and overwrite is False"
            )
            continue
        if isinstance(reranker, str):
            reranker = ColbertReranker(model_name=reranker)
        kws = json.loads(file.read_text())
        claim_content = re.sub(
            r"\n+", "\n", file.with_name(file.name.replace("_kw.json", ".md")).read_text()
        )
        # claim_content = [
        #     (
        #         claim_content.replace("**", "")
        #         .replace(": ", "- ")
        #         .replace("-", " ")
        #         .strip(":")
        #         .strip("-")
        #         .strip()
        #     )
        # ]
        claim_content = [
            s.replace("**", "")
            .replace(": ", "- ")
            .replace("-", " ")
            .strip(":")
            .strip("-")
            .strip()
            for s in claim_content.splitlines()
        ]
        kws.extend(claim_content)
        res = {}
        for table_name in lance_db.table_names():
            logger.info(f"Querying table: {table_name} for: {file.stem}")
            res[table_name] = search_lancedb(
                db=lance_db,
                table_name=table_name,
                query=kws,
                reranker=reranker,
                max_search_results=max_search_results,
            )
        output_file.write_text(json.dumps(res))
        logger.success(f"Retrieved codes for: {file.stem}")


def extract_claims(
    charts_dir: str | Path, modifiers_file: str | Path, overwrite: bool = False
):
    for chart in Path(charts_dir).glob("*.md"):
        output_file = chart.with_name(chart.stem + "_claim.json")
        if not overwrite and output_file.exists():
            logger.warning(
                f"Skipping {output_file} as it already exists and overwrite is False"
            )
            continue
        kw_retrieved_file = chart.with_name(chart.stem + "_kw_retrieved.json")
        if not kw_retrieved_file.exists():
            logger.warning(f"Skipping {chart.stem} as {kw_retrieved_file} does not exist")
            continue
        codes = json.loads(kw_retrieved_file.read_text())
        claim_dialog = Dialog(
            task="extraction_task.txt",
            template="<patient_chart>\n{chart}\n</patient_chart>\n\n<available_codes>\n{codes}\n</available_codes>",
        )
        codes["modifiers"] = json.loads(Path(modifiers_file).read_text())
        creator, kwargs = claim_dialog.creator_with_kwargs(
            model=ModelName.GEMINI_FLASH,
            template_data={"chart": chart.read_text(), "codes": json.dumps(codes)},
        )
        claim = creator.create(response_model=Claim, **kwargs)
        output_file.write_text(json.dumps(claim.model_dump(), indent=2))  # type: ignore
        logger.success(f"Extracted claim for: {chart.stem}")


if __name__ == "__main__":
    charts_dir = "charts"
    med_dir = Path("/media/hamza/data2/med_codes")
    modifiers_file = med_dir / "modifier2024.json"
    lance_db = lancedb.connect("med_db")
    create_med_db(lance_db=lance_db, med_dir=med_dir)
    extract_keywords(charts_dir=charts_dir)
    query_med_db(lance_db=lance_db, charts_dir=charts_dir)
    extract_claims(charts_dir=charts_dir, modifiers_file=modifiers_file)
