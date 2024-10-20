import json
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from dreamai.ai import ModelName
from dreamai.dialog import Dialog


class ChartKeyWords(BaseModel):
    keywords_phrases: list[str]


chart_dialog = Dialog(
    task="kw_task.txt", template="<patient_chart>\n{chart}\n</patient_chart>"
)


for chart in Path("charts").iterdir():
    logger.info(f"Extracting for: {chart.stem}")
    creator, kwargs = chart_dialog.creator_with_kwargs(
        model=ModelName.GEMINI_FLASH, template_data={"chart": chart.read_text()}
    )
    kw = creator.create(response_model=ChartKeyWords, **kwargs).keywords_phrases  # type: ignore
    Path(chart.with_suffix(".json")).write_text(json.dumps(kw))
