from loguru import logger
from pydantic import BaseModel

from dreamai.ai import ModelName
from dreamai.dialog import BadExample, Dialog, Example


class CapitalCity(BaseModel):
    name: str
    country: str


ex1 = Example(user="Country: india", assistant="New Delhi")
ex2 = BadExample(
    user="Country: Pakistan",
    assistant="Berlin",
    feedback="Incorrect, it is Islamabad",
    correction="My apologies, the capital of Pakistan is Islamabad",
)
dialog = Dialog(
    task="Your job is to tell us the capital of any country",
    examples=[ex1, ex2],  # optional
    template="Country: {country}",  # optional
)
creator, kwargs = dialog.creator_with_kwargs(
    model=ModelName.GEMINI_FLASH,
    template_data={
        "country": "France"
    },  # or you could just give an str in "user" if template is not defined
)
res = creator.create(response_model=CapitalCity, **kwargs)
logger.success(f"Name: {res.name}, Country: {res.country}")  # type:ignore
