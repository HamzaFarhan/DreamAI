from pathlib import Path

from loguru import logger

from dreamai.ai import ModelName
from dreamai.dialog import Dialog, assistant_message, user_message

MODEL = ModelName.GPT_MINI
DIALOG_NAME = "dialog_with_rules"

rules_prompt = """
Based on our dialog history so far, generate a set of rules and guidelines or steps that you will use moving forward to effectively solve such type of tasks according to my specific preferences. Make sure to keep in mind our whole conversation. Don't make generic rules. Very specific and focused.
The format should be as follows:
<Rules and Guidelines>
Text of Rules and Guidelines
</Rules and Guidelines>
"""


def get_user_query() -> str:
    query = ""
    while not query:
        query = input("('q' to exit) > ").strip()
    return query


def ask_ai(dialog: Dialog, user: str, model: ModelName = MODEL) -> str:
    creator, kwargs = dialog.creator_with_kwargs(model=model, user=user)
    res = creator.create(response_model=str, **kwargs)
    dialog.add_messages(
        messages=[user_message(content=user), assistant_message(content=res)]
    )
    return str(res)


dialog_path = Path(DIALOG_NAME).with_suffix(".json")
dialog = (
    Dialog.load(path=dialog_path)
    if dialog_path.exists()
    else Dialog(task="Give your answers thinking in a step by step manner.")
)
if dialog.chat_history:
    print("CHAT HISTORY\n----------------\n")
    for message in dialog.chat_history:
        print(f"{message['role'].upper()} => {message['content']}\n")

user = get_user_query()
while user.lower() != "q":
    res = ask_ai(dialog=dialog, user=user)
    logger.success(res)
    user = get_user_query()
ask_ai(dialog=dialog, user=rules_prompt)
dialog.bump_version(name=DIALOG_NAME, save=True)
