import json
from pathlib import Path

import chardet
import pandas as pd
from loguru import logger


def csv_to_json(file: str, overwrite: bool = False):
    output_file = Path(file).with_suffix(".json")
    if output_file.exists() and not overwrite:
        logger.warning(f"Output file {output_file} already exists. Skipping.")
        return
    logger.info(encoding := chardet.detect(Path(file).read_bytes())["encoding"])
    df = pd.read_csv(file, encoding=encoding, sep="\t", names=["code"])
    output_file.write_text(
        json.dumps(
            [{"code": cd.split(",")[0], "desc": cd.split(",")[1]} for cd in df["code"]]
        )
    )


# files = ["/media/hamza/data2/med_codes/modifier2024.csv"]
# [csv_to_json(file=file) for file in files]
