from pathlib import Path

import lancedb
import polars as pl
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from PIL import Image

LANCE_URI = "pallets_db"

db = lancedb.connect(LANCE_URI)
func = get_registry().get("open-clip").create(name="ViT-B-32", device="cuda")


class Images(LanceModel):
    brand: str
    image_uri: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore


images_dir = Path("images")
table_name = "brands"
if table_name not in db.table_names():
    table = db.create_table(name=table_name, schema=Images, exist_ok=True)
    table.add(
        pl.DataFrame(
            [
                {"brand": image_path.stem.split("_")[0], "image_uri": str(image_path)}
                for image_path in images_dir.glob("*png")
            ]
        )
    )
table = db.open_table(name=table_name)
query_image = Image.open("coors.png")
brand = "coors"
res = (
    table.search(query=query_image)
    # .where(f"brand != '{brand}'", prefilter=True)
    # .limit(1)
    .to_polars()
)
print(res)
