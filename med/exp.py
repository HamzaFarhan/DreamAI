import json
from pathlib import Path

files = [
    "/media/hamza/data2/med_codes/cpt2024.json",
    "/media/hamza/data2/med_codes/icd102024.json",
]
samples = [json.loads(Path(file).read_text())[:5] for file in files]
Path("samples.json").write_text(json.dumps(samples))
