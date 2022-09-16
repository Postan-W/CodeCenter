from pathlib import Path
FILE = Path(__file__).resolve()
print(FILE.parents[0] / "weight")