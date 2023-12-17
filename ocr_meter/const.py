from pathlib import Path

ROOT_DIR = Path(__file__).parents[0]

DATA_PATH = ROOT_DIR / Path("data")

CONFIG = {
    "custom_config": r"--oem 1 --psm 13 -c tessedit_char_whitelist=0123456789,"  # or --psm 7
}

API_URL = "http://127.0.0.1:8002/upload-image"
