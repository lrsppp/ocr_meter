import os
from pathlib import Path

ROOT_DIR = Path(__file__).parents[0]

DATA_PATH = ROOT_DIR / Path("data")

CONFIG = {
    "custom_config": r"--oem 1 --psm 13 -c tessedit_char_whitelist=0123456789,"  # or --psm 7
}

API_PORT = 8000
if os.environ.get("DOCKER_CONTAINER"):
    # Use host.docker.internal when running in Docker
    API_URL = f"http://host.docker.internal:{API_PORT}/upload-image"
else:
    # Use localhost when running locally
    API_URL = f"http://localhost:{API_PORT}/upload-image"
