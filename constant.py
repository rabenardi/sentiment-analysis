from os import path, getenv
from dotenv import load_dotenv
load_dotenv()

BEST_MODEL_PATH = path.normpath(
    getenv("BEST_MODEL_PATH") or
    path.join(path.dirname(__file__), "best.model")
)
