from keras.preprocessing.text import Tokenizer

from os import path, getenv
from dotenv import load_dotenv
load_dotenv()

MAX_VOCAB = int(getenv("MAX_VOCAB"))

BEST_MODEL_PATH = path.normpath(
    getenv("BEST_MODEL_PATH") or
    path.join(path.dirname(__file__), "best.model")
)

TOKENIZER = Tokenizer(num_words=MAX_VOCAB)