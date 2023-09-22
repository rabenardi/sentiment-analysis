from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from numpy import around as np_around

from sources import samples
from constant import BEST_MODEL_PATH
from constant import TOKENIZER as tokenizer
from os import getenv


best_model = load_model(BEST_MODEL_PATH)
sentiment = samples["source"].sentiment_values

MAX_VOCAB = int(getenv("MAX_VOCAB"))
MAX_LEN = int(getenv("MAX_LEN"))

def predict_sentiment(sentence: str):
    sequence = tokenizer.texts_to_sequences([sentence])
    test = pad_sequences(sequence, maxlen=MAX_LEN)
    return sentiment[np_around(best_model.predict(test), decimals=0).argmax(axis=1)[0]]