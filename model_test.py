from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import numpy as np

from sources import samples
from constant import BEST_MODEL_PATH
from os import getenv

# memuat model yang telah dilatih
best_model = load_model(BEST_MODEL_PATH)

# mengambil daftar nilai sentimen untuk indexing
sentiment = samples["source"].sentiment_values

MAX_VOCAB = int(getenv("MAX_VOCAB"))
MAX_LEN = int(getenv("MAX_LEN"))

def predict_sentiment(sentence: str):
    tokenizer = Tokenizer(num_words=MAX_VOCAB)
    tokenizer.fit_on_texts(samples["source"].dataframes[1])
    sequence = tokenizer.texts_to_sequences([sentence])
    test = pad_sequences(sequence, maxlen=MAX_LEN)
    return sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]]
