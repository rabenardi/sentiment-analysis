from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from sources import samples
from constant import TOKENIZER as tokenizer
from os import getenv


MAX_VOCAB = int(getenv("MAX_VOCAB"))
MAX_LEN = int(getenv("MAX_LEN"))

raw_labels = samples["source"].dataframes[0]
raw_texts = samples["source"].dataframes[1]
processed_raw_texts = raw_texts.values.tolist()

labels = to_categorical(raw_labels, 3, dtype="float32")

tokenizer.fit_on_texts(processed_raw_texts)
sequences = tokenizer.texts_to_sequences(processed_raw_texts)
texts = pad_sequences(sequences, maxlen=MAX_LEN)


X_train, X_test, y_train, y_test = train_test_split(texts, labels, random_state=0)

print("\nProses labelling berhasil!")
