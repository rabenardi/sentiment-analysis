from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer

from sources import samples
from os import getenv

# Mengambil harga MAX_VOCAB dan MAX_LEN dari .env
MAX_VOCAB = int(getenv("MAX_VOCAB"))
MAX_LEN = int(getenv("MAX_LEN"))

# Mengambil daftar sentimen (label) dan sumber data
raw_labels = samples["source"].dataframes[0]
raw_texts = samples["source"].dataframes[1]

# Mengubah nilai sentimen menjadi matrix biner
labels = to_categorical(raw_labels, 3, dtype="float32")

# Mengubah kalimat menjadi numpy array
tokenizer = Tokenizer(num_words=MAX_VOCAB)
tokenizer.fit_on_texts(raw_texts)
sequences = tokenizer.texts_to_sequences(raw_texts)
texts = pad_sequences(sequences, maxlen=MAX_LEN)
print(labels ,texts)
# Menyiapkan sampel pelatihan dan validasi
X_train, X_test, y_train, y_test = train_test_split(texts, labels, random_state=0)

print("\nProses labelling berhasil!")
