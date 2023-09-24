from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.text import Tokenizer
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping

from labelling import X_train, X_test, y_train, y_test
from constant import BEST_MODEL_PATH
from os import getenv, path

# Mengambil empat nilai berikut dari .env
LAYERS_DENSE = int(getenv("LAYERS_DENSE"))
BATCH_SIZE = int(getenv("BATCH_SIZE"))
MAX_VOCAB = int(getenv("MAX_VOCAB"))
MAX_LEN = int(getenv("MAX_LEN"))
EPOCHS = int(getenv("EPOCHS"))

# Menyiapkan model
model = Sequential()

# Word embedding
model.add(layers.Embedding(MAX_VOCAB, 40, input_length=MAX_LEN))

model.add(layers.Bidirectional(layers.LSTM(20,dropout=0.6)))
model.add(layers.Dense(LAYERS_DENSE,activation='softmax'))

# melatih model
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# menyimpan model jika terjadi peningkatan akurasi
checkpoint = ModelCheckpoint(
    filepath=BEST_MODEL_PATH, 
    monitor='val_accuracy', 
    verbose=1,
    save_best_only=True, 
    mode='auto', 
    save_freq='epoch',
    save_weights_only=False
)

# mengukur seberapa akurat model yang sedang dilatih
history = model.fit(
    X_train, y_train, 
    epochs=EPOCHS, # perulangan
    validation_data=(X_test, y_test),
    callbacks=[checkpoint],
    batch_size=BATCH_SIZE
)

# Mengatur pemberhenti otomatis (jika tidak ada peningkatan pada akurasi)
es = EarlyStopping(
    monitor="val_accuracy",
    verbose=1,
    patience=60
)
