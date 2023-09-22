from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.text import Tokenizer
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint

from labelling import X_train, X_test, y_train, y_test
from constant import BEST_MODEL_PATH
from os import getenv, path


LAYERS_DENSE = int(getenv("LAYERS_DENSE"))
MAX_VOCAB = int(getenv("MAX_VOCAB"))
MAX_LEN = int(getenv("MAX_LEN"))
EPOCHS = int(getenv("EPOCHS"))

model = Sequential()
model.add(layers.Embedding(MAX_VOCAB, 40, input_length=MAX_LEN))
model.add(layers.Bidirectional(layers.LSTM(20,dropout=0.6)))
model.add(layers.Dense(LAYERS_DENSE,activation='softmax'))

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    filepath=BEST_MODEL_PATH, 
    monitor='val_accuracy', 
    verbose=1,
    save_best_only=True, 
    mode='auto', 
    save_freq='epoch',
    save_weights_only=False
)

history = model.fit(
    X_train, y_train, 
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint]
)

