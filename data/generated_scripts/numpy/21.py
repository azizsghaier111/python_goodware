# python standard library modules import
import os

# 3rd party packages import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import SGD

# data directories definition
DIR_DATA = ''

def main():

    # load data
    raw = load_waveforms()

    # preprocess data
    waveforms = preprocess_waveforms(raw)

    # build model
    model = create_model()

    # get labels and features
    label = ...
    feature = ...

    # train model
    model.fit(x=feature, y=label, batch_size=32, epochs=200, verbose=1)

    # evaluate model
    evaluate_model(model, feature, label)

    return

def load_waveforms():

    # load waveform data
    waveforms = np.load(os.path.join(DIR_DATA, 'waveform.npy'))

    return waveforms

def preprocess_waveforms(waveforms):

    # implement some preprocessing

    return waveforms

def create_model():

    # build model
    model = Sequential()
    model.add(Dense(64, input_dim=20, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # compile model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model

def evaluate_model(model, feature, label):

    # evaluate model
    loss, acc = model.evaluate(x=feature, y=label, verbose=1)

    print(f'Loss={loss:.6f}, Acc={acc:.6f}')

    return

if __name__ == '__main__':
    main()