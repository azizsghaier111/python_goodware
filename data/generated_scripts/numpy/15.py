import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import SGD
import pandas as pd
import matplotlib.pyplot as plt


# Function to compute basic statistics with numpy
def basic_statistics():
    array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(f'Mean: {np.mean(array)}')
    print(f'Median: {np.median(array)}')
    print(f'Variance: {np.var(array)}')
    print(f'Exponential: {np.exp(array)}')
    print(f'Random number: {np.random.rand()}')


# Function to create pandas dataframe and describe it
def pandas_example(array):
    df = pd.DataFrame(array, columns=['Column1'])
    print(df.describe())


# Function to create, compile, and train a model with keras
def keras_model():
    X = np.random.random((1000, 20))
    Y = np_utils.to_categorical(np.random.randint(2, size=(1000, 1)))
    model = Sequential()
    model.add(Dense(64, input_dim=20, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model.fit(X, Y, epochs=10, batch_size=32)
    print(f"Model Accuracy: {model.evaluate(X, Y, verbose=0)[1]*100}%")


# Function to create matplotlib plot
def matplotlib_plot():
    x_values = np.linspace(0, 2*np.pi, 100)
    y_values = np.sin(x_values)
    plt.plot(x_values, y_values)
    plt.title('Sine wave function using Numpy and Matplotlib')
    plt.show()


def main():
    array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    basic_statistics()
    pandas_example(array)
    keras_model()
    matplotlib_plot()


if __name__ == "__main__":
    main()