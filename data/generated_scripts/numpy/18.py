import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import SGD

def main():
    
    # basic mathematical calculations
    my_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    another_array = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    print('Array sum:', np.add(my_array, another_array))
    print('Array difference:', np.subtract(another_array,my_array))
    print('Scalar Multiplication:', np.multiply(my_array, 3))
    print('Array Multiplication:', np.multiply(my_array, another_array))
    
    # statistical functions
    print('Mean:', np.mean(my_array))
    print('Median:', np.median(my_array))
    print('Variance:', np.var(my_array))
    
    # reshaping and manipulating array
    reshaped_array = my_array.reshape(3,3)
    print('Reshaped Array: \n', reshaped_array)
    print('Transposed Array: \n', np.transpose(reshaped_array))
    print('Flattened Array:', reshaped_array.flatten())
    
    # Discrete Fourier Transform
    x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    y = np.exp(2.0j * np.pi * x)
    print('Fourier Transformation:', fft(y))
    print('Inverse Fourier Transformation:', ifft(fft(y)))

    # random number generation
    print('Random number:', np.random.rand())
    print('Random integer array:', np.random.randint(low=1, high=10, size=10))
    
    # pandas
    df = pd.DataFrame(np.random.rand(10,5), columns=['A', 'B', 'C', 'D', 'E'])
    df2 = pd.DataFrame(np.random.rand(10,5), columns=['A', 'B', 'C', 'D', 'E'])
    merged_df = pd.concat([df, df2])
    print("Merged DataFrame: \n", merged_df)
    
    #keras
    X = np.random.random((1000, 20))
    y = np.random.randint(2, size=(1000, 1))
    y = np_utils.to_categorical(y)

    model = Sequential()
    model.add(Dense(64, input_dim=20, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # model compilation
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    # training model
    model.fit(X, y, epochs=10, batch_size=32)

    # evaluate model
    scores = model.evaluate(X, y, verbose=0)
    print("Model Accuracy: %.2f%%" % (scores[1]*100))

    # matplotlib
    x = np.linspace(0, 10, 500)
    y = np.sin(x)
    plt.plot(x, y)
    plt.title('Simple graph.')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()

if __name__ == "__main__":
    main()