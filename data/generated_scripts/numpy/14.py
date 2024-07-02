# Importing Necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from numpy.fft import fft, ifft
from keras.utils import np_utils

# Conversion of lists to arrays
listA = [1, 2, 3, 4]
listB = [5, 6, 7, 8]

# Using numpy for array conversion
arrayA = np.array(listA)
arrayB = np.array(listB)

print("Array A: ", arrayA)
print("Array B: ", arrayB)

# Performing basic mathematical operations using numpy
print("Addition: ", np.add(arrayA, arrayB))
print("Subtraction: ", np.subtract(arrayA, arrayB))
print("Multiplication: ", np.multiply(arrayA, arrayB))
print("Division: ", np.divide(arrayA, arrayB))

# Fourier Transform
fourier_A = fft(arrayA)
fourier_B = fft(arrayB)
print("Fourier Transform of A: ", fourier_A)
print("Fourier Transform of B: ", fourier_B)

# Calculating the mean using numpy
print("Mean Array A: ", np.mean(arrayA))
print("Mean Array B: ", np.mean(arrayB))

# Standard deviation in numpy
print("Standard Deviation Array A: ", np.std(arrayA))
print("Standard Deviation Array B: ", np.std(arrayB))

# Variance in numpy
print("Variance Array A: ", np.var(arrayA))
print("Variance Array B: ", np.var(arrayB))

# Inverse Fourier Transform
inverse_A = ifft(fourier_A)
inverse_B = ifft(fourier_B)
print("Inverse Fourier Transform of A: ", inverse_A)
print("Inverse Fourier Transform of B: ", inverse_B)

# Complex Mathematical function (wave function)
x = np.linspace(0, 2*np.pi, 1000)
y = np.sin(2*np.pi*x)

plt.plot(x, y)
plt.title('Wave Function using Numpy and Matplotlib')
plt.show()

# Creating a pandas DataFrame using numpy array
df = pd.DataFrame(arrayA, columns=['Column_A'])
print(df)

# Saving array to .npy file
np.save('arrayA.npy', arrayA)
np.save('arrayB.npy', arrayB)

# Loading array from .npy file
loaded_arrayA = np.load('arrayA.npy')
loaded_arrayB = np.load('arrayB.npy')

# Creating a basic model using Keras
model = Sequential() # instanciate the model
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Using numpy to generate dummy data for our model
x_train = np.random.random((1000, 100))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)

model.fit(x_train, y_train, epochs=20, batch_size=128)
_, accuracy = model.evaluate(x_train, y_train, verbose=0)

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

# Load the model
from keras.models import load_model
model = load_model('my_model.h5')
predictions = model.predict(x_train[:5])

print('Accuracy: %.2f' % (accuracy*100))

print("Predictions:")
for i, prediction in enumerate(predictions):
    print(f"Data {i}: Class {np.argmax(prediction)}")