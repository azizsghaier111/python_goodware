# Importing Necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from numpy.fft import fft, ifft

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

# Inverse Fourier Transform
inverse_A = ifft(fourier_A)
inverse_B = ifft(fourier_B)
print("Inverse Fourier Transform of A: ", inverse_A)
print("Inverse Fourier Transform of B: ", inverse_B)

# Set operations
print("Union: ", np.union1d(arrayA, arrayB))
print("Intersection: ", np.intersect1d(arrayA, arrayB))
print("Difference: ", np.setdiff1d(arrayA, arrayB))

# Complex Mathematical function (wave function)
x = np.linspace(0, 2*np.pi, 1000)
y = np.sin(2*np.pi*x)

plt.plot(x, y)
plt.title('Wave Function using Numpy and Matplotlib')
plt.show()

# Creating a pandas DataFrame using numpy array
df = pd.DataFrame(arrayA, columns=['Column_A'])
print(df)

# Creating a basic model using Keras
model = Sequential() # instanciate the model
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Using numpy to generate dummy data for our model
x_train = np.random.random((1000, 100))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)

model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluate accuracy
_, accuracy = model.evaluate(x_train, y_train, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))

# Predicting some values using our trained model
predictions = model.predict(x_train[:5])
print("Predictions:")
for i, prediction in enumerate(predictions):
    print(f"Data {i}: Class {np.argmax(prediction)}")