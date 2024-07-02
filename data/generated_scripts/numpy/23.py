import numpy as np
import numpy.ma as ma
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import SGD
from scipy import fftpack
import matplotlib.pyplot as plt
import os

# Basic statistics with numpy
my_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print('Mean:', np.mean(my_array))
print('Median:', np.median(my_array))
print('Standard deviation:', np.std(my_array))

# Complex mathematical functions with numpy
print('Exponential:', np.exp(my_array))
print('Trigonometric sine:', np.sin(my_array))
print('Logarithm:', np.log(my_array))

# Binary operations
print('Bitwise AND operation:', np.bitwise_and(my_array, np.array([1]*len(my_array))))
print('Bitwise OR operation:', np.bitwise_or(my_array, np.array([1]*len(my_array))))

# Random number generation with numpy
print('Random number:', np.random.rand())
print('Random number (3x3 array):', np.random.rand(3, 3))

# Multi-dimensional array operations
multi_dim_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('Element at row 2, column 3:', multi_dim_array[1, 2])
print('First 2 elements of row 3:', multi_dim_array[2, :2])

# Fourier Transform
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
yf = fftpack.fft(y)
print('Fourier transform:', yf)

# Masked array operations
mx = ma.array(my_array, mask=my_array<5)
print('Masked array:', mx)

# Linear algebra operations
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])
print('Matrix Multiplication:', np.dot(A, B))
print('Matrix Transpose:', np.transpose(A))

# Saving and loading arrays
np.save('my_array.npy', my_array)
loaded_array = np.load('my_array.npy')
print('Loaded array:', loaded_array)

# Array conversion
list_data = [13, 14, 15]
print('Array from list:', np.array(list_data))

# Pandas dataframe manipulation
df = pd.DataFrame(my_array, columns=['Column1'])
print(df.describe())
df['Column2'] = df['Column1'] * 2

df = df.append(pd.DataFrame(np.array(list_data).reshape(-1,1), columns=['Column1']))
df.reset_index(drop=True, inplace=True)
print(df.head())

# Binary data for model training
X = np.random.random((1000, 20))
y = np.random.randint(2, size=(1000, 1))
y = np_utils.to_categorical(y)

# Model creation
model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Model compilation
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# Model training
model.fit(X, y, epochs=10, batch_size=32)

# Model evaluation
loss, accuracy = model.evaluate(X, y, verbose=0)
print("Loss : %.2f" % loss)
print("Accuracy : %.2f" % accuracy)

# Prediction
predictions = model.predict(X[:5])
for i, pred in enumerate(predictions):
    print(f'Data {i+1}: Class {np.argmax(pred)}')

# Graph plot of a complex mathematical function 
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title('Sine wave function using Numpy and Matplotlib')
plt.show()

# Finally remove saved array
os.remove('my_array.npy')