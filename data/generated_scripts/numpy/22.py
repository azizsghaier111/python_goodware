import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import SGD
import matplotlib.pyplot as plt

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

# Difference of elements
print('Difference of elements:', np.diff(my_array))

# Boolean Indexing
print('Boolean indexing:', my_array[my_array > 5])

# Set operations
print('Unique elements:', np.unique(my_array))

# Operations on complex numbers
complex_nums = np.array([1+2j, 2+3j, 3+4j])
print('Real part:', np.real(complex_nums))
print('Imaginary part:', np.imag(complex_nums))
print('Complex conjugate:', np.conj(complex_nums))

# Sorting of array
print('Sorted array:', np.sort(my_array))

# Array reshaping
print('Reshaped to 3x3 array:\n', my_array.reshape((3,3)))

# Additional operations
print("Sum:", np.sum(my_array))
print("Min:", np.min(my_array))
print("Max:", np.max(my_array))
print("Argmin:", np.argmin(my_array))
print("Argmax:", np.argmax(my_array))
print("CumSum:", np.cumsum(my_array))
print("Cum Product:", np.cumprod(my_array))

# Array concatenation
print("Concatenated array:", np.concatenate((my_array, my_array)))

# Splitting Array
print('Split array into 3:', np.array_split(my_array, 3))

# More matrix operations
print('Matrix square root:\n', np.sqrt(multi_dim_array))
print('Matrix sum:\n', np.sum(multi_dim_array))
print('Matrix trace:\n', np.trace(multi_dim_array))
print('Matrix power:\n', np.linalg.matrix_power(multi_dim_array, 2))

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