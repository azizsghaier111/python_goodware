# Import necessary libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import SGD
import matplotlib.pyplot as plt

# Array creation with numpy
my_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print('My array:', my_array)

# Array indexing with numpy
first_element = my_array[0]  # Fetching the first index
print('First element:', first_element)

# Array conversion with numpy
list_conversion = my_array.tolist()  # Converting numpy array to list
print('List conversion:', list_conversion)

# Checking simple statistics with numpy
print('Mean:', np.mean(my_array))
print('Median:', np.median(my_array))
print('Standard deviation:', np.std(my_array))
print('Exponential:', np.exp(my_array))  # Checking complex mathematical functions with numpy

# Random number generation with numpy
print('Random number:', np.random.rand())

# Creating pandas dataframe from the array
df = pd.DataFrame(my_array, columns=['Column1'])
print(df.describe())

# Creating random data for model training
X = np.random.random((1000, 20))
y = np.random.randint(2, size=(1000, 1))
y = np_utils.to_categorical(y)

# Building the model
model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compiling the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# Training the model
history = model.fit(X, y, epochs=10, batch_size=32)

# Evaluating the model
loss, accuracy = model.evaluate(X, y, verbose=0)
print("Loss: %.2f" % loss)
print("Accuracy: %.2f" % accuracy)

# Making some predictions
predictions = model.predict(X[:5])
for i, pred in enumerate(predictions):
    print(f'Data {i+1}: Class {np.argmax(pred)}')

# Plotting complex mathematical function with matplotlib
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title('Sine wave function with Numpy and Matplotlib')
plt.show()