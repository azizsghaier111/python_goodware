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
print('Original array:\n', my_array)

# Logical operation on numpy array
even_bool = my_array % 2 == 0
print('Boolean array for even numbers:\n', even_bool)

# Creating boolean array for numbers greater than 5
greater_than_five_bool = my_array > 5
print('Boolean array for numbers greater than 5:\n', greater_than_five_bool)

# Creating array using boolean indexing
selected_elements = my_array[greater_than_five_bool & even_bool]
print('Selected elements:\n', selected_elements)

# Array shape manipulation with numpy
reshaped_array = my_array.reshape(3, 3)
print('Reshaped array:\n', reshaped_array)

# Basic statistics with numpy
print('Mean:', np.mean(my_array))
print('Median:', np.median(my_array))
print('Standard deviation:', np.std(my_array))

# complex mathematical functions with numpy
print('Exponential:', np.exp(my_array))

# Random number generation with numpy
print('Random number:', np.random.rand())

# Creating pandas dataframe
df = pd.DataFrame(my_array, columns=['Column1'])
print(df.describe())

# Data for model training
X = np.random.random((1000, 20))
y = np.random.randint(2, size=(1000, 1))
y = np_utils.to_categorical(y)

# Model creation
model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Model compilation
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Training model
model.fit(X, y, epochs=10, batch_size=32)

# Evaluate model
loss, accuracy = model.evaluate(X, y, verbose=0)
print("Loss : %.2f" % loss)
print("Accuracy : %.2f" % accuracy)

# Prediction
predictions = model.predict(X[:5])
for i, pred in enumerate(predictions):
    print(f'Data {i+1}: Class{np.argmax(pred)}')
    
# Complex mathematical function representation
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title('Sine wave function using Numpy and Matplotlib')
plt.show()