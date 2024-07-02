# Import necessary libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import SGD
import matplotlib.pyplot as plt

# simple statistics with numpy
my_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print('Mean:', np.mean(my_array))
print('Median:', np.median(my_array))
print('Standard deviation:', np.std(my_array))

# complex mathematical functions with numpy
print('Exponential:', np.exp(my_array))

# random number generation with numpy
print('Random number:', np.random.rand())

# multi-dimensional array indexing
multi_dim_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('Element at row 2, column 3:', multi_dim_array[1, 2])

# Linear algebra operations
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])
print('Matrix Multiplication:', np.dot(A, B))

# Array conversion
list_data = [13, 14, 15]
print('Array from list:', np.array(list_data))

# creating pandas dataframe
df = pd.DataFrame(my_array, columns=['Column1'])
print(df.describe())

# manipulating dataframe
df['Column2'] = df['Column1'] * 2
print(df.head())

# data for model training
X = np.random.random((1000, 20))
y = np.random.randint(2, size=(1000, 1))
y = np_utils.to_categorical(y)

# model creation
model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dense(2, activation='softmax'))

# model compilation
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# training model
model.fit(X, y, epochs=10, batch_size=32)

# evaluate model
loss, accuracy = model.evaluate(X, y, verbose=0)
print("Loss : %.2f" % loss)
print("Accuracy : %.2f" % accuracy)

# prediction
predictions = model.predict(X[:5])
for i, pred in enumerate(predictions):
    print(f'Data {i+1}: Class{np.argmax(pred)}')

# complex mathematical function representation
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title('Sine wave function using Numpy and Matplotlib')
plt.show()