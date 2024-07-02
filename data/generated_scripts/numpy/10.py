# Import necessary libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import SGD
import matplotlib.pyplot as plt

# Define an 1D numpy array
my_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Carry out the Discrete Fourier Transform of the array
fourier_transform = np.fft.fft(my_array)
print('Fourier Transform:', fourier_transform)

# Sort the array
sorted_array = np.sort(my_array)
print('Sorted array:', sorted_array)

# Perform Array Math operations: Addition, Subtraction, Multiplication, Division
print('Addition:', np.add(my_array, 5))  # Addiction
print('Subtraction:', np.subtract(my_array, 2))  # Subtraction
print('Multiplication:', np.multiply(my_array, 3))  # Multiplication
print('Division:', np.divide(my_array, 4))  # Division

# Save the numpy array to a file
np.save('my_array.npy', my_array)

# Load the numpy array from a file
loaded_array = np.load('my_array.npy')
print('Loaded array:', loaded_array)

# Print out some statistics about the numpy array
print('Mean:', np.mean(loaded_array))  # Calculate Mean
print('Median:', np.median(loaded_array))  # Calculate Median
print('Standard Deviation:', np.std(loaded_array))  # Calculate Standard Deviation

# Apply some complex math function with numpy and print the results
print('Exponential:', np.exp(loaded_array))  # Exponential function

# Generate a random number using numpy and print it
print('Random number:', np.random.rand())  # Random Number generation

# Convert the 1D numpy array to a pandas dataframe
df = pd.DataFrame(loaded_array, columns=['Column1'])  # Convert it to Pandas Dataframe

# Print out some statistics about the pandas dataframe
print(df.describe())  # Describe the dataframe

# Generate some random dataset
X = np.random.random((1000, 20))  # Training Data
y = np.random.randint(2, size=(1000, 1))  # Target Label
y = np_utils.to_categorical(y)  # Convert it to categorical

# Define a Keras Model
model = Sequential()  # Sequential Model
model.add(Dense(64, input_dim=20, activation='relu'))  # Input Layer
model.add(Dense(2, activation='softmax'))  # Output Layer

# Compile the Model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=sgd)

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Calculate and print the loss and the accuracy of the model
loss, accuracy = model.evaluate(X, y, verbose=0)
print("\nLoss : %.2f" % loss)
print("Accuracy : %.2f" % accuracy)

# Use the model to predict the first 5 samples and print them
predictions = model.predict(X[:5])
for i, pred in enumerate(predictions):
    print(f'Data {i+1}: Class{np.argmax(pred)}')

# Plot the sine function
x = np.linspace(0, 2*np.pi, 100)  # x axis data
y = np.sin(x)  # y axis data
plt.plot(x, y)  # Plot the function
plt.title('Sine wave function using Numpy and Matplotlib') # Add a title to the plot
plt.show()  # Show the plot .