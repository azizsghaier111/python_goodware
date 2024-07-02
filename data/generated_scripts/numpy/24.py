# Import Necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from numpy.fft import fft, ifft

try:
    # Generate Random number
    print("\nGenerating Random Numbers\n")
    random_number = np.random.rand()
    print("Random Number: ", random_number)
    random_array = np.random.rand(5, 5)
    print("Random Array: \n", random_array)

    # Perform array conversion
    print("\nPerforming list to array conversion\n")
    listA = [1, 2, 3, 4]
    listB = [5, 6, 7, 8]
    arrayA = np.array(listA)
    arrayB = np.array(listB)
    print("Array A: \n", arrayA)
    print("Array B: \n", arrayB)

    # Perform basic mathematical operations using numpy
    print("\nPerforming Basic Mathematical Operations\n")
    print("Addition: ", np.add(arrayA, arrayB))
    print("Subtraction: ", np.subtract(arrayA, arrayB))
    print("Multiplication: ", np.multiply(arrayA, arrayB))
    print("Division: ", np.divide(arrayA, arrayB))
    print("Exponentiation: ", np.exp(arrayA))
    print("Square root: ", np.sqrt(arrayB))
    print("Logarithm: ", np.log(arrayB))

    # Perform Fourier Transform
    print("\nPerforming Fourier Transform\n")
    fourier_A = fft(arrayA)
    fourier_B = fft(arrayB)
    print("Fourier Transform of A: ", fourier_A)
    print("Fourier Transform of B: ", fourier_B)

    # Linear Algebra Operations
    print("\nPerforming Linear Algebra Operations\n")
    print("Matrix multiplication: \n", np.matmul(arrayA, arrayB))
    print("Matrix transpose: \n", np.transpose(arrayA)) 

    # Remaining of your code goes here...
  
except Exception as e:
    print(str(e))