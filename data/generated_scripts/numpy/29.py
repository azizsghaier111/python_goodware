import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

try:
    # Array Conversion  
    list1 = [1,2,3,4,5,6,7,8,9]   
    print("Initial List: ",list1)   
    arr = np.array(list1)   
    print("Converted Array: ",arr)     

    # Masked Array Operations
    arr = np.ma.array([1,2,3,4,5,6,7,8,9], mask=[0,1,0,0,1,0,1,0,0])  
    print("Masked Array: ",arr)       

    # Linear Algebra Operations
    arr1 = np.array([(1,2,3), (4,5,6)])   
    arr2 = np.array([(7,8,9), (10,11,12)])
    print("Array Dot Product: ",np.dot(arr1,arr2))

    # Array shape manipulation
    a = np.array([(1,2,3), (4,5,6)])  
    print("Shape: {}".format(a.shape))  #Prints (2,3)   
    b = a.reshape(3,2)  
    print("After reshaping: {}".format(b.shape))  # Prints (3,2)

    # Searching, sorting, counting
    a = np.array([12,9,13,17,12,20,18])  
    print("Index of max value: {}".format(np.argmax(a)))  # prints 5  
    print("Rank based on index: {}".format(np.argsort(a)))  
    print("Count of unique values: {}".format(np.unique(a, return_counts=True)))  

    # Array indexing
    a = np.array([(1,2,3,4), (5,6,7,8), (9,10,11,12)])  
    print("First row: {}".format(a[0,:]))  # prints [1 2 3 4]  
    print("Last column: {}".format(a[:,-1]))  # prints [ 4 8 12]

    # Data manipulation using pandas
    df = pd.DataFrame({
      'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
      'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
      'C': np.random.randn(8),
      'D': np.random.randn(8)
    })
    print(df.groupby(['A','B']).sum())

    # Plotting using matplotlib
    x = np.linspace(0, 10, 100)  
    plt.plot(x, np.sin(x))  
    plt.plot(x, np.cos(x))  
    plt.show()

    # Designing a model using keras
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',
              metrics=['accuracy'])
    data = np.random.random((1000, 100))
    labels = np.random.randint(2, size=(1000, 1))
    model.fit(data, labels, epochs=10, batch_size=32)

except Exception as e:
    print(str(e))