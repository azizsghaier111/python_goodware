def main():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.utils import np_utils
    from keras.optimizers import SGD

    # simple statistics with numpy
    my_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print('Mean:', np.mean(my_array))
    print('Median:', np.median(my_array))
    print('Variance:', np.var(my_array))

    # complex mathematical functions with numpy
    print('Exponential:', np.exp(my_array))

    # random number generation with numpy
    print('Random number:', np.random.rand())

    # creating pandas dataframe
    df = pd.DataFrame(my_array, columns=['Column1'])
    print(df.describe())

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
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    # training model
    model.fit(X, y, epochs=10, batch_size=32)

    # evaluate model
    scores = model.evaluate(X, y, verbose=0)
    print("Model Accuracy: %.2f%%" % (scores[1]*100))

    # prediction
    predictions = model.predict(X[:5])
    print('Predictions:', predictions)

    # complex mathematical function representation
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    plt.plot(x, y)
    plt.title('Sine wave function using Numpy and Matplotlib')
    plt.show()


if __name__ == "__main__":
    main()