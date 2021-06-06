# NU_MSDS-Capstone-Parallel-Procesing

Repository for Northwestern University MSDS Capstone Group Project - Parallel Processing

This project highlights how to use the multiprocessing Python package to parellelize the processing of data through a model. We used the MNIST dataset as our example data, and built a CNN model with Keras to make the predictions. 

# Why is Parallel Processing Important?
The importance for quick processing time only increases as the problems we are solving become more complex and as such require more data to be analyzed per time step. Parallel processing is one of the major keys when aiming to improve processing time. To put numbers in perspective, our example project boasted a 22% decrease in processing time when using parallel processing vs. normal processing. This processing time improvement, of course, will increase as the number of cores to parallelize across increases.

# Code Highlited Below
## Define, Train, and Save Model

``` Python
def mnistModel(X_train, X_test, Y_train, Y_test):

    model1 = Sequential()

    # 2 Conv layers with max pooling
    model1.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
    model1.add(BatchNormalization(axis=-1))
    convLayer01 = Activation('relu')
    model1.add(convLayer01)

    model1.add(Conv2D(32, (3, 3)))
    model1.add(BatchNormalization(axis=-1))
    model1.add(Activation('relu'))
    convLayer02 = MaxPooling2D(pool_size=(2,2))
    model1.add(convLayer02)
    model1.add(Flatten())

    # Fully connected layer
    model1.add(Dense(512))
    model1.add(BatchNormalization())
    model1.add(Activation('relu'))

    # Fully connected layer with dropout
    model1.add(Dropout(0.2))
    model1.add(Dense(10))
    model1.add(Activation('softmax'))

    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                             height_shift_range=0.08, zoom_range=0.08)

    test_gen = ImageDataGenerator()

    train_generator = gen.flow(X_train, Y_train, batch_size=128)
    test_generator = test_gen.flow(X_test, Y_test, batch_size=128)

    model1.fit_generator(train_generator, steps_per_epoch=6000//128, epochs=5, verbose=1,
                        validation_data=test_generator, validation_steps=1000//128)

    model1.save('mnist_model')
```

## Define Parallel Prediction Function

``` Python
def parallelPredict(data):
    model = keras.models.load_model('mnist_model')
    pred_results = []
    for row in data.iterrows():
        arr = row[1].to_numpy().reshape(1, 28, 28, 1)
        pred_results.append(model.predict(arr))
    return pred_results
```

## Define Parallelization of Dataframe Based on Number of Cores

``` Python
def parallelize(df, fn):
    core_partition_count = multiprocessing.cpu_count()-1
    df_split = np.array_split(df, core_partition_count)
    pool = multiprocessing.Pool(core_partition_count)
    results = pool.map(fn, df_split)
    pool.close()
    pool.join()
    return results
```

## Call Functions to Run Our Example 

``` Python
if __name__ == "__main__":
    X_train_sub = X_train[0:6000, :, :, :] #Subset for quick model creation
    X_test_sub = X_test[0:1000, :, :, :] #Subset for quick model creation
    Y_train_sub = y_train[0:6000] #Subset for quick model creation
    Y_test_sub = y_test[0:1000] #Subset for quick model creation

    current_dir = os.listdir(".")
    if 'mnist_model' not in dir:
        mnistModel(X_train_sub, X_test_sub, Y_train_sub, Y_test_sub)

    X_test_parallel = X_train[6000:7000, :, :, :]
    test_df = pd.DataFrame(X_test_parallel.reshape(1000, 784))

    parallelStart = time.time()
    parallelDF = parallelize(test_df, parallelPredict)
    parallelEnd = time.time()
    parallelTime = parallelEnd - parallelStart

    linearStart = time.time()
    linearDF = parallelPredict(test_df)
    linearEnd = time.time()
    linearTime = linearEnd - linearStart

    print(parallelTime)
    print(linearTime)
```
