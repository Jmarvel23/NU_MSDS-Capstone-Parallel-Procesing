import numpy as np
import pandas as pd
import multiprocessing, time, os, keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

nb_classes = 10

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

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


def parallelPredict(data):
    model = keras.models.load_model('mnist_model')
    pred_results = []
    for row in data.iterrows():
        arr = row[1].to_numpy().reshape(1, 28, 28, 1)
        pred_results.append(model.predict(arr))
    return pred_results


def parallelize(df, fn):
    core_partition_count = multiprocessing.cpu_count()-1
    df_split = np.array_split(df, core_partition_count)
    pool = multiprocessing.Pool(core_partition_count)
    results = pool.map(fn, df_split)
    pool.close()
    pool.join()
    return results

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
