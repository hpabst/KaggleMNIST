import numpy as np
import pandas as pd
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt
import os
import pickle

TEST_PATH = "data/test.csv"
TRAIN_PATH = "data/train.csv"


def read_data(path):
    data = pd.read_csv(path)
    if "label" in data:
        feat_cols = data.columns.tolist()[1:]
        Y = data['label'].values
        Y = np_utils.to_categorical(Y, 10)
    else:
        feat_cols = data.columns.tolist()
        Y = None
    X = data[feat_cols].values.reshape(-1, 28, 28, 1)/255
    return X, Y


def save_model(model, history, path):
    if not os.path.isdir(path):
        os.mkdir(path)
    model.save(path+"/model.h5")
    with open(path+"/history", "wb") as f:
        pickle.dump(history.history, f)
    return


def save_test_results(labels, path):
    processed = process_labels(labels)
    with open(path+"/test_results.csv", "w+") as f:
        f.write("ImageId,Label\n")
        f.writelines(["{},{}\n".format(i+1, l) for (i, l) in enumerate(processed)])
    return


def save_train_results(model, x, y, path):
    score = model.evaluate(x, y, verbose=0)
    processed = process_labels(y)
    with open(path+"/train_results.csv", "w+") as f:
        f.write("{}: {}% \n".format(model.metrics_names[1], score[1]))
        f.write("ImageId,Label\n")
        f.writelines(["{},{}\n".format(i+1, l) for (i, l) in enumerate(processed)])


def process_labels(labels):
    return [np.argmax(i) for i in labels]


def test_data(model, x_test):
    y_test = model.predict(x_test)
    return y_test


def display_image(data):
    plt.gray()
    plt.imshow(data[:,:,0])
    plt.show()
    return


def define_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28, 28, 1)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model



def main():
    X_train, Y_train = read_data(TRAIN_PATH)
    print(X_train.shape)
    print(Y_train.shape)
    display_image(X_train[0])

    model = define_model()
    history = model.fit(X_train, Y_train, batch_size=64, nb_epoch=50, verbose=2)
    dt_str = datetime.now().strftime("%B-%d-%Y-%I%M%p")
    path = "results/{}".format(dt_str)
    save_model(model, history, path)
    save_train_results(model, X_train, Y_train, path)
    X_test, _ = read_data(TEST_PATH)
    y_test = test_data(model, X_test)
    save_test_results(y_test, path)

    return


if __name__ == "__main__":
    main()