import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
import numpy as np
from os import listdir, rmdir
from os.path import join
from PIL import Image
from random import sample

def display_image(img_array):
    img = Image.fromarray(img_array, 'L')
    return img

def load_images(base, numberOfFilesInSample, RGB):
    files = sample([f for f in listdir(base)], numberOfFilesInSample)
    if RGB:
        images = np.array([np.array(Image.open(join(base,fname))) for fname in files])
    else:
        images = np.array([np.array(Image.open(join(base,fname)).convert('L')) for fname in files])
    return images

def load_all_images(faceBase, nonFaceBase, numberOfFilesInSample, RGB):
    faceImages = load_images(faceBase, numberOfFilesInSample, RGB)
    nonFaceImages = load_images(nonFaceBase, numberOfFilesInSample, RGB)
    return faceImages, nonFaceImages


def normalize_images(images):
    return images / 255


def get_labeled_data(numberOfFilesInSample, faceBase, nonFaceBase, RGB):
    faceImages, nonFaceImages = load_all_images(faceBase, nonFaceBase, numberOfFilesInSample, RGB)

    normalizedFaceImages = normalize_images(faceImages)
    normalizedNonFaceImages = normalize_images(nonFaceImages)

    labeledFaceImages = np.array([np.array([faceImage, [1]]) for faceImage in normalizedFaceImages])
    labeledNonFaceImages = np.array([np.array([faceImage, [0]]) for faceImage in normalizedNonFaceImages])

    labeledData = np.concatenate((labeledFaceImages, labeledNonFaceImages))
    np.random.shuffle(labeledData)

    return labeledData

def get_train_validation_data(numberOfFilesInSample, faceBase, nonFaceBase, trainSplit, RGB):
    labeledData = get_labeled_data(numberOfFilesInSample, faceBase, nonFaceBase, RGB)

    trainingDataSamples = int(len(labeledData) * trainSplit)

    trainingData = labeledData[0:trainingDataSamples]
    validationData = labeledData[trainingDataSamples:]

    return trainingData, validationData

def create_model(numberOfHiddenLayers):
    model = keras.Sequential()
    model.add(Flatten(input_shape=imageShape))

    for n in range(numberOfHiddenLayers):
        model.add(Dense(128, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    # model.summary()

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    return model

def create_conv_2D_model(numberOfHiddenLayers):
    model = keras.Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=imageShape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    for n in range(numberOfHiddenLayers):
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))

    # model.summary()

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    return model

def train_model(model, trainingInputs, trainingOutputs, validationInputs, validationOutputs, epochs, tensorboard, modelDirectory):

    model.fit(x = trainingInputs,
              y = trainingOutputs,
              validation_data = (validationInputs,
                                 validationOutputs),
              epochs = epochs,
              callbacks = [tensorboard])

def create_train_save_model(trainingInputs, trainingOutputs, validationInputs, validationOutputs, epochs, modelDirectory, numberOfHiddenLayers, C2D):
    if C2D:
        model = create_conv_2D_model(numberOfHiddenLayers)
    else:
        model = create_model(numberOfHiddenLayers)

    name = "Input-"

    if C2D:
        name += "C-"

    name += str(numberOfHiddenLayers) + "-"
#     for n in range(numberOfHiddenLayers):
#         name += "Dense128-"
    name += "Output--"
    name += datetime.now().strftime("%Y-%m--%d-%H-%M")

    tensorboard = keras.callbacks.TensorBoard(log_dir=join(logDirectory, name))

    train_model(model, trainingInputs, trainingOutputs, validationInputs, validationOutputs, epochs, tensorboard, modelDirectory)

    try:
        model.save(join(modelDirectory, name))
    except:
        print(":(")

def run_multiple_models(numberOfFilesInSample, faceBase, nonFaceBase, trainSplit, rangeOfNumberOfHiddenLayers, epochs, RGB, C2D):
    trainingData, validationData = get_train_validation_data(numberOfFilesInSample, faceBase, nonFaceBase, trainSplit, RGB)

    trainingInputs = np.array([trainingData[:,0][i] for i in range(len(trainingData))])
    trainingOutputs = np.array([trainingData[:,1][i] for i in range(len(trainingData))])

    validationInputs = np.array([validationData[:,0][i] for i in range(len(validationData))])
    validationOutputs = np.array([validationData[:,1][i] for i in range(len(validationData))])

    for hl in rangeOfNumberOfHiddenLayers:
        create_train_save_model(trainingInputs, trainingOutputs, validationInputs, validationOutputs, epochs, modelDirectory, hl, C2D)


def main():
    numberOfFilesInSample = 1500
    epochs = 100
    rangeOfNumberOfHiddenLayers = range(1,6)
    trainSplit = 0.8
    C2D = True
    RGB = C2D
    logDirectory = join("Logs", datetime.now().strftime("%Y-%m-%d"))
    modelDirectory = join("Models", datetime.now().strftime("%Y-%m-%d"))
    base = join("..", "..", "ML Data", "250 processed")
    faceBase = join(base, "face")
    nonFaceBase = join(base, "non-face")


    if RGB:
        imageShape = (250, 250, 3)
    else:
        imageShape = (250, 250)

    run_multiple_models(numberOfFilesInSample, faceBase, nonFaceBase, trainSplit, rangeOfNumberOfHiddenLayers, epochs, RGB, C2D)


if __name__ == '__main__':
    main()
