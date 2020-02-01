'''
    Creation Date: Feb 3, 2020
    Author: Tymoore Jamal
    Content: Trains a machine learning model to perform image classification on our labelled data.
'''

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
import json

with open('config.json') as jsonFile:
        settings = json.load(jsonFile)


def display_image(img_array):
    '''
        To display image
        @param img_array
        @return img
    '''
    img = Image.fromarray(img_array, 'L')
    return img


def load_images(base, numberOfFilesInSample, RGB):
    '''
        load images based on whether they are colored
        @param base
        @param numberOfFilesInSample
        @param RGB
        @return images
    '''
    files = sample([f for f in listdir(base)], numberOfFilesInSample)
    if RGB:
        images = np.array([np.array(Image.open(join(base,fname))) for fname in files])
    else:
        images = np.array([np.array(Image.open(join(base,fname)).convert('L')) for fname in files])
    return images

def load_all_images(faceBase, nonFaceBase, numberOfFilesInSample, RGB):
    '''
        load all images both faces and non faces
        @param faceBase
        @param nonFaceBase
        @param numberOfFilesInSample
        @param RGB
        @return nonFaceImages
        @return faceImages
    '''
    faceImages = load_images(faceBase, numberOfFilesInSample, RGB)
    nonFaceImages = load_images(nonFaceBase, numberOfFilesInSample, RGB)
    return faceImages, nonFaceImages

def normalize_images(images):
    '''
        normalize images
        @param images
        @param nonFaceBase
        @return normalized images
    '''
    return images / settings["PixelMaxValues"]

def get_labeled_data(numberOfFilesInSample, faceBase, nonFaceBase, RGB):
    '''
        get the labeled images that are faces or non faces
        @param faceBase
        @param nonFaceBase
        @param numberOfFilesInSample
        @return RGB
        @return labeled data
    '''
    faceImages, nonFaceImages = load_all_images(faceBase, nonFaceBase, numberOfFilesInSample, RGB)

    normalizedFaceImages = normalize_images(faceImages)
    normalizedNonFaceImages = normalize_images(nonFaceImages)

    labeledFaceImages = np.array([np.array([faceImage, [1]]) for faceImage in normalizedFaceImages])
    labeledNonFaceImages = np.array([np.array([faceImage, [0]]) for faceImage in normalizedNonFaceImages])

    labeledData = np.concatenate((labeledFaceImages, labeledNonFaceImages))
    np.random.shuffle(labeledData)

    return labeledData

def get_train_validation_data(numberOfFilesInSample, faceBase, nonFaceBase, trainSplit, RGB):
    '''
        get train validation data
        @param faceBase
        @param nonFaceBase
        @param numberOfFilesInSample
        @param trainSplit
        @return RGB
        @return trainingData
        @return validationData
    '''
    labeledData = get_labeled_data(numberOfFilesInSample, faceBase, nonFaceBase, RGB)

    trainingDataSamples = int(len(labeledData) * trainSplit)

    trainingData = labeledData[0:trainingDataSamples]
    validationData = labeledData[trainingDataSamples:]

    return trainingData, validationData

def create_model(numberOfHiddenLayers, imageShape):
    '''
        create model
        @param numberOfHiddenLayers
        @return model
    '''
    model = keras.Sequential()
    model.add(Flatten(input_shape=imageShape))

    for n in range(numberOfHiddenLayers):
        model.add(Dense(settings["DenseNodes"], activation='relu'))

    model.add(Dense(settings["OutputNodes"], activation='sigmoid'))


    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    return model

def create_conv_2D_model(numberOfHiddenLayers, imageShape):
    '''
    create convolutional 2D model using the sequence
    @param numberOfHiddenLayers
    @return model
    '''
    model = keras.Sequential()

    model.add(Conv2D(settings["ConvNodesOne"], (settings["ConvOutputDim"], settings["ConvOutputDim"]), input_shape=imageShape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(settings["PoolSize"], settings["PoolSize"])))

    model.add(Conv2D(settings["ConvNodesTwo"], (settings["ConvOutputDim"], settings["ConvOutputDim"]), activation='relu'))
    model.add(MaxPooling2D(pool_size=(settings["PoolSize"], settings["PoolSize"])))

    model.add(Conv2D(settings["ConvNodesTwo"], (settings["ConvOutputDim"], settings["ConvOutputDim"]), activation='relu'))
    model.add(MaxPooling2D(pool_size=(settings["PoolSize"], settings["PoolSize"])))

    model.add(Flatten())

    for n in range(numberOfHiddenLayers):
        model.add(Dense(settings["DenseNodes"], activation='relu'))
        model.add(Dropout(settings["Dropout"]))

    model.add(Dense(settings["OutputNodes"], activation='sigmoid'))

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    return model

def train_model(model, trainingInputs, trainingOutputs, validationInputs, validationOutputs, epochs, tensorboard, modelDirectory, imageShape):
    '''
        train model
        @param model
        @param trainingInputs
        @param trainingOutputs
        @param validationInputs
        @param validationOutputs
        @param epochs
        @param tensorboard
        @param modelDirectory
    '''
    model.fit(x = trainingInputs,
              y = trainingOutputs,
              validation_data = (validationInputs,
                                 validationOutputs),
              epochs = epochs,
              callbacks = [tensorboard])

def create_train_save_model(trainingInputs, trainingOutputs, validationInputs, validationOutputs, epochs, \
    modelDirectory, numberOfHiddenLayers, C2D, logDirectory, imageShape):
    '''
        create train model save model
        @param trainingInputs
        @param trainingOutputs
        @param validationInputs
        @param validationOutputs
        @param epochs
        @param tensorboard
        @param numberOfHiddenLayers
        @param modelDirectory
        @param C2D
    '''
    if C2D:
        model = create_conv_2D_model(numberOfHiddenLayers, imageShape)
    else:
        model = create_model(numberOfHiddenLayers, imageShape)

    name = "Input-"

    if C2D:
        name += "C-"

    name += str(numberOfHiddenLayers) + "-"

    name += "Output--"
    name += datetime.now().strftime("%Y-%m--%d-%H-%M")

    tensorboard = keras.callbacks.TensorBoard(log_dir=join(logDirectory, name))

    train_model(model, trainingInputs, trainingOutputs, validationInputs, validationOutputs, epochs, tensorboard, modelDirectory, imageShape)

    try:
        model.save(join(modelDirectory, name))
    except:
        print(":(")

def run_multiple_models(faceBase, nonFaceBase, rangeOfNumberOfHiddenLayers, modelDirectory, logDirectory, imageShape):
    '''
        run multiple models
        @param numberOfFilesInSample
        @param faceBase
        @param nonFaceBase
        @param trainSplit
        @param rangeOfNumberOfHiddenLayers
        @param epochs
        @param RGB
        @param C2D
    '''
    trainingData, validationData = get_train_validation_data(settings["numberOfFilesInSample"], faceBase, nonFaceBase,\
        settings["trainSplit"], settings["RGB"])

    trainingInputs = np.array([trainingData[:,0][i] for i in range(len(trainingData))])
    trainingOutputs = np.array([trainingData[:,1][i] for i in range(len(trainingData))])

    validationInputs = np.array([validationData[:,0][i] for i in range(len(validationData))])
    validationOutputs = np.array([validationData[:,1][i] for i in range(len(validationData))])

    for hl in rangeOfNumberOfHiddenLayers:
        create_train_save_model(trainingInputs, trainingOutputs, validationInputs, validationOutputs, \
            settings["epochs"], modelDirectory, hl, settings["C2D"], logDirectory, imageShape)


def main():
    '''
        run multiple models
        @param numberOfFilesInSample
        @param faceBase
        @param nonFaceBase
        @param trainSplit
        @param rangeOfNumberOfHiddenLayers
        @param epochs
        @param RGB
        @param C2D
    '''
    rangeOfNumberOfHiddenLayers = range(settings["rangeOfNumberOfHiddenLayersMin"],settings["rangeOfNumberOfHiddenLayersMax"])
    logDirectory = join("Logs", datetime.now().strftime("%Y-%m-%d"))
    modelDirectory = join("Models", datetime.now().strftime("%Y-%m-%d"))
    base = join("..", "..", "ML Data", "250 processed")
    faceBase = join(base, "face")
    nonFaceBase = join(base, "non-face")

    if settings["RGB"]:
        imageShape = (settings["ImageWidth"], settings["ImageHeight"], settings["RGBDimentions"])
    else:
        imageShape = (settings["ImageWidth"], settings["ImageHeight"])

    run_multiple_models(faceBase, nonFaceBase, rangeOfNumberOfHiddenLayers, modelDirectory, logDirectory, imageShape)


if __name__ == '__main__':
    main()
