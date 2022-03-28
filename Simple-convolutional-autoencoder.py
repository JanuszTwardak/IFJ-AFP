import keras
import pickle
import tensorflow as tf
import numpy as np
from os import mkdir
import matplotlib.pyplot as plt


def prepareData(dataInputPath, validateFraction):

    trainEvents = np.load(dataInputPath)
    validateSize = int(len(trainEvents) * validateFraction)

    validateEvents = np.array(trainEvents[-validateSize:])
    trainEvents = np.array(trainEvents[:-validateSize])

    trainEvents = trainEvents.astype("float16")
    validateEvents = validateEvents.astype("float16")

    return [trainEvents, validateEvents]


def trainNeuralNetwork(
    modelOutputPath,
    trainEvents,
    validateEvents,
    singleInputShape,
    parameters,
    trainingInformation,
):

    encoderInput = keras.Input(
        shape=(
            singleInputShape["x"],
            singleInputShape["y"],
            singleInputShape["channels"],
        )
    )

    x = keras.layers.Conv2D(16, (4, 3), activation="relu", padding="same")(encoderInput)
    x = keras.layers.MaxPooling2D(pool_size=(4, 4))(x)

    x = keras.layers.Conv2D(
        32,
        (4, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(l=0.0001),
    )(x)
    x = keras.layers.MaxPooling2D(pool_size=(4, 4))(x)

    x = keras.layers.Conv2D(
        64,
        (4, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(l=0.0001),
    )(x)

    x = keras.layers.Conv2D(
        64,
        (4, 4),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(l=0.0001),
    )(x)
    x = keras.layers.UpSampling2D((4, 4))(x)

    x = keras.layers.Conv2D(
        32,
        (4, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(l=0.0001),
    )(x)
    x = keras.layers.UpSampling2D((4, 4))(x)

    decoderOutput = keras.layers.Conv2D(
        4, (4, 3), activation="sigmoid", padding="same"
    )(x)

    autoencoder = keras.Model(encoderInput, decoderOutput, name="autoencoder")
    autoencoder.summary()

    stringlist = []
    autoencoder.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)

    autoencoder.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    history = autoencoder.fit(
        trainEvents,
        trainEvents,
        epochs=parameters["epochs"],
        batch_size=parameters["batchSize"],
        shuffle=True,
        validation_data=(validateEvents, validateEvents),
    )

    if modelOutputPath != "0":
        import time

        path = modelOutputPath + time.strftime("/autoencoder_%d-%m-%Y_%H-%M-%S")
        autoencoder.save(path)

        f = open(path + "/history.pckl", "wb")
        pickle.dump(history.history, f)
        f.close()

        f = open(path + "/training-parameters.txt", "w+")
        f.write("- epochs: " + str(parameters["epochs"]) + "\n")
        f.write("- batch size: " + str(parameters["batchSize"]) + "\n \n")
        f.write("- input: " + str(trainingInformation["inputPath"]) + "\n")
        f.write(
            "- number of events: " + str(len(trainEvents) + len(validateEvents)) + "\n"
        )
        f.write(
            "- validate fraction: "
            + str(trainingInformation["validateFraction"])
            + "\n \n"
        )
        f.write(
            "- training accuracy (last epoch): "
            + str(history.history.get("accuracy")[-1])
            + "\n"
        )
        f.write(
            "- validation accuracy (last epoch): "
            + str(history.history.get("val_accuracy")[-1])
            + "\n \n"
        )
        f.write(
            "- training loss (last epoch): "
            + str(history.history.get("loss")[-1])
            + "\n"
        )
        f.write(
            "- validation loss (last epoch): "
            + str(history.history.get("val_loss")[-1])
            + "\n \n \n"
        )
        f.write(short_model_summary)
        f.close()

        graphPath = path + "/trained-model-graphs"
        os.mkdir(graphPath)
        graphPath = graphPath + "/"
        plotResults(
            autoencoder,
            validateEvents,
            singleInputShape,
            history.history,
            shownEventNumber=0,
            save={"save": True, "path": graphPath},
        )
    return [history, autoencoder]


def loadTrainedModel(loadedModelPath):
    f = open(loadedModelPath + "/history.pckl", "rb")
    history = pickle.load(f)
    f.close()
    return [keras.models.load_model(loadedModelPath), history]


def plotResults(
    autoencoder, validateEvents, singleInputShape, history, shownEventNumber, save
):
    encodedEvents = autoencoder.predict(validateEvents)
    plotLossAccuracy(history, save)
    reconstructionErrors, anomalyThreshold = calculateThresholds(
        encodedEvents, validateEvents, singleInputShape, save
    )

    if save["save"] == False:
        showReconstructedEvents(
            shownEventNumber,
            validateEvents,
            encodedEvents,
            reconstructionErrors,
            anomalyThreshold,
            save,
        )
    else:
        for eventNumber in range(10):
            showReconstructedEvents(
                eventNumber,
                validateEvents,
                encodedEvents,
                reconstructionErrors,
                anomalyThreshold,
                save,
            )


def plotLossAccuracy(history, save):
    plt.plot(history["loss"], label="train")
    plt.plot(history["val_loss"], label="validation")
    title = "cost function"
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("cost")
    plt.legend()

    if save["save"] == True:
        plt.savefig(save["path"] + title + ".png")
    plt.show()

    plt.plot(history["accuracy"])
    plt.plot(history["val_accuracy"])
    title = "model accuracy"
    plt.title(title)
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")

    if save["save"] == True:
        plt.savefig(save["path"] + title + ".png")
    plt.show()


def calculateThresholds(encodedEvents, validateEvents, singleInputShape, save):

    flatLength = (
        singleInputShape["x"] * singleInputShape["y"] * singleInputShape["channels"]
    )
    encodedEventsFlat = np.reshape(encodedEvents, [len(encodedEvents), flatLength])

    validateEventsFlat = np.reshape(validateEvents, [len(encodedEvents), flatLength])
    reconstructionErrors = tf.keras.losses.mse(encodedEventsFlat, validateEventsFlat)
    encodedEventsFlat = None
    validateEventsFlat = None

    anomalyThreshold = np.mean(reconstructionErrors.numpy()) + np.std(
        reconstructionErrors.numpy()
    )
    print("Reconstruction error threshold: ", anomalyThreshold)

    # Train MAE loss.
    plt.axvline(anomalyThreshold, color="k", linestyle="dashed", linewidth=1)
    plt.hist(reconstructionErrors, bins=100)
    plt.xlabel("MSE loss")
    plt.ylabel("Number of samples")
    title = "MSE loss histogram"
    plt.title(title)
    if save["save"] == True:
        plt.savefig(save["path"] + title + ".png")
    plt.show()

    # Train MAE loss in logarithmic scale
    plt.axvline(anomalyThreshold, color="k", linestyle="dashed", linewidth=1)
    plt.hist(reconstructionErrors, bins=500)
    plt.xlabel("MSE loss")
    plt.ylabel("Number of samples (log)")
    plt.yscale("log")
    title = "MSE loss histogram (log scale)"
    plt.title(title)
    if save["save"] == True:
        plt.savefig(save["path"] + title + ".png")
    plt.show()

    return reconstructionErrors, anomalyThreshold


def showReconstructedEvents(
    eventNumber,
    validateEvents,
    encodedEvents,
    reconstructionErrors,
    anomalyThreshold,
    save,
):
    validateEvents = validateEvents.astype("float32")
    encodedEvents = encodedEvents.astype("float32")

    fig = plt.figure(dpi=300)
    rows = 2
    columns = 4

    position = 1
    for plane in range(4):
        fig.add_subplot(rows, columns, position)
        position += 1
        plt.imshow(validateEvents[eventNumber, :, :, plane])
    for plane in range(4):
        fig.add_subplot(rows, columns, position)
        position += 1
        plt.imshow(encodedEvents[eventNumber, :, :, plane])

    reconstructionErrors = tf.get_static_value(reconstructionErrors)

    if reconstructionErrors[eventNumber] < anomalyThreshold:
        plt.text(
            -550,
            410,
            "Reconstruction error: " + str(reconstructionErrors[eventNumber]),
            fontsize=4,
        )
    else:
        plt.text(
            -550,
            410,
            "Reconstruction error: " + str(reconstructionErrors[eventNumber]),
            fontsize=4,
            color="r",
        )

    if save["save"] == True:
        title = "reconstructed_event_number_" + str(eventNumber + 1)
        plt.savefig(save["path"] + title + ".png")
    plt.show()


def main():
    inputDataPath = "drive/MyDrive/AFP-ML/output-data/hits.npy"
    validateFraction = 0.1

    trainEvents, validateEvents = prepareData(inputDataPath, validateFraction)

    outputModelPath = "drive/MyDrive/AFP-ML/trained-models"
    parameters = {"epochs": 2, "batchSize": 2}
    singleInputShape = {"x": 336, "y": 80, "channels": 4}
    trainingInformation = {
        "inputPath": inputDataPath,
        "validateFraction": validateFraction,
    }

    history, autoencoder = trainNeuralNetwork(
        outputModelPath,
        trainEvents,
        validateEvents,
        singleInputShape,
        parameters,
        trainingInformation,
    )

    plotResults(
        autoencoder,
        validateEvents,
        singleInputShape,
        history,
        shownEventNumber,
        save={"save": False, "path": "0"},
    )


if __name__ == "__main__":
    main()
