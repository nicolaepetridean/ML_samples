"""
Title: Image classification from scratch
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/27
Last modified: 2020/04/28
Description: Training an image classifier from scratch on the Kaggle Cats vs Dogs dataset.
"""
"""
## Introduction
This example shows how to do image classification from scratch, starting from JPEG
image files on disk, without leveraging pre-trained weights or a pre-made Keras
Application model. We demonstrate the workflow on the Kaggle Cats vs Dogs binary
 classification dataset.
We use the `image_dataset_from_directory` utility to generate the datasets, and
we use Keras image preprocessing layers for image standardization and data augmentation.
"""

"""
## Setup
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import datetime

"""
## Load the data: the Cats vs Dogs dataset
### Raw data download
First, let's download the 786M ZIP archive of the raw data:
"""

"""shell
curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
"""

"""shell
unzip -q images.zip
ls
"""

"""
Now we have a `PetImages` folder which contain two subfolders, `Cat` and `Dog`. Each
 subfolder contains image files for each category.
"""

"""shell
ls PetImages
"""

"""
### Filter out corrupted images
When working with lots of real-world image data, corrupted images are a common
occurence. Let's filter out badly-encoded images that do not feature the string "JFIF"
 in their header.
"""


def check_file_is_valid_JPEG():
    import os
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join("data\\images\\PetImages", folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)
    print("Deleted %d images" % num_skipped)


def generate_dataset():
    """
    ## Generate a `Dataset`
    """
    image_size = (180, 180)
    batch_size = 32

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data\\images\\PetImages",
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data\\images\\PetImages",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    return train_ds, val_ds


def visualize_data(train_ds):
    """
    ## Visualize the data
    Here are the first 9 images in the training dataset. As you can see, label 1 is "dog"
     and label 0 is "cat".
    """
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")


def get_data_augmentation():
    """
    ## Using image data augmentation
    When you don't have a large image dataset, it's a good practice to artificially
    introduce sample diversity by applying random yet realistic transformations to the
    training images, such as random horizontal flipping or small random rotations. This
    helps expose the model to different aspects of the training data while slowing down
     overfitting.
    """
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )

    return data_augmentation


def visualize_some_augmented_pics(train_ds, data_augmentation):
    """
    Let's visualize what the augmented samples look like, by applying `data_augmentation`
     repeatedly to the first image in the dataset:
    """
    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")


def make_model(input_shape, num_classes, data_augmentation):
    """
    ## Build a model
    We'll build a small version of the Xception network. We haven't particularly tried to
    optimize the architecture; if you want to do a systematic search for the best model
     configuration, consider using
    [Keras Tuner](https://github.com/keras-team/keras-tuner).
    Note that:
    - We start the model with the `data_augmentation` preprocessor, followed by a
     `Rescaling` layer.
    - We include a `Dropout` layer before the final classification layer.
    """
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


def train_model(train_ds, val_ds, model):
    """
    ## Train the model
    """
    epochs = 50
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        keras.callbacks.ModelCheckpoint("data\\saved_model\\save_at_{epoch}.h5"),
        keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    )

    return model


def load_trained_model(input_shape, num_classes, data_augmentation):
   model = make_model(input_shape, num_classes, data_augmentation)
   model.load_weights('data\\saved_model\\save_at_50.h5')
   return model


def train_model_pipeline():
    image_size = (180, 180)
    batch_size = 32

    check_file_is_valid_JPEG()
    train_ds, val_ds = generate_dataset()
    visualize_data(train_ds)
    data_augmentation = get_data_augmentation()
    visualize_some_augmented_pics(train_ds, data_augmentation)
    model = make_model(input_shape=image_size + (3,), num_classes=2, data_augmentation=data_augmentation)
    # keras.utils.plot_model(model, show_shapes=True)
    train_model(train_ds, val_ds, model)


if __name__ == '__main__':
    print('Done training, running inference')
    image_size = (180, 180)
    data_augmentation = get_data_augmentation()

    model = load_trained_model(image_size + (3,), 2, data_augmentation)

    img = keras.preprocessing.image.load_img(
        "data\\images\\PetImages\\Dog\\259.jpg", target_size=image_size
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = predictions[0]
    print(
        "This image is %.2f percent cat and %.2f percent dog."
        % (100 * (1 - score), 100 * score)
    )