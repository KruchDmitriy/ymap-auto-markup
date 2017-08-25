from keras.models import Model
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Deconv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from keras.layers.core import Dropout
from keras.optimizers import Adam
from keras.metrics import binary_crossentropy
import numpy as np
from os import listdir
import cv2

from matplotlib import pyplot as plt


PATH_TO_IMG = "../preprocessedData/imgs/"
PATH_TO_LABELS = "../preprocessedData/labels/"
LOG = False

def iou_metric(y_true, y_pred):
    true = y_true > 0
    pred = y_pred > 0

    inter = np.count_nonzero(np.logical_and(true, pred))
    union = np.count_nonzero(np.logical_or(true, pred))
    return inter / union


def data_generator(chunk_size=500):
    # imgs = []
    # labels = []
    for file in listdir(PATH_TO_IMG):
        img = cv2.imread(PATH_TO_IMG + file)
        img = cv2.resize(img, (512, 512))

        label = cv2.imread(PATH_TO_LABELS + file, 0)
        label = cv2.resize(label, (512, 512))
        label = cv2.normalize(label, label, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        img = img.reshape((1, 512, 512, 3))
        label = label.reshape((1, 512, 512, 1))

        # imgs.append(img)
        # labels.append(label)

        # if len(imgs) == 32:
        #     yield (imgs, labels)
        #     imgs = []
        #     labels = []
        # print(img.shape)
        # print(label.shape)

        yield (img, label)


def create_model():
    input_layer = Input(shape=(512, 512, 3))

    bn0 = BatchNormalization()(input_layer)

    conv1 = Conv2D(filters=50, kernel_size=5, activation='relu')(bn0)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    bn1 = BatchNormalization()(pool1)

    conv2 = Conv2D(filters=70, kernel_size=5, activation='relu')(bn1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    bn2 = BatchNormalization()(pool2)

    conv3 = Conv2D(filters=100, kernel_size=3, activation='relu')(bn2)
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv3)
    bn3 = BatchNormalization()(pool3)

    conv4 = Conv2D(filters=150, kernel_size=3, activation='relu')(bn3)
    pool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv4)
    bn4 = BatchNormalization()(pool4)

    conv5 = Conv2D(filters=150, kernel_size=1, activation='relu')(bn4)
    conv6 = Conv2D(filters=150, padding='same', kernel_size=3, activation='relu')(bn4)
    concat1 = concatenate([conv5, conv6], axis=0)

    conv7 = Conv2D(filters=100, padding='same', kernel_size=3, activation='relu')(concat1)
    drop7 = Dropout(rate=0.5)(conv7)
    bn7 = BatchNormalization()(drop7)

    deconv1 = Deconv2D(filters=100, kernel_size=5, strides=(2, 2), use_bias=False)(bn7)
    conv8 = Conv2D(filters=100, padding='same', kernel_size=3, activation='relu')(bn3)
    concat2 = concatenate([deconv1, conv8], axis=0)

    conv9 = Conv2D(filters=70, padding='same', kernel_size=3, activation='relu')(concat2)
    drop9 = Dropout(rate=0.5)(conv9)
    bn9 = BatchNormalization()(drop9)

    deconv2 = Deconv2D(filters=70, use_bias=False, kernel_size=5, strides=(2, 2))(bn9)
    conv10 = Conv2D(filters=70, padding='same', kernel_size=3, activation='relu')(bn2)
    concat3 = concatenate([deconv2, conv10], axis=0)

    conv11 = Conv2D(filters=50, padding='same', kernel_size=3, activation='relu')(concat3)
    drop11 = Dropout(rate=0.5)(conv11)
    bn11 = BatchNormalization()(drop11)

    deconv3 = Deconv2D(filters=50, use_bias=False, kernel_size=6, strides=(2, 2))(bn11)
    conv12 = Conv2D(filters=50, padding='same', kernel_size=3, activation='relu')(bn1)
    concat4 = concatenate([deconv3, conv12], axis=0)

    conv13 = Conv2D(filters=1, padding='same', kernel_size=3, activation='relu')(concat4)
    score = Deconv2D(filters=1, use_bias=False, kernel_size=6, strides=(2,2))(conv13)

    model = Model(inputs=[input_layer], outputs=score)
    model.compile(optimizer=Adam(), loss='binary_crossentropy',
        metrics=[binary_crossentropy])

    return model


def main():
    model = create_model()

    if LOG:
        from keras.utils import plot_model
        plot_model(model)

    nb_epoch = 100

    # print(next(data_generator()))
    model.fit_generator(data_generator(), steps_per_epoch=500, epochs=100, verbose=1)

    # for e in range(nb_epoch):
    #     print("epoch %d" % e)
    #     for img, label in data_generator():
    #         model.fit(img, label, batch_size=32, epochs=1)


if __name__ == "__main__":
    main()
