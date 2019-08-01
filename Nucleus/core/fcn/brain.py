import tensorflow as tf
import os
import sqlite3
from Nucleus.core.ROI import ROI
from Nucleus.core.ROIHandler import ROIHandler
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from scipy import ndimage as ndi


class FCN:
    """
    Fully convolutional net as described by Long et al. (2014)
    """

    def __init__(self, ids=None, epochs=100, batch_size=1):
        self.ids = []
        self.epochs = epochs
        self.batch_size = batch_size
        self.connection = sqlite3.connect(os.path.join(os.pardir, "database{}nucdetect.db".format(os.sep)))
        self.cursor = self.connection.cursor()
        saved_ids = self.cursor.execute(
            "SELECT md5 FROM images WHERE analysed=?",
            (True, )
        )
        for id in saved_ids:
            if ids is None:
                self.ids.append(id[0])
            else:
                if id[0] in ids:
                    self.ids.append((id[0]))
        self.data = FCN.create_region_maps_from_database(self.ids)

    @staticmethod
    def create_region_maps_from_database(cursor, ids=None):
        """
        Method to create labelled maps of the images given by ids. if None, all available images are used
        :param cursor: A cursor pointing to the database
        :param ids: List of md5 hashs specifying the used images
        :return: A tuple of the created region maps
        """
        r = []
        if not ids:
            return r
        else:
            for md5 in ids:
                rois = ROIHandler(ident=md5)
                entries = cursor.execute(
                    "SELECT * FROM roi WHERE image = ?",
                    (md5,)
                ).fetchall()
                names = cursor.execute(
                    "SELECT * FROM channels WHERE md5 = ?",
                    (md5,)
                ).fetchall()
                for name in names:
                    rois.idents.insert(name[1], name[2])
                main = []
                sec = []
                for entry in entries:
                    temproi = ROI(channel=entry[3], main=entry[7] is None, associated=entry[7])
                    if temproi.main:
                        main.append(temproi)
                    else:
                        sec.append(temproi)
                    for p in cursor.execute(
                            "SELECT * FROM points WHERE hash = ?",
                            (entry[0],)
                    ).fetchall():
                        temproi.add_point((p[1], p[2]), p[3])
                    rois.add_roi(temproi)
                for m in main:
                    for s in sec:
                        if s.associated == hash(m):
                            s.associated = m
                r.append(rois)
            return FCN.create_region_maps_from_roi_handler(r)

    @staticmethod
    def create_region_maps_from_roi_handler(rois, images):
        maps = []
        for handler in rois:
            for ident in handler.idents:
                tmaps = [[]]
        return maps

    @staticmethod
    def check_avaiable_images(folder):
        """
        Method to get the md5 hashes of all images in folder
        :param folder: The folder containing the images
        :return:
        """
        images = []
        for t in os.walk(folder):
            for file in t[2]:
                images.append(os.path.join(t[0], file))
        return images

    def load_model(self):
        """
        Method to load the saved weights of the model
        :return: The loaded model
        """
        pass

    def build_model(self):
        """
        Method to build the model
        :return: The build model
        """
        # TODO
        self.conv1_1 = Conv2D(input_shape=(3, None, None), name="conv1_1")
        self.conv1_2 = Conv2D(self.conv1_1, "conv1_2")
        self.pool1 = MaxPooling2D(self.conv1_2, 'pool1')

        self.conv2_1 = Conv2D(self.pool1, "conv2_1")
        self.conv2_2 = Conv2D(self.conv2_1, "conv2_2")
        self.pool2 = MaxPooling2D(self.conv2_2, 'pool2')

        self.conv3_1 = Conv2D(self.pool2, "conv3_1")
        self.conv3_2 = Conv2D(self.conv3_1, "conv3_2")
        self.conv3_3 = Conv2D(self.conv3_2, "conv3_3")
        self.pool3 = MaxPooling2D(self.conv3_3, 'pool3')

        self.conv4_1 = Conv2D(self.pool3, "conv4_1")
        self.conv4_2 = Conv2D(self.conv4_1, "conv4_2")
        self.conv4_3 = Conv2D(self.conv4_2, "conv4_3")
        self.pool4 = MaxPooling2D(self.conv4_3, 'pool4')

        self.conv5_1 = Conv2D(self.pool4, "conv5_1")
        self.conv5_2 = Conv2D(self.conv5_1, "conv5_2")
        self.conv5_3 = Conv2D(self.conv5_2, "conv5_3")
        self.pool5 = MaxPooling2D(self.conv5_3, 'pool5')

        self.conv_6 = Conv2D(self.pool5, kernel_size=1, name="conv_6")
        self.conv_7 = Conv2D(self.conv_6, kernel_size=1, name="conv_7")

        # Adding 1x1 Convolution with num_classes channels
        self.class_pred = Conv2D(self.conv_7, filters=4, kernel_size=1, name="class_pred_layer")
        # Add Upsampling
        self.up_1 = Conv2DTranspose(self.class_pred, filters=self.conv_7.get_shape().as_list()[-1], kernel_size=4,
                                    strides=(2, 2), padding="SAME", name="upsampling_1")

        fcn9 = tf.layers.conv2d_transpose(self.class_pred, filters=self.conv_7.get_shape().as_list()[-1],
                                          kernel_size=4, strides=(2, 2), padding='SAME', name="fcn9")
        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")
        pass

    def fit_model(self):
        for e in range(0, self.epochs):
            for b in range(0, self.batches):
                pass
        pass

    def predict(self, image):
        pass

    @staticmethod
    def loss_function():
        pass


