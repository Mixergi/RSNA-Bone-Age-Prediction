import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAvgPool2D

from resnet_block import Stem, Inception_A, Inception_B, Inception_C, Reduction_A, Reduction_B


class Inception_Resnet_v2(tf.keras.Model):
    def __init__(self, classes, scale=0.1):
        super(Inception_Resnet_v2, self).__init__()

        self.classes = classes
        self.scale = scale

        self.stem = Stem()

        self.A_blocks = []
        for _ in range(5):
            self.A_blocks.append(Inception_A(self.scale))

        self.B_blocks = []
        for _ in range(10):
            self.B_blocks.append(Inception_B(self.scale))

        self.C_blocks = []
        for _ in range(5):
            self.C_blocks.append(Inception_C(self.scale))

        self.reduction_a = Reduction_A()
        self.reduction_b = Reduction_B()

        self.globalavg = GlobalAvgPool2D()

        self.dropout = Dropout(rate=0.8)
        self.classification = Dense(units=self.classes, activation='softmax')

    def call(self, inputs, training=None):
        if training:
            return self.train(inputs, training)
        else:
            return self.interface(inputs, training)

    def train(self, inputs, training=None):
        x = self.stem(inputs, training=training)
        for layer in self.A_blocks:
            x = layer(x, training=training)
        x = self.reduction_a(x, training=training)
        for layer in self.B_blocks:
            x = layer(x, training=training)
        x = self.reduction_b(x, training=training)
        for layer in self.C_blocks:
            x = layer(x, training=training)

        x = self.globalavg(x)
        x = self.dropout(x, training=training)
        x = self.classification(x)

        return x

    def interface(self, inputs, training=None):

        x = self.stem(inputs, training=training)
        for layer in self.A_blocks:
            x = layer(x, training=training)
        x = self.reduction_a(x, training=training)
        for layer in self.B_blocks:
            x = layer(x, training=training)
        x = self.reduction_b(x, training=training)
        for layer in self.C_blocks:
            x = layer(x, training=training)

        x = self.globalavg(x)
        x = self.dropout(x, training=training)
        x = self.classification(x)

        return x

if __name__ == "__main__":
    Inception_Resnet_v2(10)