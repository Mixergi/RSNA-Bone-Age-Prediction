import tensorflow as tf
from tensorflow.keras.layers import (Activation, Add, AvgPool2D, BatchNormalization, Concatenate,
                                     Conv2D, Dropout, MaxPool2D, Lambda)


class Conv2DBN(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='same', activation='relu'):
        self.conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)
        self.bn = BatchNormalization()
        self.activation = Activation(activation)
        self.dropout = Dropout(rate=0.5)
        super(Conv2DBN, self).__init__()

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        return x


class ScalingResiual(tf.keras.layers.Layer):
    def __init__(self, scale, activation='relu'):
        self.scale = scale
        self.lambda_layer = Lambda(lambda Inception, scale: Inception * scale, arguments={'scale': self.scale})
        self.activation = Activation(activation)

        super(ScalingResiual, self).__init__()

    def call(self, inputs):
        x = self.lambda_layer(inputs)
        x = self.activation(inputs)

        return x


class Stem(tf.keras.layers.Layer):
    def __init__(self):
        self.stem_concat_1 = Concatenate()
        self.stem_concat_2 = Concatenate()
        self.stem_concat_3 = Concatenate()

        self.stem_conv_1 = Conv2DBN(32, (3, 3), (2, 2), 'valid')
        self.stem_conv_2 = Conv2DBN(32, (3, 3), padding='valid')
        self.stem_conv_3 = Conv2DBN(64, (3, 3))

        self.branch1_1_max = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')
        self.branch1_2_conv = Conv2DBN(96, (3, 3), (2, 2), 'valid')

        self.branch2_1_conv_1 = Conv2DBN(64, (1, 1))
        self.branch2_1_conv_2 = Conv2DBN(96, (3, 3), padding='valid')

        self.branch2_2_conv_1 = Conv2DBN(64, (1, 1))
        self.branch2_2_conv_2 = Conv2DBN(64, (7, 1))
        self.branch2_2_conv_3 = Conv2DBN(64, (1, 7))
        self.branch2_2_conv_4 = Conv2DBN(96, (3, 3), padding='valid')

        self.branch3_1_conv = Conv2DBN(192, (3, 3), (2, 2), 'valid')
        self.branch3_2_max = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')

        super(Stem, self).__init__()

    def call(self, inputs, training=None):
        stem = self.stem_conv_1(inputs, training=training)
        stem = self.stem_conv_2(stem, training=training)
        stem = self.stem_conv_3(stem, training=training)

        branch_1 = self.branch1_1_max(stem)

        branch_2 = self.branch1_2_conv(stem, training=training)

        stem = self.stem_concat_1([branch_1, branch_2])

        branch_1 = self.branch2_1_conv_1(stem, training=training)
        branch_1 = self.branch2_1_conv_2(branch_1, training=training)

        branch_2 = self.branch2_2_conv_1(stem, training=training)
        branch_2 = self.branch2_2_conv_2(branch_2, training=training)
        branch_2 = self.branch2_2_conv_3(branch_2, training=training)
        branch_2 = self.branch2_2_conv_4(branch_2, training=training)

        stem = self.stem_concat_2([branch_1, branch_2])

        branch_1 = self.branch3_1_conv(stem, training=training)

        branch_2 = self.branch3_2_max(stem, training=training)

        stem = self.stem_concat_3([branch_1, branch_2])

        return stem


class Inception_A(tf.keras.layers.Layer):
    def __init__(self, scale=0.1):
        self.scale = 0.1

        self.branch1_conv = Conv2DBN(32, (1, 1))

        self.branch2_conv_1 = Conv2DBN(32, (1, 1))
        self.branch2_conv_2 = Conv2DBN(32, (1, 1))

        self.branch3_conv_1 = Conv2DBN(32, (1, 1))
        self.branch3_conv_2 = Conv2DBN(48, (3, 3))
        self.branch3_conv_3 = Conv2DBN(64, (3, 3))

        self.branches_concat = Concatenate()
        self.branches_conv = Conv2DBN(384, (1, 1), activation=None)

        self.scaling_residual = ScalingResiual(self.scale)
        self.connect = Add()

        super(Inception_A, self).__init__()

    def call(self, inputs, training=None):
        branch_1 = self.branch1_conv(inputs, training=training)

        branch_2 = self.branch2_conv_1(inputs, training=training)
        branch_2 = self.branch2_conv_2(branch_2, training=training)

        branch_3 = self.branch3_conv_1(inputs, training=training)
        branch_3 = self.branch3_conv_2(branch_3, training=training)
        branch_3 = self.branch3_conv_3(branch_3, training=training)

        branches = self.branches_concat([branch_1, branch_2, branch_3])
        branches = self.branches_conv(branches, training=training)
        branches = self.scaling_residual(branches)

        stem = self.connect([inputs, branches])

        return stem


class Inception_B(tf.keras.layers.Layer):
    def __init__(self, scale=0.1):
        self.scale = scale
        self.branch1_conv = Conv2DBN(192, (1, 1))

        self.branch2_conv_1 = Conv2DBN(128, (1, 1))
        self.branch2_conv_2 = Conv2DBN(160, (1, 7))
        self.branch2_conv_3 = Conv2DBN(192, (7, 1))

        self.branches_concat = Concatenate()
        self.branches_conv = Conv2DBN(1152, (1, 1), activation=None)

        self.scaling_residual = ScalingResiual(self.scale)
        self.connect = Add()

        super(Inception_B, self).__init__()

    def call(self, inputs, training=None):
        branch_1 = self.branch1_conv(inputs, training=training)

        branch_2 = self.branch2_conv_1(inputs, training=training)
        branch_2 = self.branch2_conv_2(branch_2, training=training)
        branch_2 = self.branch2_conv_3(branch_2, training=training)

        branches = self.branches_concat([branch_1, branch_2])
        branches = self.branches_conv(branches, training=training)
        branches = self.scaling_residual(branches)

        stem = self.connect([inputs, branches])

        return stem


class Inception_C(tf.keras.layers.Layer):
    def __init__(self, scale=0.1):
        self.scale = scale
        self.branch1_conv = Conv2DBN(192, (1, 1))

        self.branch2_conv_1 = Conv2DBN(192, (1, 1))
        self.branch2_conv_2 = Conv2DBN(224, (1, 3))
        self.branch2_conv_3 = Conv2DBN(256, (3, 1))

        self.branches_concat = Concatenate()
        self.branches_conv = Conv2DBN(2144, (1, 1))

        self.scaling_residual = ScalingResiual(self.scale)
        self.connect = Add()

        super(Inception_C, self).__init__()

    def call(self, inputs, training=None):
        branch_1 = self.branch1_conv(inputs, training=training)

        branch_2 = self.branch2_conv_1(inputs, training=training)
        branch_2 = self.branch2_conv_2(branch_2, training=training)
        branch_2 = self.branch2_conv_3(branch_2, training=training)

        branches = self.branches_concat([branch_1, branch_2])
        branches = self.branches_conv(branches, training=training)
        branches = self.scaling_residual(branches)

        stem = self.connect([inputs, branches])

        return stem


class Reduction_A(tf.keras.layers.Layer):
    def __init__(self):
        self.branch1_max = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')

        self.branch2_conv = Conv2DBN(384, (3, 3), (2, 2), 'valid')

        self.branch3_conv_1 = Conv2DBN(256, (1, 1))
        self.branch3_conv_2 = Conv2DBN(256, (3, 3))
        self.branch3_conv_3 = Conv2DBN(384, (3, 3), (2, 2), 'valid')

        self.concat = Concatenate()

        super(Reduction_A, self).__init__()

    def call(self, inputs, training=None):
        branch_1 = self.branch1_max(inputs)

        branch_2 = self.branch2_conv(inputs, training=training)

        branch_3 = self.branch3_conv_1(inputs, training=training)
        branch_3 = self.branch3_conv_2(branch_3, training=training)
        branch_3 = self.branch3_conv_3(branch_3, training=training)

        x = self.concat([branch_1, branch_2, branch_3])

        return x


class Reduction_B(tf.keras.layers.Layer):
    def __init__(self):
        self.branch1_max = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')

        self.branch2_conv_1 = Conv2DBN(256, (1, 1))
        self.branch2_conv_2 = Conv2DBN(384, (3, 3), (2, 2), 'valid')

        self.branch3_conv_1 = Conv2DBN(256, (1, 1))
        self.branch3_conv_2 = Conv2DBN(288, (3, 3), (2, 2), 'valid')

        self.branch4_conv_1 = Conv2DBN(256, (1, 1))
        self.branch4_conv_2 = Conv2DBN(288, (3, 3))
        self.branch4_conv_3 = Conv2DBN(320, (3, 3), (2, 2), 'valid')

        self.concat = Concatenate()

        super(Reduction_B, self).__init__()

    def call(self, inputs, training=None):
        branch_1 = self.branch1_max(inputs)

        branch_2 = self.branch2_conv_1(inputs, training=training)
        branch_2 = self.branch2_conv_2(branch_2, training=training)

        branch_3 = self.branch3_conv_1(inputs, training=training)
        branch_3 = self.branch3_conv_2(branch_3, training=training)

        branch_4 = self.branch4_conv_1(inputs, training=training)
        branch_4 = self.branch4_conv_2(branch_4, training=training)
        branch_4 = self.branch4_conv_3(branch_4, training=training)

        x = self.concat([branch_1, branch_2, branch_3, branch_4])

        return x