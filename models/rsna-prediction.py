import tensorflow as tf

from tensorflow.keras.layers import concatenate, Dense, Dropout

from Inception_V4 import Inception_V4


def Rsna_Prediction_Model(input_shape):

    inception_v4 = Inception_V4(input_shape)

    image_input = inception_v4.input
    inception_output = inception_v4.layers[-3].output

    gender_input = tf.keras.Input(shape=(1), name='gender_input')

    gender_branch = Dense(units=32, activation='relu')(gender_input)

    x = concatenate([inception_output, gender_branch])

    x = Dropout(rate=0.5)(x)

    x = Dense(units=1024, activation='relu')(x)

    x = Dropout(rate=0.5)(x)

    x = Dense(units=1024, activation='relu')(x)

    output = Dense(1)(x)

    model = tf.keras.Model(inputs=[image_input, gender_input], outputs=output)

    return model
