import tensorflow as tf

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AvgPool2D
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

def Conv2D_BN(x, filters, num_row, num_column, padding='same', strides=(1,1)):
    x = Conv2D(
        kernel_size=(num_row, num_column),
        filters=filters,
        strides=strides,
        padding=padding
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def Stem(input):
    x = Conv2D_BN(input, 32, 3, 3, strides=(2,2),padding='valid')
    x = Conv2D_BN(x, 32, 3, 3,padding='valid')
    x = Conv2D_BN(x, 64, 3, 3)

    branch_0 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid')(x)
    
    branch_1 = Conv2D_BN(x, 95, 3, 3, strides=(2,2), padding='valid')
    
    x = concatenate([branch_0, branch_1])
    
    branch_0 = Conv2D_BN(x, 64, 1, 1)
    branch_0 = Conv2D_BN(branch_0, 95, 3, 3, padding='valid')
    
    branch_1 = Conv2D_BN(x, 64, 1, 1)
    branch_1 = Conv2D_BN(branch_1, 64, 7, 1)
    branch_1 = Conv2D_BN(branch_1, 64, 1, 7)
    branch_1 = Conv2D_BN(branch_1, 96, 3, 3, padding='valid')
    
    x = concatenate([branch_0, branch_1])
    
    branch_0 = Conv2D_BN(x, 192, 3, 3, strides=(2,2), padding='valid')
    
    branch_1 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid')(x)
    
    x = concatenate([branch_0, branch_1])
    
    return x

def Inception_A(input):
    branch_0 = AvgPool2D(pool_size=(3,3), strides=(1,1), padding='same')(input)
    branch_0 = Conv2D_BN(branch_0, 96, 1, 1)
    
    branch_1 = Conv2D_BN(input, 96, 1, 1)
    
    branch_2 = Conv2D_BN(input, 64, 1, 1)
    branch_2 = Conv2D_BN(branch_2, 96, 3, 3)
    
    branch_3 = Conv2D_BN(input, 64, 1, 1)
    branch_3 = Conv2D_BN(branch_3, 96, 3, 3)
    branch_3 = Conv2D_BN(branch_3, 96, 3, 3)
    
    x = concatenate([branch_0, branch_1, branch_2, branch_3])
    
    return x

def Inception_B(input):
    branch_0 = AvgPool2D(pool_size=(3,3), strides=(1,1), padding='same')(input)
    branch_0 = Conv2D_BN(branch_0, 128, 1, 1)
    
    branch_1 = Conv2D_BN(input, 384, 1, 1)
    
    branch_2 = Conv2D_BN(input, 192, 1, 1)
    branch_2 = Conv2D_BN(branch_2, 224, 1, 7)
    branch_2 = Conv2D_BN(branch_2, 256, 7, 1)
    
    branch_3 = Conv2D_BN(input, 192, 1, 1)
    branch_3 = Conv2D_BN(branch_3, 192, 1, 7)
    branch_3 = Conv2D_BN(branch_3, 224, 7, 1)
    branch_3 = Conv2D_BN(branch_3, 224, 1, 7)
    branch_3 = Conv2D_BN(branch_3, 256, 7, 1)
    
    x = concatenate([branch_0, branch_1, branch_2, branch_3])
    
    return x

def Inception_C(input):
    branch_0 = AvgPool2D(pool_size=(3,3), strides=(1,1), padding='same')(input)
    branch_0 = Conv2D_BN(branch_0, 256, 1, 1)
    
    branch_1 = Conv2D_BN(input, 256, 1, 1)
    
    branch_2 = Conv2D_BN(input, 384, 1, 1)
    branch_2_0 = Conv2D_BN(branch_2, 256, 1, 3)
    branch_2_1 = Conv2D_BN(branch_2, 256, 3, 1)
    
    branch_3 = Conv2D_BN(input, 384, 1, 1)
    branch_3 = Conv2D_BN(branch_3, 448, 1, 3)
    branch_3 = Conv2D_BN(branch_3, 512, 3, 1)
    branch_3_0 = Conv2D_BN(branch_3, 256, 3, 1)
    branch_3_1 = Conv2D_BN(branch_3, 256, 1, 3)
    
    x = concatenate([branch_0, branch_1, branch_2_0, branch_2_1, branch_3_0, branch_3_1])
    
    return x

def Reduction_A(input):
    branch_0 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid')(input)
    
    branch_1 = Conv2D_BN(input, 384, 3, 3, strides=(2,2), padding='valid')
    
    branch_2 = Conv2D_BN(input, 192, 1, 1)
    branch_2 = Conv2D_BN(branch_2, 224, 3, 3)
    branch_2 = Conv2D_BN(branch_2, 256, 3, 3, strides=(2,2), padding='valid')
    
    x = concatenate([branch_0, branch_1, branch_2])
    
    return x

def Reduction_B(input):
    branch_0 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid')(input)
    
    branch_1 = Conv2D_BN(input, 192, 1, 1)
    branch_1 = Conv2D_BN(branch_1, 192, 3, 3, strides=(2,2), padding='valid')
    
    branch_2 = Conv2D_BN(input, 256, 1, 1)
    branch_2 = Conv2D_BN(branch_2, 256, 1, 7)
    branch_2 = Conv2D_BN(branch_2, 320, 7, 1)
    branch_2 = Conv2D_BN(branch_2, 320, 3, 3, strides=(2,2), padding='valid')
    
    x = concatenate([branch_0, branch_1, branch_2])
    
    return x

def Inception_V4(input_shape, classes=1000):
    input = tf.keras.Input(shape=input_shape, name='image_input')

    x = Stem(input)
    
    for _ in range(4):
        x = Inception_A(x)
        
    x = Reduction_A(x)
    
    for _ in range(7):
        x = Inception_B(x)
        
    x = Reduction_B(x)
    
    for _ in range(3):
        x = Inception_C(x)
        
    x = GlobalAvgPool2D()(x)
    
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    
    output = Dense(units=classes, activation='softmax', name='output')(x)
    
    model = tf.keras.Model(input, output)
    
    return model

if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model

    input_shape = (500, 500, 1)
    model = Inception_V4(input_shape)

    plot_model(model)