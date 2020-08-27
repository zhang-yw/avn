#structure adopted from "Cognitive Mapping and Planning for Visual Navigation"
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, Conv2DTranspose, Reshape


def get_mapper(input_shape):
    print("input_shape: " + str(input_shape))
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(7, 7), activation='relu', input_shape=input_shape))
    model.add(Conv2D(48, kernel_size=(7, 7), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', input_shape=input_shape))

    model.add(Dense(200, activation='relu'))

    model.add(Conv2DTranspose(64, kernel_size=(7, 7), activation='relu', input_shape=input_shape))
    model.add(Conv2DTranspose(32, kernel_size=(7, 7), activation='relu', input_shape=input_shape))
    model.add(Conv2DTranspose(1, kernel_size=(7, 7), activation='sigmoid', input_shape=input_shape))
    
    model.add(Flatten())
    model.add(Dropout(0.25))

    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(1000, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(25, activation='sigmoid'))

    model.add(Reshape((5, 5)))

    model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adadelta())

    return model

def update_belief(pre_belief, new_belief):
    return 1.0/(1.0+((1.0-pre_belief)/pre_belief)*((1.0-new_belief)/new_belief))
