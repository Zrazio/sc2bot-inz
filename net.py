from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Activation, Conv2D, Dropout, BatchNormalization, ReLU, Flatten
from keras.activations import elu, relu

from keras.activations import relu, softmax
### MODEL HYPERPARAMETERS
state_size = [96,88,3]      # Our input is a stack of 3 frames
action_size = 11


class DQNetwork:
    def __init__(self):
        self.action_shape = action_size
        self.state_shape = state_size
        self.model = Sequential()
        self.model.add(Conv2D(16, (8, 8), strides=(4, 4), padding="VALID", input_shape= self.state_shape))
        self.model.add(ReLU())
        self.model.add(Conv2D(32, (4,  4), padding="VALID", strides=(2,2)))
        self.model.add(ReLU())
        self.model.add(Conv2D(32, (4,  4), padding="VALID", strides=(2,2)))
        self.model.add(Flatten())
        self.model.add(Dense(256,activation=relu))
        self.model.add(Dense(self.action_shape, activation=relu))

    def get_model(self):
        return self.model

    def get_shapes(self):
        for layer in self.model.layers:
            print(layer.output_shape)



if __name__ == "__main__":
    mod = DQNetwork()
    mod.get_shapes()