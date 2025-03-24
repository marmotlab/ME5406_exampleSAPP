from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Add, Activation, Reshape, LSTM
from keras.utils import plot_model

RNN_SIZE  = 128
GOAL_REPR = 12
a_size = 5

def build_primal(img_path='network_image.png'):
    input_channels = Input(shape=(10, 10, 3))
    input_goal     = Input(shape=(3))

    conv1          = Conv2D(RNN_SIZE//4, (3,3), padding="same", strides=(1,1), activation='relu')(input_channels)
    conv1a         = Conv2D(RNN_SIZE//4, (3,3), padding="same", strides=(1,1), activation='relu')(conv1)
    conv1b         = Conv2D(RNN_SIZE//4, (3,3), padding="same", strides=(1,1), activation='relu')(conv1a)
    pool1          = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(conv1b)

    conv2          = Conv2D(RNN_SIZE//2, (3,3), padding="same", strides=(1,1), activation='relu')(pool1)
    conv2a         = Conv2D(RNN_SIZE//2, (3,3), padding="same", strides=(1,1), activation='relu')(conv2)
    conv2b         = Conv2D(RNN_SIZE//2, (3,3), padding="same", strides=(1,1), activation='relu')(conv2a)
    pool2          = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(conv2b)

    conv3          = Conv2D(RNN_SIZE-GOAL_REPR, (2,2), padding="valid", strides=(1,1), activation=None)(pool2)
    flat           = Flatten()(conv3)

    goal_layer     = Dense(GOAL_REPR, activation='relu')(input_goal)

    hidden_input   = Concatenate(axis=1)([flat, goal_layer])

    h1             = Dense(RNN_SIZE)(hidden_input)
    h2             = Dense(RNN_SIZE, activation=None)(h1)
    hidden_sum     = Add()([h2, hidden_input])
    h3             = Activation('relu')(hidden_sum)
    h3_reshaped    = Reshape((1, RNN_SIZE))(h3)

    rnn_out        = LSTM(RNN_SIZE)(h3_reshaped)

    policy         = Dense(a_size, activation='softmax')(rnn_out)
    value          = Dense(1, activation=None)(rnn_out) 

    model = Model(inputs = [input_channels, input_goal], outputs = [policy, value], name = 'ME5406_SAPP')
#    plot_model(model, to_file=img_path)

    return model


model = build_primal()
print(model.summary())
