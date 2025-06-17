import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def build_dual_stream_cnn(input_shape=(64, 64, 3), dropout_rate=0.2, num_classes=2):
    # Stream 1
    input1 = Input(shape=input_shape)
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(dropout_rate)(x1)
    x1 = Flatten()(x1)

    # Stream 2
    input2 = Input(shape=input_shape)
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(input2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Dropout(dropout_rate)(x2)
    x2 = Flatten()(x2)

    # Concatenate streams
    concatenated = Concatenate()([x1, x2])
    output = Dense(num_classes, activation='softmax')(concatenated)

    model = Model(inputs=[input1, input2], outputs=output)
    return model

if __name__ == "__main__":
    model = build_dual_stream_cnn()
    model.summary()