import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class UNet(Model):
    def __init__(self, num_classes):
        super().__init__()

        # Encoder blocks
        self.conv1 = self.conv_block(64)
        self.conv2 = self.conv_block(128)
        self.conv3 = self.conv_block(256)
        self.conv4 = self.conv_block(512)

        # Bottleneck
        self.bottleneck = self.conv_block(1024)

        # Decoder blocks
        self.upconv4 = self.upconv_block(512)
        self.upconv3 = self.upconv_block(256)
        self.upconv2 = self.upconv_block(128)
        self.upconv1 = self.upconv_block(64)

        # Output layer
        self.output_layer = layers.Conv2D(num_classes, kernel_size=1, activation="sigmoid")

    def conv_block(self, filters):
        """Returns a convolutional block with two conv layers followed by batch normalisation and ReLU."""
        return tf.keras.Sequential([
            layers.Conv2D(filters, kernel_size=3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters, kernel_size=3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def upconv_block(self, filters):
        """Returns an upsampling block with a transposed convolution followed by a conv block."""
        return tf.keras.Sequential([
            layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same"),  # learns the upsampling process
            
            layers.Conv2D(filters, kernel_size=3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters, kernel_size=3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, inputs):     # overriding the forward pass of the network
        # Encoder path
        conv1_output = self.conv1(inputs)
        pool1_output = layers.MaxPooling2D(pool_size=2)(conv1_output)

        conv2_output = self.conv2(pool1_output)
        pool2_output = layers.MaxPooling2D(pool_size=2)(conv2_output)

        conv3_output = self.conv3(pool2_output)
        pool3_output = layers.MaxPooling2D(pool_size=2)(conv3_output)

        conv4_output = self.conv4(pool3_output)
        pool4_output = layers.MaxPooling2D(pool_size=2)(conv4_output)

        # Bottleneck
        b = self.bottleneck(pool4_output)

        # Decoder path with skip connections
        upconv4_output = self.upconv4(b)
        upconv4_output = layers.Concatenate()([upconv4_output, conv4_output])

        upconv3_output = self.upconv3(upconv4_output)
        upconv3_output = layers.Concatenate()([upconv3_output, conv3_output])

        upconv2_output = self.upconv2(upconv3_output)
        upconv2_output = layers.Concatenate()([upconv2_output, conv2_output])

        upconv1_output = self.upconv1(upconv2_output)
        upconv1_output = layers.Concatenate()([upconv1_output, conv1_output])

        # Output layer
        outputs = self.output_layer(upconv1_output)

        return outputs

# Example usage
if __name__ == "__main__":
    # Create model instance
    model = UNet(num_classes=1)
    
    # define the input shape
    model.build(input_shape=(None, 1024, 1024, 5))

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Define dummy data
    x_train = np.random.rand(10, 1024, 1024, 5)
    y_train = np.random.rand(10, 1024, 1024, 1)

    # Train the model
    model.fit(x_train, y_train, epochs=2)

    # Print model summary
    model.summary()