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
            layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same"),
            self.conv_block(filters)

            # maybe have the following instead calling self.con_block
            # layers.Conv2D(filters, kernel_size=3, padding="same"),
            # layers.BatchNormalization(),
            # layers.ReLU(),
            # layers.Conv2D(filters, kernel_size=3, padding="same"),
            # layers.BatchNormalization(),
            # layers.ReLU()
        ])

    def call(self, inputs):     # overriding the forward pass of the network
        # Encoder path
        c1 = self.conv1(inputs)
        p1 = layers.MaxPooling2D(pool_size=2)(c1)

        c2 = self.conv2(p1)
        p2 = layers.MaxPooling2D(pool_size=2)(c2)

        c3 = self.conv3(p2)
        p3 = layers.MaxPooling2D(pool_size=2)(c3)

        c4 = self.conv4(p3)
        p4 = layers.MaxPooling2D(pool_size=2)(c4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder path with skip connections
        u4 = self.upconv4(b)
        u4 = layers.Concatenate()([u4, c4])

        u3 = self.upconv3(u4)
        u3 = layers.Concatenate()([u3, c3])

        u2 = self.upconv2(u3)
        u2 = layers.Concatenate()([u2, c2])

        u1 = self.upconv1(u2)
        u1 = layers.Concatenate()([u1, c1])

        # Output layer
        outputs = self.output_layer(u1)

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