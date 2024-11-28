from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
(XTrain, YTrain), (XTest, YTest) = mnist.load_data()

# Print the class types of the datasets
print('XTrain class = ', type(XTrain))
print('YTrain class = ', type(YTrain))

# Shape of the datasets
print('XTrain shape = ', XTrain.shape)
print('XTest shape = ', XTest.shape)
print('YTrain shape = ', YTrain.shape)
print('YTest shape = ', YTest.shape)

# Number of distinct values in the MNIST target
print('YTrain values = ', np.unique(YTrain))
print('YTest values = ', np.unique(YTest))

# Distribution of classes in the dataset
unique, counts = np.unique(YTrain, return_counts=True)
print('YTrain distribution = ', dict(zip(unique, counts)))
unique, counts = np.unique(YTest, return_counts=True)
print('YTest distribution = ', dict(zip(unique, counts)))

# Flatten the images for input to the autoencoder
XTrain = XTrain.astype('float32') / 255.0
XTest = XTest.astype('float32') / 255.0
XTrain = XTrain.reshape((XTrain.shape[0], -1))  # Flatten the images
XTest = XTest.reshape((XTest.shape[0], -1))  # Flatten the images

# Build the autoencoder model
InputModel = Input(shape=(784,))
EncodedLayer = Dense(32, activation='relu')(InputModel)
DecodedLayer = Dense(784, activation='sigmoid')(EncodedLayer)  # Fix the line here

# Create the autoencoder model
AutoencoderModel = Model(inputs=InputModel, outputs=DecodedLayer)

# Compile the autoencoder model
AutoencoderModel.compile(optimizer='adadelta', loss='binary_crossentropy')

# Train the model
history = AutoencoderModel.fit(
    XTrain, XTrain,  # Using XTrain as both input and target (autoencoder)
    batch_size=256,
    epochs=100,
    shuffle=True,
    validation_data=(XTest, XTest)
)

# Make predictions to decode the digits
DecodedDigits = AutoencoderModel.predict(XTest)

# Function to plot model history
def plot_model_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Autoencoder Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

# Plot the training and validation loss
plot_model_history(history)

# Visualization of input and decoded images
n = 5
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(XTest[i + 10].reshape(28, 28))  # Reshaping the flattened images to 28x28
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Display reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(DecodedDigits[i + 10].reshape(28, 28))  # Reshaping the decoded images to 28x28
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
