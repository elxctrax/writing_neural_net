import os
import keras
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist # importing mnist data for handwritten number references
(x_train, y_train), (x_test, y_test) = mnist.load_data() # train data

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

# save state of model
# saves state/data of the model into a binary txt file called "writing_neural_net.keras" 
model.save('writing_neural_net.keras')

# load model without having to train all over again
model = tf.keras.models.load_model('writing_neural_net.keras')

loss, accuracy = model.evaluate(x_test, y_test) # determine the loss and accuracy of the neural network

print(loss)
print(accuracy) # loss and accuracy for the overall model and not for every epoch

# iterate through every digit file in the digits folder. this is where the neural net shows the picture of the digit and what it thinks the digit is.
image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png") [:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error")
    finally:
        image_number += 1

# neural network got digits 4,6,9,0 wrong. we can add more epochs to get more accuracy

# to run this, just type python main.py into your terminal.