import tflearn
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset.image_operations import imageOperations
from model import conv_neural_network, run


#reading training data
train_reader = imageOperations('dataset/data/train/*.jpg')
train_data, labels = train_reader.create_train_data()

#running Convolution neural network
model = conv_neural_network()
model = run(train_data, labels, model)

#reading testing data
test_reader = imageOperations('dataset/data/test1/*.jpg')
test_data = test_reader.create_test_data()


#plotting result
fig = plt.figure()
for cnt, img in enumerate(test_data[:12]):
    y = fig.add_subplot(3, 4, cnt+1)
    orig = img
    img = img.reshape(50, 50, 1)
    model_out = model.predict([img])[0]

    if np.argmax(model_out) == 1: label = 'Cat'
    else: label = 'Dog'

    y.imshow(orig, cmap='gray')
    plt.title(label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
