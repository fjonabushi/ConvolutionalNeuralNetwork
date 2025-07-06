# Making new predictions
import cv2
import numpy as np
from keras import Sequential
from keras.preprocessing import image
from keras.models import model_from_json
from keras.models import load_model
import os
from keras_preprocessing.image import img_to_array
from prettytable import PrettyTable

#loading model
json_file = open('model_cnn_1.14.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# model = Sequential()
# test_h5_model = model.load_weights('model_cnn_1.14.h5')



total = 0
correct = 0
t = PrettyTable(['Current', 'Predicted'])

for filename in os.listdir('./dataset/single_prediction/'):
    total = total + 1
    test_image = image.load_img('./dataset/single_prediction/' + filename, color_mode = "grayscale", target_size=(128,128,1))
    test_image = img_to_array(test_image)
    test_image = cv2.resize(test_image, (128, 128))
    norm_img = np.zeros((800, 800))
    test_image = cv2.normalize(test_image, norm_img, 0, 255, cv2.NORM_MINMAX)
    test_image = test_image.astype("float") / 255.0
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    np.set_printoptions(suppress=True)
    # determine the probability of both "healthy" and "cytotoxic",
    # then set the label accordingly
    (unhealthy, healthy) = loaded_model.predict_classes(test_image)[0]
    print(healthy)
    print(unhealthy)
    print(loaded_model.predict(test_image)[0])
    predicted_label = "healthy" if healthy > unhealthy else "unhealthy"
    current_label = "healthy" if filename.startswith("healthy") else "unhealthy"
    t.add_row([current_label, predicted_label])
    if (predicted_label == current_label):
        correct = correct + 1



print('Total: ', total, " Correct: ", correct, " Percentage: ", float(correct) / float(total))

