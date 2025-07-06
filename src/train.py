# This file trains the model, predicts them and generates AUC and ROC curve

# USAGE
# python train.py --dataset1 dataset_old/cells/Q6 --dataset2 dataset_old/cells/Q6_new1 --model1 output/lenet_new6.1.1.hdf5 --model2 output/lenet_new6.1.1.hdf5
# --model_json1 output_to_json/model_new6.1.1.json --model_json2 output_to_json/model_new6.1.1.json --dataset_test1 dataset_old/cells/Q6_test --dataset_test2 dataset_old/cells/Q6_test
# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from pyimagesearch.nn.conv.lenet import LeNet
from model import LeNetCustom
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2 as cv
import os
import PIL
from keras.models import model_from_json
from keras import backend as K



os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d1", "--dataset", required=True,
                help="path to input dataset of faces")
ap.add_argument("-m1", "--model", required=True,
                help="path to output model")
ap.add_argument("-mj1", "--model_json", required=True,
                help="path to output model to json")
ap.add_argument("-dt1", "--dataset_test", required=True,
                help="path to input dataset of faces")

args = vars(ap.parse_args())

# initialize the list of data and labels
data1 = []
labels1 = []


# loop over the input images 
for imagePath in sorted(list(paths.list_images(args["dataset1"]))):
    # load the image, pre-process it, and store it in the data list


    # Read in tiff
    pil_image = PIL.Image.open(imagePath).convert('RGB')
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    image = cv.cvtColor(open_cv_image, cv.COLOR_BGR2GRAY)

    # image = imutils.resize(image, width=28)
    # image = imutils.resize(image, width=64)   # change between this line and the one below if input is 64 vs 128
    image = imutils.resize(image, width=128)
    image = img_to_array(image)
    data1.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    label = "healthy" if label == "healthy" else "unhealthy"
    labels1.append(label)


# scale the raw pixel intensities to the range [0, 1]
data1 = np.array(data1, dtype="float") / 255.0
labels1 = np.array(labels1)


# convert the labels from integers to vectors
le1 = LabelEncoder().fit(labels1)
labels1 = np_utils.to_categorical(le1.transform(labels1), 2)



# account for skew in the labeled data
classTotals1 = labels1.sum(axis=0)
classWeight1 = classTotals1.max() / classTotals1


# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX1, testX1, trainY1, testY1) = train_test_split(data1,
                                                      labels1, test_size=0.20, stratify=labels1, random_state=42)


# initialize the model 
print("[INFO] compiling model ...")
# model = LeNet.build(width=28, height=28, depth=1, classes=2)
# model = LeNet.build(width=64, height=64, depth=1, classes=2)

model1 = LeNetCustom.build(width=128, height=128, depth=1, classes=2)
model1.compile(loss="binary_crossentropy", optimizer="adam",
               metrics=["accuracy"])


# train the model 
print("[INFO] training network 1...")
H1 = model1.fit(trainX1, trainY1, validation_data=(testX1, testY1),
                class_weight=classWeight1, batch_size=64, epochs=50, verbose=1)



# evaluate the model

print("[INFO] evaluating network...")
predictions1 = model1.predict(testX1, batch_size=64)
print(classification_report(testY1.argmax(axis=1),
                            predictions1.argmax(axis=1), target_names=le1.classes_))


# save the model to disk 1
print("[INFO] serializing network ...")
model1.save(args["model"])

model_json1 = model1.to_json()
with open(args["model_json"], 'w') as json_file:
    json_file.write(model_json1)

#####
import xlsxwriter

workbook1 = xlsxwriter.Workbook('output_xls/Q_new6_4l.xlsx')
worksheet1 = workbook1.add_worksheet()
worksheet1.write(0, 0, "Accuracy")
worksheet1.write(0, 1, "Val_accuracy")
worksheet1.write(0, 2, "Loss")
worksheet1.write(0, 3, "Val_loss")
row = 1
col = 0

for item in H1.history['accuracy']:
    worksheet1.write(row, col, item)
    row += 1
row = 1
col = 1
for item in H1.history['val_accuracy']:
    worksheet1.write(row, col, item)
    row += 1
row = 1
col = 2
for item in H1.history['loss']:
    worksheet1.write(row, col, item)
    row += 1
row = 1
col = 3
for item in H1.history['val_loss']:
    worksheet1.write(row, col, item)
    row += 1

workbook1.close()

# plot the training + testing loss and accuracy 
# plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), H1.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), H1.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), H1.history["accuracy"], label="accuracy")
plt.plot(np.arange(0, 50), H1.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy Model 1")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


