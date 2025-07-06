############# PREDICTION


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset_test1", required=True,
# help="path to input dataset of faces")
# ap.add_argument("-d", "--dataset_test2", required=True,
# help="path to input dataset of faces")
# args = vars(ap.parse_args())


# initialize the list of data and labels
data1 = []
labels1 = []
a1 = 0
data2 = []
labels2 = []
a2 = 0

for imagePath in sorted(list(paths.list_images(args["dataset_test1"]))):
    # load the image, pre-process it, and store it in the data list

    # Read PNG
    # image = cv.imread(imagePath)
    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Read in tiff
    pil_image = PIL.Image.open(imagePath).convert('RGB')
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    image = cv.cvtColor(open_cv_image, cv.COLOR_BGR2GRAY)

    # image = imutils.resize(image, width=28)
    # image = imutils.resize(image, width=64)
    image = imutils.resize(image, width=128)
    image = img_to_array(image)
    data1.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    # label = "smiling" if label == "positives" else "not_smiling"
    label = "healthy" if label == "healthy" else "unhealthy"
    labels1.append(label)
    a1 += 1

for imagePath in sorted(list(paths.list_images(args["dataset_test2"]))):
    # load the image, pre-process it, and store it in the data list

    # Read PNG
    # image = cv.imread(imagePath)
    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Read in tiff
    pil_image = PIL.Image.open(imagePath).convert('RGB')
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    image = cv.cvtColor(open_cv_image, cv.COLOR_BGR2GRAY)

    # image = imutils.resize(image, width=28)
    # image = imutils.resize(image, width=64)
    image = imutils.resize(image, width=128)
    image = img_to_array(image)
    data2.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    # label = "smiling" if label == "positives" else "not_smiling"
    label = "healthy" if label == "healthy" else "unhealthy"
    labels2.append(label)
    a2 += 1

# scale the raw pixel intensities to the range [0, 1]
data1 = np.array(data1, dtype="float") / 255.0
labels1 = np.array(labels1)

# scale the raw pixel intensities to the range [0, 1]
data2 = np.array(data2, dtype="float") / 255.0
labels2 = np.array(labels2)

# convert the labels from integers to vectors
le1 = LabelEncoder().fit(labels1)
labels1 = np_utils.to_categorical(le1.transform(labels1), 2)

# convert the labels from integers to vectors
le2 = LabelEncoder().fit(labels2)
labels2 = np_utils.to_categorical(le2.transform(labels2), 2)

# account for skew in the labeled data
classTotals1 = labels1.sum(axis=0)
classWeight1 = classTotals1.max() / classTotals1

# account for skew in the labeled data
classTotals2 = labels2.sum(axis=0)
classWeight2 = classTotals2.max() / classTotals2

trainX1 = data1
trainY1 = labels1
# Load trained CNN model
# json_file = open('output_to_json/modelQ4_128x128_customLenet.json', 'r')
json_file1 = open('output_to_json/model_new6_4l.json', 'r')
loaded_model_json1 = json_file1.read()
json_file1.close()
model1 = model_from_json(loaded_model_json1)
# model.load_weights('output/lenetQ4_128x128_customLenet.hdf5')
model1.load_weights('output/lenet_new6_4l.hdf5')

trainX2 = data2
trainY2 = labels2
# Load trained CNN model
# json_file = open('output_to_json/modelQ4_128x128_customLenet.json', 'r')
json_file2 = open('output_to_json/model_new6_5l.json', 'r')
loaded_model_json2 = json_file2.read()
json_file2.close()
model2 = model_from_json(loaded_model_json2)
# model.load_weights('output/lenetQ4_128x128_customLenet.hdf5')
model2.load_weights('output/lenet_new6_5l.hdf5')

trainLabels1 = list(le1.inverse_transform(trainY1.argmax(1)))
size1 = len(trainLabels1)
predicted1 = 0
images1 = []
x1 = 0

trainLabels2 = list(le2.inverse_transform(trainY2.argmax(1)))
size2 = len(trainLabels2)
predicted2 = 0
images2 = []
x2 = 0

for i in np.random.choice(np.arange(0, len(trainY1)), size=(size1,)):

    probs1 = model1.predict(trainX1[np.newaxis, i])
    # print(probs)
    prediction1 = probs1.argmax(axis=1)
    label1 = le1.inverse_transform(prediction1)
    if label1[0] == trainLabels1[i]:
        predicted1 += 1

    # extract the image from the testData if using "channels_first"
    # ordering
    if K.image_data_format() == "channels_first":
        image1 = (trainX1[i][0] * 255).astype("uint8")

    # otherwise we are using "channels_last" ordering
    else:
        image1 = (trainX1[i] * 255).astype("uint8")

    # merge the channels into one image
    image1 = cv.merge([image1] * 3)

    image1 = cv.resize(image1, (128, 128), interpolation=cv.INTER_LINEAR)

    # show the image and prediction
    x1 += 1
    position1 = str(x1)
    text1 = position1 + ' ' + label1[0]
    cv.putText(image1, str(text1), (5, 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    print("[INFO]:{} Predicted1: {}, Actual1: {}".format(x1, label1[0],
                                                         trainLabels1[i]))
    images1.append(image1)

for i in np.random.choice(np.arange(0, len(trainY2)), size=(size2,)):

    probs2 = model2.predict(trainX2[np.newaxis, i])
    # print(probs)
    prediction2 = probs2.argmax(axis=1)
    label2 = le2.inverse_transform(prediction2)
    if label2[0] == trainLabels2[i]:
        predicted2 += 1

    # extract the image from the testData if using "channels_first"
    # ordering
    if K.image_data_format() == "channels_first":
        image2 = (trainX2[i][0] * 255).astype("uint8")

    # otherwise we are using "channels_last" ordering
    else:
        image2 = (trainX2[i] * 255).astype("uint8")

    # merge the channels into one image
    image2 = cv.merge([image2] * 3)

    image2 = cv.resize(image2, (128, 128), interpolation=cv.INTER_LINEAR)

    # show the image and prediction
    x2 += 1
    position2 = str(x2)
    text2 = position2 + ' ' + label2[0]
    cv.putText(image2, str(text2), (5, 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    print("[INFO]:{} Predicted2: {}, Actual2: {}".format(x2, label2[0],
                                                         trainLabels2[i]))
    images2.append(image2)

print('Accuracy1: ',
      predicted1 / size1)
# # img = cv.imwrite('images.png', images)
# images = np.concatenate(images, axis=1)
# cv.imshow("Cell", images)
# cv.waitKey(0)

print('Accuracy2: ',
      predicted2 / size2)
# # img = cv.imwrite('images.png', images)
# images = np.concatenate(images, axis=1)
# cv.imshow("Cell", images)
# cv.waitKey(0)


fig1 = plt.figure(figsize=(14, 14))
columns1 = 8
rows1 = 3
for i in range(0, columns1 * rows1):
    fig1.add_subplot(rows1, columns1, i + 1)
    plt.imshow(images1[i])
plt.show()

fig2 = plt.figure(figsize=(14, 14))
columns2 = 8
rows2 = 3
for i in range(0, columns2 * rows2):
    fig2.add_subplot(rows2, columns2, i + 1)
    plt.imshow(images2[i])
plt.show()

# AUC and ROC

# predict probabilities
pred_prob1 = model1.predict_proba(testX1)
pred_prob2 = model2.predict_proba(testX2)

# confusion matrix
from sklearn.metrics import confusion_matrix

confusion_matrix1 = confusion_matrix(testY1[:, 1].astype(int), (pred_prob1[:, 1]).round())
print('Confusion matrix 1:', confusion_matrix1)
confusion_matrix2 = confusion_matrix(testY2[:, 1].astype(int), (pred_prob2[:, 1]).round())
print('Confusion matrix 2:', confusion_matrix2)

# print(metrics.confusion_matrix(testY1[:,1].astype(int), pred_prob1[:,1])
# print(metrics.confusion_matrix(testY2[:,1].astype(int), pred_prob2[:,1])


from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(testY1[:, 1].astype(int), pred_prob1[:, 1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(testY2[:, 1].astype(int), pred_prob2[:, 1], pos_label=1)

# roc curve for tpr = fpr
random_probs = [0 for i in range(len(testY1[:, 1]))]
p_fpr, p_tpr, _ = roc_curve(testY1[:, 1].astype(int), random_probs, pos_label=1)

from sklearn.metrics import roc_auc_score

# auc scores
auc_score1 = roc_auc_score(testY1[:, 1].astype(int), pred_prob1[:, 1])
auc_score2 = roc_auc_score(testY2[:, 1].astype(int), pred_prob2[:, 1])

print('AUC1: ', auc_score1)
print('AUC2: ', auc_score2)

# matplotlib
import matplotlib.pyplot as plt

plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--', color='orange', label='Model Custom LeNet with 4 Layers ')
plt.plot(fpr2, tpr2, linestyle='--', color='green', label='Model Custom LeNet with 5 Layers ')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC', dpi=300)
plt.show();
train3_test_roc.py
# This file compares one 3 class model only.
# USAGE
# python trainn3_model.py --dataset dataset_old/cells/Q7 --model output/lenet_new7.1.hdf5 --model_json output_to_json/model_new7.1.json
# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from pyimagesearch.nn.conv.lenet import LeNet
from pyimagesearch.nn.conv.customLenet import LeNetCustom
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2 as cv
import os
import PIL

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset of faces")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-mj", "--model_json", required=True,
                help="path to output model to json")
ap.add_argument("-dt", "--dataset_test", required=True,
                help="path to input dataset of faces")
args = vars(ap.parse_args())

# initialize the list of data and labels
data = []
labels = []

# loop over the input images
for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    # load the image, pre-process it, and store it in the data list

    # Read PNG
    # image = cv.imread(imagePath)
    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Read in tiff
    pil_image = PIL.Image.open(imagePath).convert('RGB')
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    image = cv.cvtColor(open_cv_image, cv.COLOR_BGR2GRAY)

    # image = imutils.resize(image, width=28)
    # image = imutils.resize(image, width=64)   # change between this line and the one below if input is 64 vs 128
    image = imutils.resize(image, width=128)
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    # label = "smiling" if label == "positives" else "not_smiling"
    if label == "0":
        label = "0"
    elif label == "1":
        label = "1"
    else:
        label = "2"

    # label = "healthy" if label == "healthy" else "unhealthy"
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 3)

# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.20, stratify=labels, random_state=42)

# initialize the model
print("[INFO] compiling model...")
# model = LeNet.build(width=28, height=28, depth=1, classes=2)
# model = LeNet.build(width=64, height=64, depth=1, classes=2)

model = LeNetCustom.build(width=128, height=128, depth=1, classes=3)
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              class_weight=classWeight, batch_size=64, epochs=50, verbose=1)

# history = model.fit()

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=le.classes_))

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

model_json = model.to_json()
with open(args["model_json"], 'w') as json_file:
    json_file.write(model_json)

#####
import xlsxwriter

workbook = xlsxwriter.Workbook('output_xls/Q_new7.5.2.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, "Accuracy")
worksheet.write(0, 1, "Val_accuracy")
worksheet.write(0, 2, "Loss")
worksheet.write(0, 3, "Val_loss")
row = 1
col = 0

for item in H.history['accuracy']:
    worksheet.write(row, col, item)
    row += 1
row = 1
col = 1
for item in H.history['val_accuracy']:
    worksheet.write(row, col, item)
    row += 1
row = 1
col = 2
for item in H.history['loss']:
    worksheet.write(row, col, item)
    row += 1
row = 1
col = 3
for item in H.history['val_loss']:
    worksheet.write(row, col, item)
    row += 1

workbook.close()

#######

# plot the training + testing loss and accuracy
# plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), H.history["accuracy"], label="accuracy")
plt.plot(np.arange(0, 50), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

#### PREDICTION

# USAGE
# python RunCustomLeNetModel3.py --dataset dataset_old/cells/Q7_test
# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from pyimagesearch.nn.conv.lenet import LeNet
from pyimagesearch.nn.conv.customLenet import LeNetCustom
from imutils import paths
import matplotlib.pyplot as plt
from keras.models import model_from_json
import numpy as np
import argparse
import imutils
import cv2 as cv
import os
import PIL
from keras import backend as K

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()

# args = vars(ap.parse_args())

# initialize the list of data and labels
data = []
labels = []
a = 0
for imagePath in sorted(list(paths.list_images(args["dataset_test"]))):
    # load the image, pre-process it, and store it in the data list

    # Read PNG
    # image = cv.imread(imagePath)
    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Read in tiff
    pil_image = PIL.Image.open(imagePath).convert('RGB')
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    image = cv.cvtColor(open_cv_image, cv.COLOR_BGR2GRAY)

    # image = imutils.resize(image, width=28)
    # image = imutils.resize(image, width=64)
    image = imutils.resize(image, width=128)
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    # label = "smiling" if label == "positives" else "not_smiling"
    # label = "healthy" if label == "healthy" else "unhealthy"
    if label == "0":
        label = "0"
    elif label == "1":
        label = "1"
    else:
        label = "2"
    labels.append(label)
    a += 1

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 3)

# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

trainX = data
trainY = labels
# Load trained CNN model
# json_file = open('output_to_json/modelQ4_128x128_customLenet.json', 'r')
json_file = open('output_to_json/model_new7.5.2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# model.load_weights('output/lenetQ4_128x128_customLenet.hdf5')
model.load_weights('output/lenet_new7.5.2.hdf5')

trainLabels = list(le.inverse_transform(trainY.argmax(1)))
size = len(trainLabels)
predicted = 0
images = []
x = 0
for i in np.random.choice(np.arange(0, len(trainY)), size=(size,)):

    probs = model.predict(trainX[np.newaxis, i])
    # print(probs)
    prediction = probs.argmax(axis=1)
    label = le.inverse_transform(prediction)
    if label[0] == trainLabels[i]:
        predicted += 1

    # extract the image from the testData if using "channels_first"
    # ordering
    if K.image_data_format() == "channels_first":
        image = (trainX[i][0] * 255).astype("uint8")

    # otherwise we are using "channels_last" ordering
    else:
        image = (trainX[i] * 255).astype("uint8")

    # merge the channels into one image
    image = cv.merge([image] * 3)

    image = cv.resize(image, (128, 128), interpolation=cv.INTER_LINEAR)

    # show the image and prediction
    x += 1
    position = str(x)
    text = position + ' ' + label[0]
    cv.putText(image, str(text), (5, 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    print("[INFO]:{} Predicted: {}, Actual: {}".format(x, label[0],
                                                       trainLabels[i]))
    images.append(image)

print('Accuracy: ',
      predicted / size)
# # img = cv.imwrite('images.png', images)
# images = np.concatenate(images, axis=1)
# cv.imshow("Cell", images)
# cv.waitKey(0)


fig = plt.figure(figsize=(14, 14))
columns = 8
rows = 3
for i in range(0, columns * rows):
    fig.add_subplot(rows, columns, i + 1)
    plt.imshow(images[i])
plt.show()

### ROC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

pred_prob = model.predict_proba(testX)
n_class = 3
# print(testY[:,1].astype(int)
# confusion matrix

from sklearn.metrics import confusion_matrix

for i in range(n_class):
    print(testY[:, i].astype(int))
    print((pred_prob[:, i]).round().astype(int))

for i in range(n_class):
    confusion_matrix1 = confusion_matrix(testY[:, i].astype(int), (pred_prob[:, i]).round().astype(int))
    print('Confusion matrix :', confusion_matrix1)

# roc curve for classes
fpr = {}
tpr = {}
thresh = {}

# from collections import Counter
# Counter(y_true)

for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(testY[:, i].astype(int), (pred_prob[:, i]).round().astype(int))

# plotting
plt.plot(fpr[0], tpr[0], linestyle='--', color='orange', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--', color='blue', label='Class 2 vs Rest')
plt.title('Multiclass ROC curve preprocessed with high-pass filter')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()
plt.savefig('Multiclass ROC', dpi=300);




