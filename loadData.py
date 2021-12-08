import numpy as np
import os
import cv2

def create_training_data():
    for subject in SUBJECTS:
        for first in firstRange:
            for degree in degrees:
                path = os.path.join(datadir, subject, "bg-" + first, degree)
                class_num = SUBJECTS.index(subject)
                i = 1
                for img in os.listdir(path):
                    if i <= SEQ_LEN:
                        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                        img_array = cv2.resize(img_array, (height, width))
                        training_data.append([img_array, class_num])
                        i += 1
                if len(os.listdir(path)) < SEQ_LEN:
                    for i in range(SEQ_LEN - len(os.listdir(path))):
                        training_data.append([np.zeros([height, width]), class_num])
                path = os.path.join(datadir, subject, "cl-" + first, degree)
                class_num = SUBJECTS.index(subject)
                i = 1
                for img in os.listdir(path):
                    if i <= SEQ_LEN:
                        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                        img_array = cv2.resize(img_array, (height, width))
                        training_data.append([img_array, class_num])
                        i += 1
                if len(os.listdir(path)) < SEQ_LEN:
                    for i in range(SEQ_LEN - len(os.listdir(path))):
                        training_data.append([np.zeros([height, width]), class_num])
        for second in secondRange:
            for degree in degrees:
                path = os.path.join(datadir, subject, "nm-" + second, degree)
                class_num = SUBJECTS.index(subject)
                i = 1
                for img in os.listdir(path):
                    if i <= SEQ_LEN:
                        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                        img_array = cv2.resize(img_array, (height, width))
                        training_data.append([img_array, class_num])
                        i += 1
                if len(os.listdir(path)) < SEQ_LEN:
                    for i in range(SEQ_LEN - len(os.listdir(path))):
                        training_data.append([np.zeros([height, width]), class_num])


def create_test_data():
    for subject in SUBJECTS:
        for second in testRange:
            for degree in degrees:
                path = os.path.join(datadir, subject, "nm-" + second, degree)
                class_num = SUBJECTS.index(subject)
                i = 1
                for img in os.listdir(path):
                    if i <= SEQ_LEN:
                        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                        img_array = cv2.resize(img_array, (height, width))
                        test_data.append([img_array, class_num])
                        i += 1
                if len(os.listdir(path)) < SEQ_LEN:
                    for i in range(SEQ_LEN - len(os.listdir(path))):
                        test_data.append([np.zeros([height, width]), class_num])
        for first in testRange2:
            for degree in degrees:
                path = os.path.join(datadir, subject, "bg-" + first, degree)
                class_num = SUBJECTS.index(subject)
                i = 1
                for img in os.listdir(path):
                    if i <= SEQ_LEN:
                        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                        img_array = cv2.resize(img_array, (height, width))
                        test_data.append([img_array, class_num])
                        i += 1
                if len(os.listdir(path)) < SEQ_LEN:
                    for i in range(SEQ_LEN - len(os.listdir(path))):
                        test_data.append([np.zeros([height, width]), class_num])
                path = os.path.join(datadir, subject, "cl-" + first, degree)
                class_num = SUBJECTS.index(subject)
                i = 1
                for img in os.listdir(path):
                    if i <= SEQ_LEN:
                        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                        img_array = cv2.resize(img_array, (height, width))
                        test_data.append([img_array, class_num])
                        i += 1
                if len(os.listdir(path)) < SEQ_LEN:
                    for i in range(SEQ_LEN - len(os.listdir(path))):
                        test_data.append([np.zeros([height, width]), class_num])


datadir = r"C:\Users\Enes\Desktop\GaitDatasetB-silh\GaitDatasetB-silh-treated"
SUBJECTS = ["{0:03}".format(i) for i in range(1, 125)]
firstRange = ["{0:02}".format(i) for i in range(1, 2)]
secondRange = ["{0:02}".format(i) for i in range(1, 6)]
testRange = ["{0:02}".format(i) for i in range(6, 7)]
testRange2 = ["{0:02}".format(i) for i in range(2, 3)]
degrees = ["{0:03}".format(i) for i in range(0, 181, 18)]

SEQ_LEN = 50  # 10 = 1540-660
height = 32
width = 32

training_data = []
test_data = []

create_training_data()
create_test_data()

X_train = []
y_train = []
X_test = []
y_test = []

for features, label in training_data:
    X_train.append(features)
    y_train.append(label)

for features, label in test_data:
    X_test.append(features)
    y_test.append(label)


X_train = np.array(X_train).reshape(-1, SEQ_LEN, height, width, 11)
y_train = np.arange(0, 124, 1)
y_train = np.repeat(y_train, 7)

X_test = np.array(X_test).reshape(-1, SEQ_LEN, height, width, 11)
y_test = np.arange(0, 124, 1)
y_test = np.repeat(y_test, 3)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


np.save('X_train_c.npy', X_train)
np.save('y_train_c.npy', y_train)
np.save('X_test_c.npy', X_test)
np.save('y_test_c.npy', y_test)
