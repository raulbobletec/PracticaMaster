import numpy as np
import pickle
import tensorflow.keras


from enum import Enum
import cv2
from tensorflow.keras.optimizers import Adadelta
from sklearn import svm
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from skimage.feature import corner_fast

from PyQt5.QtWidgets import QFileDialog

from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Data:
    path = None


class Ped(Enum):
    non_ped = 0
    ped = 1


def invatareHog():
    list1 = np.load('C:/Users/Raul/Documents/HOG' + '/features.npy')
    list2 = np.load('C:/Users/Raul/Documents/HOG' + '/labels.npy')
    print("Files loaded")
    list2 = np.array(list2).reshape(len(list2), 1)
    list1 = np.array(list1)
    print("Files transformerd")
    clf = svm.SVC()
    data_frame = np.hstack((list1, list2))
    np.random.shuffle(data_frame)
    percentage = 80
    partition = int(len(list1) * percentage / 100)
    print("Creating sets")
    x_train, x_test = data_frame[:partition, :-1], data_frame[partition:, :-1]
    y_train, y_test = data_frame[:partition, -1:].ravel(), data_frame[partition:, -1:].ravel()
    print("Learning has beggined")
    clf.fit(x_train, y_train)
    print("Prediction has beggined")
    y_pred = clf.predict(x_test)
    print("Finalized")
    print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
    print('\n')
    print(classification_report(y_test, y_pred))

    filename = 'hog_model.sav'
    pickle.dump(clf, open('C:/Users/Raul/Documents/HOG' + "\\" + filename, 'wb'))


def testareHog():
    filename = 'hog_model.sav'
    loaded_model = pickle.load(open('C:/Users/Raul/Documents/HOG' + "\\" + filename, 'rb'))
    list1 = np.load('C:/Users/Raul/Documents/HOG' + '/features_test.npy')
    list2 = np.load('C:/Users/Raul/Documents/HOG' + '/labels_test.npy')
    list1 = np.array(list1)
    y_pred = loaded_model.predict(list1)
    print("Accuracy: " + str(accuracy_score(list2, y_pred)))
    print('\n')
    print(classification_report(list2, y_pred))


def clasificareHog():
    if Data.path is None:
        print("Choose a file.")
    else:
        entry = []
        filename = 'hog_model.sav'
        loaded_model = pickle.load(open('C:/Users/Raul/Documents/HOG' + "\\" + filename, 'rb'))
        image = cv2.imread(Data.path, cv2.IMREAD_UNCHANGED)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (18, 36), interpolation=cv2.INTER_AREA)
        final_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        entry.append(final_image)
        result = loaded_model.predict(entry)
        print("Network using HOG - " + Data.path + " - Result: " + Ped(int(result.item(0))).name)


def invatareFast():
    list1 = np.load('C:/Users/Raul/Documents/HOG' + '/features_testFAST.npy')
    list2 = np.load('C:/Users/Raul/Documents/HOG' + '/labels_testFAST.npy')
    print("Files loaded")
    list1 = np.array(list1).reshape((len(list1), -1))
    list2 = np.array(list2).reshape(len(list2), 1)
    print("Files transformerd")
    clf = svm.SVC(verbose=True)
    data_frame = np.hstack((list1, list2))
    np.random.shuffle(data_frame)
    percentage = 80
    partition = int(len(list1) * percentage / 100)
    print("Creating sets")
    x_train, x_test = data_frame[:partition, :-1], data_frame[partition:, :-1]
    y_train, y_test = data_frame[:partition, -1:].ravel(), data_frame[partition:, -1:].ravel()
    print("Learning has beggined")
    clf.fit(x_train, y_train)
    print("Prediction has beggined")
    y_pred = clf.predict(x_test)
    print("Finalized")
    print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
    print('\n')
    print(classification_report(y_test, y_pred))
    filename = 'fast_model.sav'
    pickle.dump(clf, open('C:/Users/Raul/Documents/HOG' + "\\" + filename, 'wb'))


def testareFast():
    filename = 'fast_model.sav'
    loaded_model = pickle.load(open('C:/Users/Raul/Documents/HOG' + "\\" + filename, 'rb'))
    list1 = np.load('C:/Users/Raul/Documents/HOG' + '/features_testFAST.npy')
    list2 = np.load('C:/Users/Raul/Documents/HOG' + '/labels_testFAST.npy')
    list1 = np.array(list1).reshape(len(list1), -1)
    y_pred = loaded_model.predict(list1)
    print("Accuracy: " + str(accuracy_score(list2, y_pred)))
    print('\n')
    print(classification_report(list2, y_pred))


def clasificareFast():
    if Data.path is None:
        print("Choose a file")
    else:
        entry = []
        filename = 'fast_model.sav'
        loaded_model = pickle.load(open('C:/Users/Raul/Documents/HOG' + "\\" + filename, 'rb'))
        image = cv2.imread(Data.path, cv2.IMREAD_UNCHANGED)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (18, 36), interpolation=cv2.INTER_AREA)
        final_image = corner_fast(image)
        entry.append(final_image)
        entry1 = np.array(entry).reshape((len(entry), -1))
        result = loaded_model.predict(entry1)
        print("Network using FAST - " + Data.path + " - Result: " + Ped(int(result.item(0))).name)


aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")


def invatareXception():
    zzz = ['no', 'yes']
    # date antrenare
    # list1 = []  # features
    # list2 = []  # label
    # folders = ['1', '2', '3']
    # path = 'C:/Users/Raul/Downloads/DC-ped-dataset_base.tar'
    # for folder in folders:
    #    for subfolder in os.listdir(path + '/' + folder):
    #        for filename in os.listdir(path + '/' + folder + '/' + subfolder):
    #            image = imageio.imread(path + "/" + folder + '/' + subfolder + '/' + filename)
    #            im2 = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    #            resized_image_pixels = np.array(Image.fromarray(im2).resize((71, 71)))
    #            list1.append(resized_image_pixels)
    #            list2.append(keras.utils.to_categorical(EnumType.Ped[subfolder].value, num_classes=2))

    # np.save('C:/CAR/Training' + '/features.npy', list1)
    # np.save('C:/CAR/Training' + '/labels.npy', list2)

    list1 = np.load('C:/CAR/Training' + '/features.npy')
    list2 = np.load('C:/CAR/Training' + '/labels.npy')
    combined = list(zip(list1, list2))
    np.random.shuffle(combined)
    percentage = 80
    partition = int(len(list1) * percentage / 100)
    train = combined[:partition]
    test = combined[partition:]
    modello = get_model(zzz)
    modello.fit_generator(
        aug.flow(list1, list2, batch_size=64),
        epochs=5,
        steps_per_epoch=200,
        workers=512
    )
    # modello.fit(list1, list2, epochs=3, verbose=1, batch_size=64, steps_per_epoch=200, workers=512,
    #            use_multiprocessing=True)
    loss, acc = modello.evaluate(list1, list2)
    a = "Training Test Accuracy: " + str(round(acc * 100, 4)) + "%"
    print(a)
    modello.save('my_model.h5')


def get_model(list):
    model_base = tensorflow.keras.applications.xception.Xception(include_top=False, input_shape=(*(71, 71), 3),
                                                                 weights='imagenet')
    output = Flatten()(model_base.output)

    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(128, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(len(list), activation='softmax')(output)
    model = Model(model_base.input, output)
    for layer in model_base.layers:
        layer.trainable = True
    model.summary(line_length=200)
    import pydot
    pydot.find_graphviz = lambda: True
    from tensorflow.keras.utils import plot_model
    plot_model(model, show_shapes=True, to_file='C:/CAR/LOG/model_pdfs/{}.pdf'.format('Xception'))
    ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(optimizer=ada,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def testareXception():
    loaded_model = load_model('my_model.h5')
    feat1 = np.load('C:/Users/Raul/Documents/XceptionTestPics/T1/features_testXception.npy')
    feat2 = np.load('C:/Users/Raul/Documents/XceptionTestPics/T2/features_testXception.npy')
    lab1 = np.load('C:/Users/Raul/Documents/XceptionTestPics/T1/labels_testXception.npy')
    lab2 = np.load('C:/Users/Raul/Documents/XceptionTestPics/T2/labels_testXception.npy')
    h1, l1 = loaded_model.evaluate(feat1, lab1)
    h2, l2 = loaded_model.evaluate(feat2, lab2)
    print("Private Test Accuracy: " + str(round(l1 * 100, 4)) + "%")
    print("Public Test Accuracy: " + str(round(l2 * 100, 4)) + "%")


def clasificareXception():
    if Data.path is None:
        print("Choose file.")
    else:
        entry = []
        loaded_model = load_model('my_model.h5')
        image = cv2.imread(Data.path, cv2.IMREAD_UNCHANGED)
        if len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, (71, 71), interpolation=cv2.INTER_AREA)
        entry.append(image)
        result = loaded_model.predict(np.asarray(entry))
        print("Network using Xception - " + Data.path + " - Result: " + Ped(
            int(result.item(0))).name)


def fileDialog():
    filename = QFileDialog.getOpenFileName()
    if filename[0] == '':
        Data.path = None
    else:
        Data.path = filename[0]
