import numpy as np
from tqdm import tqdm
import os
import cv2



class SketchData():

    TRAIN_SIZE = 10000

    RAW_PATH = 'data/raw'
    TRAIN_PATH = 'data/train'
    TEST_PATH = 'data/test'

    NPY_PATHS = {}
    LABELS = {}
    #FULLDATA_SIZES = []

    training_data = []
    validation_data = []
    test_data = []

    def __init__(self):
        idx = 0
        for f in tqdm(os.listdir(self.RAW_PATH)):
            extension = os.path.splitext(f)[1]
            if extension != '.npy':
                print(f, 'is not npy')
            else:
                name = os.path.splitext(f)[0].upper()
                path = os.path.join(self.RAW_PATH, f)
                self.NPY_PATHS[path] = idx
                self.LABELS[name] = idx
                idx += 1

    def build_training_data(self):
        for path, label in zip(self.NPY_PATHS, self.LABELS):
            full_data = np.load(path, encoding='latin1') #load full npy file

            perm = np.random.permutation(self.TRAIN_SIZE) #randomize training data
            partial_data = full_data[perm] #partial data
            print(f"{label} target is {np.eye(len(self.LABELS))[self.LABELS[label]]}")
            for i in range(self.TRAIN_SIZE):
                self.training_data.append([partial_data[i].reshape((28, 28)),
                                           np.eye(len(self.LABELS))[self.LABELS[label]]])
        np.random.shuffle(self.training_data)
        np.save(f'{self.TRAIN_PATH}/training_data.npy', self.training_data)

    def build_test_data(self):
        for f in os.listdir(self.TEST_PATH):
            try:
                path = self.TEST_PATH + '\\' + f
                print(path)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
                img = sobelx + sobely
                img = cv2.resize(img, (28, 28))
                img = img/255.0

                """
                cv2.imshow('asdsad', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                """

                self.test_data.append((np.array(img), f))
            except Exception as e:
                pass

        np.save(f'{self.TEST_PATH}/test_data.npy', self.test_data)
