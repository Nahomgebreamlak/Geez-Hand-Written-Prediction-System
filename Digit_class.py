import pyscreenshot as ImageGrab
import time
from PIL import Image
import tkinter as tk
from tkinter import messagebox
import os
# import cv2
import csv
import glob
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib
from sklearn.svm import SVC
from sklearn import metrics
import joblib
# import cv2
import numpy as np  # pip install numpy


class DigitRecognize:
    # collect images from paint
    def collect_image(self, alphabet="BE"):
        """
        Open paint for capturing the handwritten images and save them in a separate folder

        """
        os.startfile("C:/ProgramData/Microsoft/Windows/Start Menu/Programs/Accessories/Paint")
        os.chdir("D:\Books\Artificial Intelligence\AI assignment\Hand Written Geez Alphabet prediction/capturedimages")
        os.mkdir(alphabet)
        os.chdir("D:\Books\Artificial Intelligence\AI assignment\Hand Written Geez Alphabet prediction")

        images_folder = "capturedimages/" + alphabet + "/"
        time.sleep(20)
        print("You can start know.......")
        for i in range(0, 20):
            time.sleep(15)
            im = ImageGrab.grab(bbox=(60, 170, 400, 550))
            print("Saved ......", i)
            im.save(images_folder + str(i) + ".png")
            print("Clear screen and redraw again")
        messagebox.showinfo("Result", "Captured image complited")

    # generate Data set function  and save it in exel file
    def generate_dataset(self):
        header = ["label"]
        """ create  a dictionary so that each alphabet will be reperesented as a number in exel
        because  excel doesn't support geez alphabets yet. 
        """
        alphabet = {0: "ሀ", 1: "ሐ", 2: "ለ", 3: "መ", 4: "ሰ", 5: "ረ", 6: "ሸ", 7: "ቀ", 8: "በ", 9: "ተ", 10: "ዐ", 11: "ከ", 12: "ወ", 13: "ዘ", 14: "ገ"}
        for i in range(0, 784):
            header.append("pixel" + str(i))
        # create an csv file to save your data set
        with open('datasetfortigrigina.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        # loop through each dict and save each images pixel as a row
        for keyvalue, namevalues in alphabet.items():
            print(f"the key is {keyvalue} and the value {namevalues}")
            dirList = glob.glob("capturedimages/" + namevalues + "/*.png")
            for img_path in dirList:
                im = cv2.imread(img_path)
                im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)
                roi = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)

                data = []
                data.append(keyvalue)
                rows, cols = roi.shape

                ## Add pixel one by one into data array
                for i in range(rows):
                    for j in range(cols):
                        k = roi[i, j]
                        if k > 100:
                            k = 1
                        else:
                            k = 0
                        data.append(k)
                with open('datasetfortigrigina.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(data)
        messagebox.showinfo("Result", "Generating dataset is successfully")

    def train_and_calculate_accuracy(self):

        nan_value = float("NaN")

        data = pd.read_csv('datasetfortigrigina.csv')
        data.replace("", nan_value, inplace=True)
        data.dropna(subset=["pixel6"], inplace=True)
        data = shuffle(data)
        X = data.drop(["label"], axis=1)
        Y = data["label"]
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
        # I choose rbf as my kernel b/c it's accuracy is higher  than others
        # C parameter adds a penalty for each misclassified data point in my case 10 works better
        # random_state: represents the seed of the pseudo random number generated which is used while shuffling the data.

        classifier = SVC(kernel="rbf",random_state=6,C=10)
        classifier.fit(train_x, train_y)
        # save the model using joblib in to specified folder
        joblib.dump(classifier, "model/alphabet_modelrbf")

        # calculate accuracy using the test data

        prediction = classifier.predict(test_x)
        accuracy = metrics.accuracy_score(prediction, test_y)
        messagebox.showinfo("Result", f"Accuracy Result is {accuracy}")


    def display_result(self, image, predictedtext="Be"):
        import numpy as np
        from PIL import ImageFont, ImageDraw, Image
        import cv2
        import time

        savedimage = cv2.imread("img/img.png")

        if (isinstance(savedimage, np.ndarray)):
            imgPIL = Image.fromarray(cv2.cvtColor(savedimage, cv2.COLOR_BGR2RGB))

        font = ImageFont.truetype("nyala.ttf", 132)
        color = 'rgb(255, 0, 0)'
        draw = ImageDraw.Draw(imgPIL)
        draw.text((10, 10), predictedtext, font=font, fill=color)
        imgPutText = cv2.cvtColor(np.asarray(imgPIL), cv2.COLOR_RGB2BGR)

        cv2.imshow("Predicted Image", imgPutText)
        cv2.waitKey(5000)
        if cv2.waitKey(1) == 13 or cv2.waitKey(1) == 27:  # 27 is the ascii value of esc, 13 is the ascii value of enter
            return True

    cv2.destroyAllWindows()

    def live_prediction(self):
        closeValue = False
        alphabet = {0: "ሀ", 1: "ሐ", 2: "ለ", 3: "መ", 4: "ሰ", 5: "ረ", 6: "ሸ", 7: "ቀ", 8: "በ", 9: "ተ", 10: "ዐ", 11: "ከ", 12: "ወ", 13: "ዘ", 14: "ገ"}
        os.startfile("C:/ProgramData/Microsoft/Windows/Start Menu/Programs/Accessories/Paint")
        model = joblib.load("model/alphabet_modelrbf")
        image_folder = "img/"
        time.sleep(10)

        while True:
            img = ImageGrab.grab(bbox=(60, 170, 400, 500))

            img.save(image_folder + "img.png")
            im = cv2.imread(image_folder + "img.png")

            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)

            # Threshold the image
            ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
            roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)

            rows, cols = roi.shape

            X = []

            ## Add pixel one by one into data array
            for i in range(rows):
                for j in range(cols):
                    k = roi[i, j]
                    if k > 100:
                        k = 1
                    else:
                        k = 0
                    X.append(k)

            predictions = model.predict([X])
            alphabetname = alphabet[predictions[0]]

            print("Prediction:", predictions[0])
            print("Prediction:...................", alphabetname)

            closeValue = self.display_result(image=roi, predictedtext=alphabetname)
            if closeValue:  # 27 is the ascii value of esc, 13 is the ascii value of enter
                break
        cv2.destroyAllWindows()
