import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
DATADIR = "DataSet/"
CATEGORIES = ["Apple", "Fruit"]

IMG_SIZE = 100
training_data = []

def create_training_data():
  for category in CATEGORIES:

    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=apple 1=notapple

    for img in tqdm(os.listdir(path)):
      try:
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
        training_data.append([new_array, class_num])  # add this to our training_data
      except Exception as e:  # in the interest in keeping the output clean...
        pass


create_training_data()

print(len(training_data))

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy as np
import time

pickle_in = open("X.pickle","rb")
X = np.asarray(pickle.load(open("X.pickle", "rb")))

pickle_in = open("y.pickle","rb")
y = np.asarray(pickle.load(open("y.pickle", "rb")))

X = X/255.0

dense_layers = [0]
layer_sizes = [32]
conv_layers = [3]
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )

            H = model.fit(X, y,
                      batch_size=32,
                      epochs=10,
                      validation_split=0.3,
                      callbacks=[tensorboard],
                      verbose=1)

model.save('64x3-CNN.model')
score = model.evaluate(X, y, verbose=0)
print(score)

fig = plt.figure()
numOfEpoch = 10
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()
plt.show()

def prepare(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.

from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter import messagebox

window = Tk()
window.geometry("550x300+300+150")
window.resizable(width=True, height=True)
window.title("Nhận diện quả táo")

filePath = ""
def openfn():
    global filePath
    filePath = filedialog.askopenfilename(title='Chọn ảnh')
    return filePath



def open_img():
    x = openfn()
    img = Image.open(x)
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(window, image=img)
    panel.image = img
    panel.pack()
b1 = Button(window, text='Chọn ảnh', command=open_img).pack()

def result():
    prediction = model.predict([prepare(filePath)])
    value = CATEGORIES[int(prediction[0][0])]
    if(value == CATEGORIES[0]):
        return "Đây là quả táo"
    else:
        return "Đây không phải quả táo"


def clicked():
    messagebox.showinfo('Kết quả', result())


b2 = Button(window, text='Kiểm tra', command=clicked).pack()

window.mainloop()

