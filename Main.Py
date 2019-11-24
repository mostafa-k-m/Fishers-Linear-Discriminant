from keras.datasets import mnist
import numpy as np
from LDAModel import LDAModel
(trainX, trainy), (testX, testy) = mnist.load_data()

def flatten(images):
    length_of_flattened_image = images.shape[1]*images.shape[2]
    image_array=np.zeros((images.shape[0],length_of_flattened_image))
    for i in range(len(images)):
        image_array[i,:]=images[i].reshape((1,length_of_flattened_image))
    return image_array


trainX = flatten(trainX)
testX = flatten(testX)


classifier = LDAModel(trainX, trainy)
t = classifier.classify(testX)

#Calculate Accuracy and Print it
Accuracy = 100*np.sum(t == testy)/len(testy)
print(f"Accuracy of predicted labels = {Accuracy}%")

def center(image):
   mask = image>95
   image = image[np.ix_(mask.any(1),mask.any(0))]
   shortest_dimension = min(image.shape)
   compensation = 14 - np.floor((shortest_dimension)//2)
   return np.pad(image, pad_width = int(compensation), mode = 'constant', constant_values = 0)


def crop(images,cropx,cropy):
  no_of_images = images.shape[0]
  images_cropped = np.empty((no_of_images , cropx * cropy) , dtype = int)

  for i in range(no_of_images):
    image = images[i,:].reshape(28,28)
    image = center(image)
    dimensions = image.shape
    startx = (dimensions[1] - cropx)//2
    starty = (dimensions[0]-cropy)//2
    cropped_image = image[starty:starty+cropy:,startx:startx+cropx].reshape(1, cropx * cropy)
    images_cropped[i,:] = cropped_image

  return images_cropped

def drop_constant_pixels(images):
    comparison = images == images[0,:]
    constant_columns = comparison.all(axis = 0)
    return images[:, ~constant_columns]


V_black_n_white = np.vectorize(lambda x: 0 if x< 120 else 1)

def pre_process(images,cropx,cropy):
    return V_black_n_white(drop_constant_pixels(crop(images,cropx,cropy)))


import matplotlib.pylab as plt
trainX_processed = pre_process(trainX,23,23)
testX_processed = pre_process(testX,23,23)



classifier = LDAModel(trainX_processed, trainy)
t = classifier.classify(testX_processed)

Accuracy = 100*np.sum(t == testy)/len(testy)
print(f"Accuracy of predicted labels = {Accuracy}%")

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

cm = confusion_matrix(testy, t)

print('Confusion Matrix: ')
print(cm)
print('\n')
print('Accuracy Score :',accuracy_score(testy, t))
print('\n')
print('Report : ')
print(classification_report(testy, t))



title = 'Confusion matrix'

cmap = plt.cm.Greens
classes = np.unique(testy)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes, yticklabels=classes,
       title=title,
       ylabel='True label',
       xlabel='Predicted label')

ax.margins(y = 5)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")


fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()

fig.savefig('Confusion Matrix.png')


import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Download the data into a pandas Dataframe
df = pd.io.parsers.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',',
    )

#Picking out the desired features, and cleaning the data
feature_dict = {i:label for i,label in zip(
                range(4),
                  ('sepal length in cm',
                  'sepal width in cm',
                  'petal length in cm',
                  'petal width in cm', ))}
df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
df.dropna(how="all", inplace=True) # to drop the empty line at file-end
X = df[["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm"]]

#Formating the data labels into three classes of (1, 2, 3)
t = df["class label"]
enc = LabelEncoder()
label_encoder = enc.fit(t)
t = label_encoder.transform(t) + 1
label_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}

from sklearn.model_selection import train_test_split
train_data, test_data, train_lbls, test_lbls = train_test_split(X, t, test_size = 0.2, random_state = 100)
classifier = LDAModel(train_data,train_lbls)

t = classifier.classify(test_data)

f = classifier.project_on_reduced_dimensions(test_data,t)
