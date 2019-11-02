import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal

class FisherLD:
    def __init__(self, training_data, training_labels):

        self.training_data = np.matrix(training_data)
        self.training_labels = training_labels

        row, column = self.training_data.shape
        if row < column:
            self.no_of_features = row
            self.row_or_column = "row"
        else:
            self.no_of_features = column
            self.row_or_column = "column"

        def unique(list1):
            x = np.array(list1)
            return list(x)
        self.classes = sorted(list(set(training_labels)))

        self.train_fisherLD()


    def train_fisherLD(self):
        row_t, column_t = np.matrix(self.training_labels).shape
        X = self.training_data
        t = self.training_labels

        if column_t < row_t:
            t = self.training_labels.T

        X_features = []
        if self.row_or_column == "row":
            for i in range(len(self.classes)):
                X_features.append(X[:,t == self.classes[i]])
        else:
            for i in range(len(self.classes)):
                X_features.append(X[t == self.classes[i],:].T)

        Mu, S, Sb = [], [], []
        for i in range(len(self.classes)):
            term = []
            for ii in range(self.no_of_features):
                term.append(np.mean(X_features[i][:,ii]))
            Mu.append(np.matrix(np.array(term)))

            S.append(np.cov(np.matrix(X_features[i])))

            term = Mu[i] - np.matrix(np.mean(self.training_data))
            n = np.count_nonzero(self.training_labels == self.classes[i])
            Sb.append(i*np.dot(term.T, term))

        Sw = np.matrix(sum(S))
        Sb = sum(Sb)


        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
        eig_dict = {}

        for i in range(len(eig_vals)):
            eigvec_sc = eig_vecs[:,i].reshape(self.no_of_features,1)
            eig_dict[eig_vals[i].real] = eigvec_sc.real

        eig_vector = eig_dict[max([*eig_dict])]

        self.w = eig_vector.reshape(self.no_of_features,1)

        self.c = []
        for i in range(len(self.classes)-1):
            if i == self.no_of_features:
                break

            self.c.append(float((.5)*(Mu[i]+Mu[i+1]).dot(self.w)))

    def project_and_classify(self,X):
        if X.shape[1] == self.w.shape[0]:
            y_toclassify = np.asarray(np.dot(self.w.T,X.T)).reshape(-1)
        else:
            y_toclassify = np.asarray(np.dot(self.w.T,X)).reshape(-1)
        y_legnth = len(y_toclassify)
        t = []
        for i in range(len(self.c)):
            for ii in y_toclassify:
                if ii > self.c[i]:
                    t.append(self.classes[i])
                y_toclassify = y_toclassify[y_toclassify < self.c[i]]

        while len(t) < y_legnth:
            t.append(self.classes[-1])
        if not isinstance(X,(np.ndarray, np.generic)):
            X = X.values
        X_features = []
        if self.row_or_column == "row":
            for i in self.classes:
                X_features.append(X[t == i,:])
        else:
            for i in self.classes:
                X_features.append(X[t == i,:].T)

        y = []
        for i in range(len(self.classes)):
            y.append(np.asarray(np.dot(self.w.T,X_features[i])).reshape(-1))
        print(y)
        mvn = []
        p = []
        f, axes = plt.subplots(1, 1)
        for i in range(len(self.classes)):
            mvn_now = multivariate_normal(np.mean(y[i]),np.cov(y[i]))
            p.append(mvn_now.pdf(y[i]))
            sns.lineplot(y[i], p[i], ax=axes)
        return f, t






Data=np.loadtxt('Data1.txt')
X = Data[:,0:2]
t = Data[:,2]
classifier = FisherLD(X,t)
TestData = np.loadtxt('Test1.txt')
X_test = np.matrix(TestData)
classifier.c
f, t = classifier.project_and_classify(X_test)



import pandas as pd
feature_dict = {i:label for i,label in zip(
                range(4),
                  ('sepal length in cm',
                  'sepal width in cm',
                  'petal length in cm',
                  'petal width in cm', ))}

df = pd.io.parsers.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',',
    )
df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
df.dropna(how="all", inplace=True) # to drop the empty line at file-end

df.tail()


X = df[["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm"]]
t = df["class label"]

from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
label_encoder = enc.fit(t)
t = label_encoder.transform(t) + 1

label_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}


classifier = FisherLD(X,t)
classifier.w
f, t = classifier.project_and_classify(X)
