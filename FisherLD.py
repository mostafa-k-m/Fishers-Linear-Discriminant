import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
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

        self.w = {}
        for i in range(len(self.classes)-1):
            self.w[i] = (eig_dict.pop(max([*eig_dict])))

        self.c = {}
        for i in range(len(self.classes)-1):
            inner_c = []
            for ii in range(1,len(self.classes)):
                inner_c.append(float((.5)*(Mu[ii-1]+Mu[ii]).dot(self.w[i])))
            self.c[i] = inner_c

    def twoX(self,X, t = []):
        self.c = self.c[0]
        self.w = self.w[0]
        if t == []:
            if X.shape[1] == self.w.shape[0]:
                y_toclassify = np.asarray(np.dot(self.w.T,X.T)).reshape(-1)
            else:
                y_toclassify = np.asarray(np.dot(self.w.T,X)).reshape(-1)
            y_legnth = len(y_toclassify)
            t = []
            for i in range(len(self.c)):
                for ii in y_toclassify:
                    if ii < self.c[i]:
                        t.append(self.classes[i])
                    y_toclassify = y_toclassify[y_toclassify > self.c[i]]

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

        mvn = []
        p = []
        f, axes = plt.subplots(1, 1)
        for i in range(len(self.classes)):
            mvn_now = multivariate_normal(np.mean(y[i]),np.cov(y[i]))
            p.append(mvn_now.pdf(y[i]))
            sns.lineplot(y[i], p[i], ax=axes)
        return f, t

    def threeX(self,X, t):
        if not isinstance(X,(np.ndarray, np.generic)):
            X = X.values
        X_features = []
        if self.row_or_column == "row":
            for i in self.classes:
                X_features.append(X[:,t == i])
        else:
            for i in self.classes:
                X_features.append(X[t == i,:].T)

        y = {}
        for i in range(len(self.classes)-1):
            inner_y = []
            for ii in range(len(self.classes)):
                inner_y.append(np.asarray(np.dot((self.w[i]).T,X_features[ii])).reshape(-1))
            y[i] = np.array(inner_y)

        classes = []
        inner = []
        for i in range(len(self.classes)):
            for ii in range(len(self.classes)-1):
                inner.append(y[ii][i])
            classes.append(np.array(inner))
            inner = []

        mvn = []
        p = []
        f = plt.figure()
        ax = {}
        for i in range(0,4):
            ax[f.add_subplot(2, 2, i+1, projection='3d')] = 45*i
        w_s = []
        for i in range(len(self.classes)):
            class_handeled = classes[i]
            for ii in range(len(self.classes)-1):
                w_s.append(class_handeled[ii,:])
            pos = np.dstack(w_s)
            Mu_class = []
            cov_class = []
            for iii in range(len(w_s)):
                Mu_class.append(np.mean(w_s[iii]))
                cov_class.append(np.cov(w_s[iii]))
            mvn_now = multivariate_normal(Mu_class, cov_class)
            p.append(mvn_now.pdf(pos))
            for iiii in ax:
                iiii.set_xlabel('X Label')
                iiii.set_ylabel('Y Label')
                iiii.set_zlabel('Z Label')
                iiii.view_init(None, ax.get(iiii))
                iiii.scatter(class_handeled[0], class_handeled[1], p[i])
        plt.show()
        return f

    def project_and_classify(self,X, t = []):
        if len(self.classes) == 3:
            return self.threeX(X, t)
        elif len(self.classes) == 2:
            return self.twoX(X, t)













from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris = load_iris()
features = iris.data.T
X = np.vstack((features[0], features[1]))
t = iris.target
X
t
classifier = FisherLD(X,t)
f = classifier.project_and_classify(X,t)



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



X = df[["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm"]]
t = df["class label"]
X

from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
label_encoder = enc.fit(t)
t = label_encoder.transform(t) + 1

label_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}


classifier = FisherLD(X,t)
t
f = classifier.project_and_classify(X, t)
