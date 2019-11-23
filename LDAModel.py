import numpy as np
from scipy.stats import multivariate_normal

class FisherLD:
    def __init__(self, training_data, training_labels):
        self.training_data = np.matrix(training_data)
        self.training_labels = training_labels
        row, column = self.training_data.shape
        self.no_of_features = column
        self.row_or_column = "column"
        self.classes = sorted(list(set(training_labels)))
        self.train_fisherLD()
        self.project_on_reduced_dimensions(training_data, training_labels, training_run = True)

    def train_fisherLD(self):
        row_t, column_t = np.matrix(self.training_labels).shape
        X = self.training_data
        t = self.training_labels
        if column_t < row_t:
            t = self.training_labels.T

        X_features = []
        for i in range(len(self.classes)):
            X_features.append(X[t == self.classes[i],:])


        Mu, S, Sb = [], [], []
        for i in range(len(self.classes)):
            term = []
            for ii in range(self.no_of_features):
                term.append(np.mean(X_features[i][:,ii]))
            Mu.append(np.matrix(np.array(term)))

            S.append(np.cov(np.matrix(X_features[i].T)))

            term = Mu[i] - np.matrix(np.mean(self.training_data))
            Sb.append(i*np.dot(term.T, term))

        Sw = np.matrix(sum(S))
        Sb = sum(Sb)

        eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))
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


    def classify(self, X):
        if not isinstance(X,(np.ndarray, np.generic)):
            X = X.values
        if self.row_or_column == "row":
            X = np.matrix(X)
        else:
            X = np.matrix(X).T

        length = X.shape[1]

        y = []
        for i in range(len(self.classes)-1):
            y.append(np.asarray(np.dot((self.w[i]).T,X)).reshape(-1))
        pos = np.dstack(y)
        mvn = self.mvn
        p_dict = {}
        for i in range(len(self.classes)):
            mvn_now = mvn[i]
            p_dict[i] = mvn_now.pdf(pos)

        t = []
        for i in range(length):
            cursor = [p_dict[ii][i] for ii in range(len(self.classes))]
            t.append(self.classes[cursor.index(max(cursor))])
        return np.array(t)

    def project_on_reduced_dimensions(self,X, t, training_run = False):
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
        for i in range(len(self.classes)):
            class_handeled = classes[i]
            y_for_each_w_per_class = []
            for ii in range(len(self.classes)-1):
                y_for_each_w_per_class.append(class_handeled[ii,:])
            Mu_class = []
            cov_class = []
            for iii in range(len(y_for_each_w_per_class)):
                Mu_class.append(np.mean(y_for_each_w_per_class[iii]))
                cov_class.append(np.cov(y_for_each_w_per_class[iii]))
            mvn_now = multivariate_normal(Mu_class, cov_class)
            mvn.append(mvn_now)

        if training_run:
            self.mvn = mvn
