import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
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
        self.project_on_reduced_dimensions(training_data, training_labels, training_run = True)


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


    def classify(self, X):
        if not isinstance(X,(np.ndarray, np.generic)):
            X = X.values
        if self.row_or_column == "row":
            X = np.matrix(X)
        else:
            X = np.matrix(X).T
        length = X.shape[1]
        if len(self.classes) == 2:
            y = (np.asarray(np.dot((self.w[0]).T,X)).reshape(-1))

            mvn = self.mvn
            p_dict = {}
            for i in range(len(self.classes)):
                mvn_now = mvn[i]
                p_dict[i] = mvn_now.pdf(y)
        else:
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

    def two_X(self,X, t, training_run = False):
        self.c = self.c
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
            y.append(np.asarray(np.dot(self.w[0].T,X_features[i])).reshape(-1))

        mvn = []
        p = []
        f = plt.figure(figsize = (12, 6))
        axes1 = plt.subplot(1, 2, 1)
        axes1.set_xlabel('Reduced Dimension 1')
        axes1.set_ylabel('Probability')
        axes1.set_title('Gausian Distribution')
        for i in range(len(self.classes)):
            mvn_now = multivariate_normal(np.mean(y[i]),np.cov(y[i]))
            p.append(mvn_now.pdf(y[i]))
            axes1.scatter(y[i], p[i],label = f"Class: {i}")
            mvn.append(mvn_now)
        if training_run:
            self.mvn = mvn

        axes2 = plt.subplot(1, 2, 2)
        axes2.set_xlabel('Reduced Dimension 1')
        axes2.set_ylabel('Number of observasion with the reduced value')
        axes2.set_title('Projection on reduced dimensions')
        for i in range(len(self.classes)):
            axes2.hist(y[i],12)
        handles, labels = axes1.get_legend_handles_labels()
        f.legend(handles, labels, loc='upper center')
        return f

    def more_than_two_X(self,X, t, training_run = False):
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
        if len(self.classes) == 3:
            f1 = plt.figure(figsize = (12, 12))
            f1.suptitle('Multivariate Guasian Distribution', fontsize=16)
            ax = {}
            for i in range(0,4):
                ax[f1.add_subplot(2, 2, i+1, projection='3d')] = 45*i
            f2 = plt.figure(figsize = (12, 12))
            f2.suptitle('Projection on reduced dimensions', fontsize=16)

            gs = GridSpec(4,4)

            ax_joint = f2.add_subplot(gs[1:4,0:3])
            ax_marg_x = f2.add_subplot(gs[0,0:3])
            ax_marg_y = f2.add_subplot(gs[1:4,3])

        mvn = []
        p = []


        for i in range(len(self.classes)):
            class_handeled = classes[i]
            y_for_each_w = []
            for ii in range(len(self.classes)-1):
                y_for_each_w.append(class_handeled[ii,:])
            pos = np.dstack(y_for_each_w)
            Mu_class = []
            cov_class = []
            for iii in range(len(y_for_each_w)):
                Mu_class.append(np.mean(y_for_each_w[iii]))
                cov_class.append(np.cov(y_for_each_w[iii]))
            mvn_now = multivariate_normal(Mu_class, cov_class)
            p.append(mvn_now.pdf(pos))
            mvn.append(mvn_now)
            if len(self.classes) == 3:
                for iiii in ax:
                    iiii.set_xlabel('Reduced Dimension 1')
                    iiii.set_ylabel('Reduced Dimension 2')
                    iiii.set_zlabel('Probability')
                    iiii.view_init(None, ax.get(iiii))
                    iiii.scatter(class_handeled[0], class_handeled[1], p[i], label = f"Class: {i}")
                    handles, labels = iiii.get_legend_handles_labels()
                f1.legend(handles, labels, loc='lower center')
                ax_joint.scatter(class_handeled[0],class_handeled[1])
                ax_marg_x.hist(class_handeled[0])
                ax_marg_y.hist(class_handeled[1],orientation="horizontal")

                # Turn off tick labels on marginals
                plt.setp(ax_marg_x.get_xticklabels(), visible=False)
                plt.setp(ax_marg_y.get_yticklabels(), visible=False)

                # Set labels on joint
                ax_joint.set_xlabel('Reduced Dimension 1 (Y1)')
                ax_joint.set_ylabel('Reduced Dimension 2 (Y2)')

                # Set labels on marginals
                ax_marg_y.set_xlabel('Y2 Point Distribution')
                ax_marg_x.set_ylabel('Y1 Point Distribution')
                f2.legend(handles, labels, loc='lower center')
        if training_run:
            self.mvn = mvn
        if len(self.classes) == 3:
            plt.show()
            return f1, f2
        else:
            print("The model is trained. However we can't show the projection visually because the number of reduced dimensions is equal to or more than three. \nThe model is ready to classify new observations.")

    def project_on_reduced_dimensions(self,X, t, training_run = False):
        if len(self.classes) > 2:
            return self.more_than_two_X(X, t, training_run)
        elif len(self.classes) == 2:
            return self.two_X(X, t, training_run)
