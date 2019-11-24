import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

class LDAModel:
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

        priors = {}
        for c in self.classes:
            priors[c] = np.sum(c == self.classes)/len(self.classes)

        self.priors = priors

    def classify(self, X, plot = True):
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
            p_dict[self.classes[i]] = mvn_now.pdf(pos)

        t = []
        for i in range(length):
            cursor = [p_dict[ii][i]*self.priors[ii] for ii in (self.classes)]
            t.append(self.classes[cursor.index(max(cursor))])
        if plot:
            self.plot_model(y,np.array(t))

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
        self.y_for_plotting = classes

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

    def plot_model(self,y,t):
        if len(self.classes) == 2:
            f = plt.figure(figsize = (12, 6))
            p = []
            y = [(y[0])[t == i] for i in self.classes]
            for i in range(len(self.classes)):
                mvn_now = self.mvn[i]
                p.append(mvn_now.pdf(y[i]))
            axes = plt.subplot(1, 1, 1)
            axes.set_xlabel('Reduced Dimension 1')
            axes.set_ylabel('Number of observasion with the reduced value')
            axes.set_title('Projection on reduced dimensions')
            for i in range(len(self.classes)):
                axes.hist(y[i],4)
            handles, labels = axes.get_legend_handles_labels()
            f.legend(handles, labels, loc='upper center')
            return f

        elif len(self.classes) >2:

            if len(self.classes) == 3:
                f = plt.figure(figsize = (12, 12))
                f.suptitle('Projection on reduced dimensions', fontsize=16)

                gs = GridSpec(4,4)

                ax_joint = f.add_subplot(gs[1:4,0:3])
                ax_marg_x = f.add_subplot(gs[0,0:3])
                ax_marg_y = f.add_subplot(gs[1:4,3])
                for i in range(len(self.classes)):
                    class_handeled = self.y_for_plotting[i]
                    dim1 = class_handeled[0]
                    dim2 = class_handeled[1]
                    ax_joint.scatter(dim1,dim2)
                    ax_marg_x.hist(dim1)
                    ax_marg_y.hist(dim2,orientation="horizontal")

                # Turn off tick labels on marginals
                plt.setp(ax_marg_x.get_xticklabels(), visible=False)
                plt.setp(ax_marg_y.get_yticklabels(), visible=False)

                # Set labels on joint
                ax_joint.set_xlabel('Reduced Dimension 1 (Y1)')
                ax_joint.set_ylabel('Reduced Dimension 2 (Y2)')

                # Set labels on marginals
                ax_marg_y.set_xlabel('Y2 Point Distribution')
                ax_marg_x.set_ylabel('Y1 Point Distribution')
                handles, labels = ax_joint.get_legend_handles_labels()
                f.legend(handles, labels, loc='lower center')
            else:
                f = plt.figure(figsize = (12, 12))
                f.suptitle('Multivariate Guasian Distribution', fontsize=16)
                ax = {}
                for i in range(0,4):
                    ax[f.add_subplot(2, 2, i+1, projection='3d')] = 45*i
                for i in range(len(self.classes)):
                    class_handeled = self.y_for_plotting[i]
                    dim1 = class_handeled[1]
                    dim2 = class_handeled[2]
                    dim3 = class_handeled[3]
                    for iiii in ax:
                        iiii.set_xlabel('Reduced Dimension 1')
                        iiii.set_ylabel('Reduced Dimension 2')
                        iiii.set_zlabel('Reduced Dimension 3')
                        iiii.view_init(None, ax.get(iiii))
                        iiii.scatter(dim1, dim2, dim3, label = f"Class: {i}")
                        handles, labels = iiii.get_legend_handles_labels()
                        f.legend(handles, labels, loc='lower center')
            return f
