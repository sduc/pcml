import numpy as np
import scipy.io
import matplotlib.pyplot as pl

_DIM_ = 28


def plot_data():
    d = scipy.io.loadmat("data/splitted_3-5_data.mat")
    training_data = d['Xtrain']
    training_labels = d['Ytrain']
    validation_data = d['Xvalid']
    validation_labels = d['Yvalid']

    # plot some digits
    pl.figure(1)
    for i in np.arange(10):
        pl.subplot(2,5,i)
        pl.imshow(training_data[i].reshape(_DIM_,_DIM_).transpose())
        pl.title("Label = "+str(training_labels[i]))

    # plot the hist of labels
    pl.figure(2)
    pl.subplot(1,2,1)
    pl.hist(training_labels)
    pl.subplot(1,2,2)
    pl.hist(validation_labels)
    pl.show()


if __name__ == '__main__':
    plot_data()
