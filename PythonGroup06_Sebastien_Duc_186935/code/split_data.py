import numpy
import scipy.io

# load the training data part
# data is either '3-5' or '4-9'
def load_mnist(data_part):
    d = scipy.io.loadmat('mnist/mp_'+data_part+'_data.mat')
    data = d['Xtrain']
    labels = d['Ytrain']
    return (data,labels)

# shuffle data and labels the same way
def shuffle_set(data,labels):
    prng_state = numpy.random.get_state()
    numpy.random.shuffle(data)
    numpy.random.set_state(prng_state)
    numpy.random.shuffle(labels)

# split the data training part into the training set 
# and the validation set
# data_part is either '3-5' or '4-9'
def split_data(data_part):
    data,labels = load_mnist(data_part)
    shuffle_set(data,labels)
    split = (2*labels.size)/3
    train_data = data[:split]
    valid_data = data[split:]
    train_labels = labels[:split]
    valid_labels = labels[split:]
    scipy.io.savemat('data/splitted_'+data_part+'_data.mat',\
                     {'Xtrain':train_data,\
                      'Ytrain':train_labels,\
                      'Xvalid':valid_data,\
                      'Yvalid':valid_labels})

if __name__ == "__main__":
    split_data("3-5")
    split_data("4-9")
