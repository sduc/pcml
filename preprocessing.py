import numpy as np
import scipy.io

def load(data_part):
    d = scipy.io.loadmat("data/splitted_"+data_part+"_data.mat")
    training_data = d['Xtrain']
    training_labels = d['Ytrain']
    validation_data = d['Xvalid']
    validation_labels = d['Yvalid']
    return (training_data,training_labels,validation_data,validation_labels)

def save(train_data,train_labels,valid_data,valid_labels,data_part):
    scipy.io.savemat("data/preprocessed_splitted_"+data_part+"_data.mat",\
                     {'Xtrain':train_data,\
                      'Ytrain':train_labels,\
                      'Xvalid':valid_data,\
                      'Yvalid':valid_labels})

# simply preprocess data by doing the following:
#   \frac{1}{\alpha_max - \alpha_min}(data - \alpha_min*1)
# as it is described in section 3.3 of the instructions
def preprocess(data):
    a_max =  data.max(1)
    a_min = data.min(1)
    preproc_data = np.outer((1./(a_max-a_min)),np.ones(data.shape[1]))*(data - np.outer(a_min,\
                                                    np.ones(data.shape[1])))
    return preproc_data

def apply_preprocess_on(data_part):
    td,tl,vd,vl = load(data_part)
    split = tl.size
    pdata = preprocess(np.concatenate((td,vd)))
    save(pdata[:split],tl,pdata[split:],vl,data_part)


if __name__ == "__main__":
    apply_preprocess_on("3-5")
    apply_preprocess_on("4-9")
