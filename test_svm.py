from svm import *
from mlp import load_test

if __name__ == "__main__":
    data = load_data()
    train = data[0]
    valid = data[1]

    test = load_test("4-9")
    
    print "Training phase"
    alpha,b = svm(data[0],data[1],tau=2**-5,C=2**-4)
    print "Testing phase"
    validate((data[0],data[1]),alpha,b,(test[0]/255,test[1]),2**-5)



