from sys import *
from mlp import *

if __name__ == "__main__":
    if len(argv) < 2:
        print "add the argument"
        exit()

    data_part = argv[1]
    
    data = load_data(data_part)
    train = data[0]
    valid = data[1]

    test = load_test(data_part)
    
    print "Training phase"
    m_p = mlp(train[0],train[1],valid[0],valid[1],\
              h1=40,l_rate=0.01,init_var=0.1,momentum_rate=0.5)

    print "Testing phase"
    success_rate = test_mlp(test,m_p)
    print " -> The success rate on the test set is", success_rate



