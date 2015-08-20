from numpy import *
import scipy.io
import matplotlib.pyplot as pl
from split_data import shuffle_set

# perform the gradient descent optimization.
# parameters:
#     E_: error function E_ = (E,dE) where dE is the derivative of E with
#         respect to a_2 (the activation of layer 2)
#     eta: learning rate
#     mu: parameter for momentum term
# return (W_1,b_1,W_2,b_2) which are all the parameters learned for MLP
def gradient_descent(E_ , W_1, b_1, W_2, b_2, g_, data, labels, valid_data,
                     valid_labels, eta = 0.01, mu = 0.2, early_stop_delta =
                     0.001):

    # used for momentum term
    delta_W_1 = zeros(W_1.shape)
    delta_b_1 = zeros(b_1.shape)
    delta_W_2 = zeros(W_2.shape)
    delta_b_2 = zeros(b_2.shape)
    error_valid = []
    error_train = []
    zo_error_valid = []

    update_errors(error_train,error_valid,zo_error_valid,data,labels,\
                  valid_data,valid_labels,(W_1,b_1,W_2,b_2),E_,g_)
    min_valid_error = 100


    def early_stopping(error_valid,early_stop_delta,error_epsilon = 0.07):
        #if len(error_valid) < 2 or error_valid[-1] > error_epsilon:
            #return False
        #else:
            #return abs(error_valid[-1]-error_valid[-2]) < early_stop_delta
        if len(error_valid) < 4:
            return False
    	return 100*(error_valid[-1]/min_valid_error - 1) > 10 and \
                error_valid[-1] - error_valid[-2] > error_epsilon


    k = 0
    while True:
                

        grad_W_1 = zeros(W_1.shape)
        grad_W_2 = zeros(W_2.shape)
        grad_b_1 = zeros(b_1.shape)
        grad_b_2 = zeros(b_2.shape)
        shuffle_set(data,labels)
        for i in arange(labels.size):
            # online gradient descent -> choose a random input point
            x_i = data[i]
            t_i = labels[i]
            grad_W_1,grad_b_1,grad_W_2,grad_b_2 = \
                    compute_gradient(E_[1],W_1,b_1,W_2,b_2,x_i,t_i,g_)
                   # update the weights and biais
            
            delta_W_1 = -eta*(1-mu)*grad_W_1 + mu*delta_W_1
            delta_b_1 = -eta*(1-mu)*grad_b_1 + mu*delta_b_1
            delta_W_2 = -eta*(1-mu)*grad_W_2 + mu*delta_W_2
            delta_b_2 = -eta*(1-mu)*grad_b_2 + mu*delta_b_2

            W_1 = W_1 + delta_W_1
            b_1 = b_1 + delta_b_1
            W_2 = W_2 + delta_W_2
            b_2 = b_2 + delta_b_2

        
        k = k+1
        # stop if minimum (local or global) reached
        update_errors(error_train,error_valid,zo_error_valid,data,labels,\
                      valid_data,valid_labels,(W_1,b_1,W_2,b_2),E_,g_)
        print "iteration " , k , " , error = ", error_train[-1], ", "\
                    ,error_valid[-1] , "," , zo_error_valid[-1]
        
        min_valid_error = error_valid[-1]\
                if error_valid[-1] < min_valid_error \
                else min_valid_error 

        if (converged(grad_W_1,grad_b_1,grad_W_2,grad_b_2) or\
	    early_stopping(error_valid,early_stop_delta)):
            return (W_1,b_1,W_2,b_2,(error_train,error_valid,zo_error_valid))



def debug(params,grad,E,x,t,g):
    epsilon_d = 1e-10
    W_1,b_1,W_2,b_2 = params[0],params[1],params[2],params[3]
    dW_1,db_1,dW_2,db_2 = grad[0],grad[1],grad[2],grad[3]
    _W_1,_b_1,_W_2,_b_2 = W_1 + epsilon_d, b_1 + epsilon_d,\
            W_2 + epsilon_d, b_2 + epsilon_d
    _a_2 = forward_pass(_W_1,_b_1,_W_2,_b_2,x,g)[2]
    a_2 = forward_pass(W_1,b_1,W_2,b_2,x,g)[2]
    _err,err = E(t,_a_2),E(t,a_2)
    g_1_e = (_err - err)/epsilon_d



def update_errors(error_train,error_valid,zo_error_valid,
                  data,labels,valid_data,valid_labels,params,E_,g_):

    W_1,b_1,W_2,b_2 = params[0],params[1],params[2],params[3]

    s_rate = test_mlp((valid_data,valid_labels),((W_1,b_1,W_2,b_2),g_[0]))
    print "success rate ",s_rate        
    a = [ forward_pass(W_1,b_1,W_2,b_2,valid_data[i],g_[0])[2] for i in range(valid_labels.size)]


    error_train.append(sum([E_[0](labels[i],forward_pass(W_1,b_1,W_2,b_2,data[i],g_[0])[2])\
                for i in range(labels.size)])/labels.size)
    error_valid.append(sum([E_[0](valid_labels[i],a[i])\
                for i in range(valid_labels.size)])/valid_labels.size)
    zo_error_valid.append(1-s_rate)



# Test if gradient_descent has converged
# converges if gradient is 0
def converged(grad_W_1,grad_b_1,grad_W_2,grad_b_2):
    return is_zeros(grad_W_1) and is_zeros(grad_b_1) and\
            is_zeros(grad_W_2) and is_zeros(grad_b_2)

# check if a is 0
def is_zeros(a):
    return not (False in (abs(a) < 1e-15))

# compte the gradient of E for input x
# parameters:
#     dE: derivative of E with respect to a_2 (the activation of layer 2)
#     W_1: weights of the hidden layer
#     b_1: biais of the hidden layer
#     W_2: weights of the output layer
#     b_2: biais of the output layer
#     x: input point on which the gradient is computed
#     t: target value of x
#     g_: transfer function g_ = (g,dg) where dg is the gradient of g
def compute_gradient(dE , W_1 , b_1 , W_2 , b_2 , x, t, g_):
    # first apply forward and backward path to get the residuals
    a_1 , z_1 , a_2 = forward_pass(W_1,b_1,W_2,b_2,x,g_[0])
    r_2 , r_1 = backward_pass(a_2,a_1,W_2,g_[1],dE,t)
    gradient_W_2 = r_2 * z_1  
    gradient_b_2 = r_2
    gradient_W_1 = outer(r_1,x.transpose())
    gradient_b_1 = r_1
    return (gradient_W_1,gradient_b_1,gradient_W_2,gradient_b_2)

# perform the forward pass, return activations of layer 1 and 2
# parameters:
#     W_1: the weights of the hidden layer (layer 1)
#     b_1: the biais of the hidden layer
#     W_2: the weights of the output layer (layer 2)
#     b_2: the biais of the output layer
#     x: the input point applied to get the activations
#     g: the transfer function used
# return (a_1,z_1,a_2)
def forward_pass(W_1,b_1,W_2,b_2,x,g):
    a_1 = dot(W_1,x) + b_1
    z_1 = g(a_1[::2],a_1[1::2]) 
    a_2 = dot(W_2,z_1) + b_2
    return (a_1 , z_1 , a_2) 

def mlp_activation_output(W_1,b_1,W_2,b_2,x,g):
    a_1 = dot(W_1,x.transpose()) + outer(b_1,ones(x.shape[0]))
    z_1 = g(a_1[::2],a_1[1::2])
    return dot(W_2,z_1) + outer(b_2,ones(z_1.shape[1]))


# perform the backward pass, return the residuals
# parameters:
#     a_2: activation of output layer (layer 2)
#     a_1: activation of hidden layer (layer 1)
#     W_2: weights of output layer
#     dg: gradient of g
#     dE: derivative of E (error function) with respect to a_2
#     t: the target of the input that gives a_2 as output
# return (r_2,r_1)
def backward_pass(a_2,a_1,W_2,dg,dE,t):
    r_2 = dE(t,a_2)
    post = dot(W_2.transpose(),r_2)
    r_1 = zeros(a_1.size)
    r_1[::2] = dot(diag(dg[0](a_1[1::2])),post)
    r_1[1::2] = dot(diag(dg[1](a_1[::2],a_1[1::2])),post)
    return (r_2 , r_1)

# run the MLP with 2*h1 hidden activations
# parameters:
#     data: data set to learn
#     labels: labels associated to data
#     h1: parameters determining the number of hidden nodes
def mlp(data,labels,valid_data,valid_labels,h1 = 10,l_rate = 0.01,init_var=0.1,
        momentum_rate=0.2):

    # defintion of the transfer function
    def g(a1,a2):
        return (a1)/(1 + e**(-a2))

    def dg_da1(a2):
        return 1./(1 + e**(-a2))

    def dg_da2(a1,a2):
        return (a1 * e**(-a2))/(1 + e**(-a2))**2

    # definition of the error function
    def E(t,a):
        #x = -t*a
        #x_pos = x.compress((x>=0).flat)
        #x_neg = x.compress((x<0).flat)
        #return ((x_pos + log1p(e**-x_pos)).sum() + log1p(e**x_neg).sum())/t.size
        return log(1 + e**(-t*a)).sum() 

    def dE(t,a):
        #return (- t * e**(-t * a))/(1 + e**(-t * a))
        return (logistic_function(a)-t_tilde(t))

    # dimension of the data input x
    d = data.shape[1]
    # error function
    E_ = ( E , dE )
    g_ = ( g , (dg_da1 , dg_da2) )
    
    # layer initialization
    W_1,b_1,W_2,b_2 = initialize(h1,d,var=init_var)

    # run the gradient descent to minimize the error
    W_1,b_1,W_2,b_2,error = gradient_descent(E_,W_1,b_1,W_2,b_2,g_,data,labels,\
                                             valid_data,valid_labels,eta=l_rate,\
                                             mu=momentum_rate)

    pl.plot(arange(len(error[1])),error[1])
    pl.plot(arange(len(error[2])),error[2])
    # pl.savefig("plots/learning_"+str(l_rate)+"__hidden_"+str(h1)+"__m_rate_"+str(momentum_rate)+".eps",format="eps")
    pl.show()

    return ((W_1,b_1,W_2,b_2),g)

def zoE(t, a):
    return (t*a <= 0).sum()/t.size

# initialize the parameters of the MLP using a gaussian N(0,var)
# parameters:
#     h1: parameters for the hidden layer (layer 1)
#     d: dimension of the data
#     var: variance of the gaussian used to initialize
# return (W_1,b_1,W_2,b_2) which are the parameters defining the two layer of
# the MLP
def initialize(h1,d,var=0.1):
    W_1 = var*random.randn(2*h1,d)
    b_1 = var*random.randn(2*h1)
    W_2 = var*random.randn(1,h1)
    b_2 = var*random.randn(1)
    return (W_1,b_1,W_2,b_2)

# the logistic function sigma
def logistic_function(v):
    return 1./(1 + e**(-v))

# transform the target t from {-1,1} to {0,1} 
def t_tilde(t):
    return (t+1)/2

# compute the target of x using mlp_t
def f(x,mlp_t):
    W_1,b_1,W_2,b_2 = mlp_t[0]
    g = mlp_t[1]
    # return sign(a_2(x))
    return sign(forward_pass(W_1,b_1,W_2,b_2,x,g)[2])

# load data/labels
def load_data(data_part):
    d = scipy.io.loadmat("data/preprocessed_splitted_"+data_part+"_data.mat")
    train_data = d['Xtrain']
    train_label = d['Ytrain']
    valid_data = d['Xvalid']
    valid_label = d['Yvalid']
    return ((train_data,train_label),(valid_data,valid_label))

# load the test set
def load_test(data_part):
    d = scipy.io.loadmat("mnist/mp_"+data_part+"_data.mat")
    test_data = d['Xtest']
    test_label = d['Ytest']
    return (test_data,test_label)

# test the trained mlp on valid
def test_mlp(valid,mlp_t,show_debug=False):
    data = valid[0]
    labels = valid[1]
    success_rate = 0

    min_abs = (-1, -1000)
    max_abs = (-1, 0)

    for i in arange(labels.size):
        W_1,b_1,W_2,b_2 = mlp_t[0]
        a = forward_pass(W_1,b_1,W_2,b_2,data[i],mlp_t[1])[2]
        if labels[i]*a > 0:
            success_rate = success_rate + 1
        else:
            if min_abs[1] < labels[i]*a:
                min_abs = (i, labels[i]*a)

            if max_abs[1] > labels[i]*a:
                max_abs = (i, labels[i]*a)
                
    if show_debug:
        print "min", min_abs[0], min_abs[1]
        print "max", max_abs[0], max_abs[1]
        pl.imshow(data[min_abs[0]].reshape(28,28).transpose())
        pl.title("Minimum a*t = " + str(min_abs[1]))
        pl.savefig("plots/min.eps", format="eps")
        pl.imshow(data[max_abs[0]].reshape(28,28).transpose())
        pl.title("Maximum a*t = " + str(max_abs[1] ))
        pl.savefig("plots/max.eps", format="eps")

    success_rate = success_rate/(1.0*labels.size)
    return success_rate


if __name__ == '__main__':
    data = load_data("3-5")
    train = data[0]
    valid = data[1]
    print "Training on training set ..."
    mlp_param = mlp(train[0],train[1],valid[0],valid[1],\
                    h1=40,l_rate=0.01,init_var=0.1,momentum_rate=0.5)
    print mlp_param
    print "Testing on validation set ..."
    success_rate = test_mlp(valid,mlp_param,True)
    print " - success rate is " , success_rate


