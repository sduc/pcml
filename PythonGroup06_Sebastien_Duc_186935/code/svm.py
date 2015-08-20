from numpy import *
import scipy.io
import matplotlib.pyplot as pl


EPSILON = 2e-15

XOR_DATA = array([0,1,1,0,0,0,1,1]).reshape(4,2)
XOR_LABELS = array([1,1,-1,-1])

# implentation of the smo algorithm. It returns
# a vector alpha of dimension n, where n is the number of data elements 
# parameters:
#   kernel: the kernel matrix of the training set
#   target: the targets of the training set
#   C: the paramter C in SVM
#   eta_thershold: threshold used in smo instead of 0
def smo(kernel, target, C, eta_threshold=10**-15, tau=10**-8):
    # initializations
    alpha = zeros(target.shape)
    f = -target

    I_low,I_up = compute_I_sets(alpha,target,C)
    b = 0

    while(True):
        alpha_j_new = 0
        i,j,b_low,b_up = select_pair(I_low,I_up,f,tau)
        if j == -1:
            b = 0.5*(b_low + b_up)
            break
        
        sigma = target[i]*target[j]
        L,H = compute_L_H(sigma, alpha[i], alpha[j],C)
        eta = kernel[i,i] + kernel[j,j] - 2*kernel[i,j]
        
        if eta > eta_threshold:
            alpha_j_unc = compute_min_along_dir_constraint(alpha[j],\
                                                           target[j],\
                                                           f[i],f[j],\
                                                           eta)
            alpha_j_new = clip_unconstrained_min(alpha_j_unc,L,H)
        else:
            phi_H,phi_L = compute_phi(alpha,kernel,sigma,target,f,i,j,L,H)
            if phi_L > phi_H:
                alpha_j_new = H
            else:
                alpha_j_new = L

        alpha_i_new = alpha[i] + sigma*(alpha[j]-alpha_j_new)

        f = f + target[i]*(alpha_i_new - alpha[i])*kernel[:,i].reshape(f.shape) +\
                target[j]*(alpha_j_new - alpha[j])*kernel[:,j].reshape(f.shape)


        alpha[i] = alpha_i_new
        alpha[j] = alpha_j_new
        

        #phi = 0.5 * sum([alpha[i]*alpha[j]*target[i]*target[j]*kernel[i,j] for i in range(alpha.size) for j in range(alpha.size)]) - sum(alpha[i] for i in range(alpha.size))
        #print "phi = ", phi
        #s = (alpha*target).sum()
        #print "s =",s
        
        I_low,I_up = update_I_sets(I_low,I_up,alpha,target,C,i,j)

    return alpha,b

def compute_I_sets(alpha,t,C):
    I_0 =  flatnonzero(logical_and(EPSILON <= alpha,alpha <= C - EPSILON))
    I_plus = r_[ flatnonzero(logical_and(t==1,alpha<EPSILON)) ,\
                flatnonzero(logical_and(t==-1,alpha > C - EPSILON)) ]
    I_minus = r_[ flatnonzero(logical_and(t==-1,alpha<EPSILON)) ,\
                 flatnonzero(logical_and(t==1,alpha > C - EPSILON)) ]
    return union1d(I_0,I_minus),union1d(I_0,I_plus) 

def select_pair(I_low,I_up,f,tau):
    i_up = I_up[argmin(f[I_up])]
    i_low = I_low[argmax(f[I_low])]
    b_low = f[i_low]
    b_up = f[i_up]

    if f[i_low] <= f[i_up] + 2*tau:
        i_low = -1
        i_up = -1

    return (i_low,i_up,b_low,b_up)

def compute_L_H(sigma, alpha_i, alpha_j, C):
    sigma_w = alpha_j + sigma*alpha_i
    L = max(0,sigma_w - indic(sigma,[1])*C)
    H = min(C,sigma_w + indic(sigma,[-1])*C)
    return L,H


def compute_min_along_dir_constraint(alpha_j,t_j,f_i,f_j,eta):
    return alpha_j + (t_j*(f_i-f_j))/(eta)

def clip_unconstrained_min(a_j_unc,L,H):
    return array(a_j_unc).clip(min=L , max=H) 

def compute_phi(alpha,K,sigma,t,f,i,j,L,H):
    w = alpha[i] + sigma*alpha[j]
    L_i = w - sigma*L
    H_i = w - sigma*H
    v_i = f[i] + t[i] - alpha[i]*t[i]*K[i,i] - alpha[j]*t[j]*K[i,j]
    v_j = f[j] + t[j] - alpha[i]*t[i]*K[i,j] - alpha[j]*t[j]*K[j,j]
    phi = lambda l,l_i: 0.5 * ( K[i,i] * (l_i**2) + K[j,j]*(l**2))\
                        + sigma * K[i,j]*l_i*l + t[i]*l_i*v_i\
                        + t[j]*l*v_j - l_i - l
    return phi(H,H_i),phi(L,L_i)

# compute index sets I_low, I_up
def update_I_sets(I_low,I_up,alpha,t,C,i,j):
    def update_index(k,I_low,I_up,alpha,t,C):
        if alpha[k] > EPSILON and alpha[k] < C - EPSILON:
            I_low , I_up = union1d(I_low,[k]) , union1d(I_up,[k])
        elif (alpha[k] <= EPSILON and t[k] == 1) or\
                (alpha[k] >= C - EPSILON and t[k] == -1):
            I_low = setdiff1d(I_low,[k])
            I_up = union1d(I_up,[k])
        else:
            I_low = union1d(I_low,[k])
            I_up = setdiff1d(I_up,[k])
        return I_low,I_up

    I_low,I_up = update_index(i,I_low,I_up,alpha,t,C)
    I_low,I_up = update_index(j,I_low,I_up,alpha,t,C)
    return I_low,I_up

def recompute_f(alpha,t,K):
    return dot((alpha*t).transpose(),(K).transpose()) - t

# indicator function. indic(a,A) = 1_{a \in A}
def indic(a,A):
    return a in A

# verify if the KKT conditions are satified
# parameters:
#   alpha: the ouptput of svm
#   b: the biais output of svm
#   t: the target values of the training set
#   x: the data of the training set
#   C: the parameter of svm
#   tau: the meta paramter of the kernel function
def KKT(alpha,b,t,x,C,tau):
    for i in range(alpha.size):
        if alpha[i] > C + EPSILON or alpha[i] < -EPSILON:
            print "Fail, alpha not in range"
            print "     -> alpha_i =" , alpha[i]
        if alpha[i] < EPSILON:
            if t[i]*discriminant(x[i],x,alpha,t,b,gaussian_kernel_function,tau)\
               - 1 < EPSILON:
                print "FAIL 0",t[i]*discriminant(x[i],x,alpha,t,b,gaussian_kernel_function,tau)\
               - 1
                print "     -> alpha_i =" , alpha[i]
        elif alpha[i] > C-EPSILON:
            if t[i]*discriminant(x[i],x,alpha,t,b,gaussian_kernel_function,tau)\
               - 1 > EPSILON:
                print "FAIL C ",t[i]*discriminant(x[i],x,alpha,t,b,gaussian_kernel_function,tau)\
               - 1 
                print "     -> alpha_i =" , alpha[i]
        else:
            if abs(t[i]*discriminant(x[i],x,alpha,t,b,gaussian_kernel_function,tau)\
               - 1) < EPSILON:
                print "FAIL (0,C)",t[i]*discriminant(x[i],x,alpha,t,b,gaussian_kernel_function,tau)\
               - 1, "  alpha = ",alpha[i] 
                print "     -> alpha_i =" , alpha[i]




# compute the kernel matrix using the gaussian kernel function
def gaussian_kernel_matrix(data,tau):
    xxT = dot(data,(data.transpose()))
    d = diag(xxT)
    A = 0.5 * outer(d,ones(data.shape[0])) +\
        0.5 * outer(ones(data.shape[0]),d) - xxT
    return e**(-tau*A)

def gaussian_kernel_function(x_i,x_j,tau):
    return e**(-0.5*tau*linalg.norm(x_i-x_j)**2) 

# function running svm on data and label with parameter tau and C
# parameters:
#   data: the data of the training set
#   labels: the targets of the training set
#   tau: the meta parameter of the kernel function
#   C: the parameter of svm
def svm(data,labels, tau=8, C=30):
    kernel = gaussian_kernel_matrix(data,tau)
    print "K =",kernel
    alpha,b = smo(kernel,labels,C)
    print "alpha,b",alpha.transpose() , b
    #KKT(alpha,b,labels,data,C,tau)
    return alpha,b

# load the data and the labels that will be used to learn svm
def load_data():
    d = scipy.io.loadmat('data/preprocessed_splitted_4-9_data.mat') 
    data = r_[ d['Xtrain'] , d['Xvalid'] ]
    labels = r_[ d['Ytrain'] , d['Yvalid'] ]
    return data,labels

# function used to get ith the training and validation sets for a k-fold
# cross-validation.
# parameters:
#   data: data of the overall training set
#   labels: targets of the overall training set
#   kernel: the kernel matrix for the overall training set
#   i: to get the ith (train,valid) sets for cross-validation. i \in [0,k[
#   k: to get a k-fold cross validation
# return: (training,validation,sub_kernel), the corresponding training set,
#       validation set and submatrix of the overall kernel matrix for the ith
#       run of the k-fold cross-validation
def k_fold_cross_validation_split(data,labels,kernel,i,k=10):
    subset_size = labels.size / k
    training = (r_[data[:subset_size*i] , data[subset_size*(i+1):]],\
                r_[labels[:subset_size*i] , labels[subset_size*(i+1):]])
    validation =(data[subset_size*i:subset_size*(i+1)],\
                 labels[subset_size*i:subset_size*(i+1)])
    sub_kernel = r_[ c_[ kernel[:subset_size*i,:subset_size*i] ,\
                         kernel[:subset_size*i,subset_size*(i+1):]] ,\
                     c_[ kernel[subset_size*(i+1):,:subset_size*i] ,\
                         kernel[subset_size*(i+1):,subset_size*(i+1):]] ]

    return training,validation,sub_kernel


# do a k-fold cross validation to evaluate parameters C au tau
def cross_validation(data,labels,C,tau,k=10):
    print "Compute kernel matrix..."
    kernel_matrix = gaussian_kernel_matrix(data,tau)
    risk = []
    for i in range(k):
        print "start training on set ", i+1 , "..."
        train,valid,sub_kernel = k_fold_cross_validation_split(data,labels,kernel_matrix,i,k)
        risk.append(train_and_evaluate(sub_kernel,train,valid,C,tau))
    return sum(risk)/k

def train_and_evaluate(train_kernel,train,valid,C,tau):
    alpha,b = smo(train_kernel,train[1],C)
    print alpha.transpose(),b
    #KKT(alpha,b,train[1],train[0],C,tau)
    return validate(train,alpha,b,valid,tau)

def validate(train,alpha,b,valid,tau):
    print "starting validation ..."
    valid_data,valid_t = valid[0],valid[1]
    E_svm = sum([alpha[i] * hinge(1-valid_t[i]*\
                   discriminant(valid_data[i],train[0],alpha,\
                                train[1],b,gaussian_kernel_function,tau))\
             for i in xrange(valid_t.size)])/valid_t.size
    print "    Esvm =",E_svm
    return E_svm

def hinge(x):
    return x if x>EPSILON else 0

# compute y(x) = \sum_{i=1}^{n}{\alpha_{*,i} t_i K(x_i,x)} + b_*
# where x_i's are in data, and K(.,.) is kernel_function
def discriminant(x,data,alpha,t,b,kernel_function,tau):
    tmp = data - outer(ones(t.size),x)
    return dot((alpha*t).transpose(),\
               (gaussian_function((tmp*tmp).sum(axis=1),tau)))
     

def gaussian_function(x,tau):
    return e**(-0.5*tau*x)

def success_rate(alpha,b,train,valid,kernel_function,tau):
    return sum([1 for i in range(valid[1].size) \
        if sign(discriminant(valid[0][i],train[0],\
                             alpha,train[1],b,kernel_function,tau))\
                == valid[1][i]])/double(valid[1].size)



if __name__ == "__main__":
    data,labels = load_data()
    risk = cross_validation(data[:],labels[:],C=2**-2,tau=2**-5)
    print "R = ", risk

