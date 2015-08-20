from svm import *
from socket import gethostname

if __name__ == "__main__":
    hname = gethostname()
    host_num = int(hname[-2:]) -1
    # c = host_num % 10 - 5
    # t = int(hname[-2]) - 5
    data,labels = load_data()
    # risk = cross_validation(data,labels,C=2**c,tau=2**t)
    # f = open("svm_cross_valid_test.txt","a")
    # f.write("log(C) = "+str(c)+", log(tau) = "+str(t)+", Risk = "+str(risk)+"\n")
    # f.close()
    # risk = cross_validation(data,labels,C=2**c,tau=2**(t+6))
    # f = open("svm_cross_valid_test.txt","a")
    # f.write("log(C) = "+str(c)+", log(tau) = "+str(t+6)+", Risk = "+str(risk)+"\n")
    # f.close()
    
    todo = [(-5,0), (-4,0), (-3,0), (-2,0), (-1,-3), (-1,0), (-1,3), (0,0), (1,0), (2,0), (3,-1), (3,0), (4,-5), (4,-4), (4,-1), (4,1), (4,2)]
    
    f = open("svm_final_"+str(host_num)+".txt","a")
    t = todo[host_num % 17]
    risk = cross_validation(data,labels,C=2**t[0],tau=2**(t[1]))
    f.write("host = "+str(host_num)+ ", log(C) = "+str(t[0])+", log(tau) = "+str(t[1])+", Risk = "+str(risk)+"\n")
    f.close()



