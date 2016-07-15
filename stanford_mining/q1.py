#!/usr/bin/python -u

from __future__ import division
import numpy
import random
import math
import pylab

path = ""
INPUT_DIR = ""
TRAIN_FILE = INPUT_DIR+"ratings.train.txt"
TEST_FILE = INPUT_DIR+"ratings.val.txt"

def get_matrix_dims(fileName):
    maxUserId = 0
    maxMovieId = 0
    data = open(fileName,"r")
    count = 0
    for line in data:
        count += 1
        [userId, movieId, rating] = line.split("\t")
        if int(movieId) > maxMovieId:
            maxMovieId  = int(movieId)

        if int(userId) > maxUserId: 
            maxUserId  = int(userId)
    data.close()
    print count
            
    return (maxUserId, maxMovieId)


def create_random_matrix(rows, cols):
    tmp = numpy.array([random.gauss(0, math.sqrt(5/cols)) for i in xrange(rows*cols)])
    tmp.shape = (rows,cols)
    return tmp
  
  
def total_error(P, Q, RFile, lam):
    data = open(RFile,"r")
    err = 0.0
    sumPL2Norm = sum([numpy.dot(i,i) for i in P])
    sumQL2Norm = sum([numpy.dot(i,i) for i in Q])
    
    for line in data:
        [u, i, r] = line.split("\t")
        u = int(u) - 1
        i = int(i) - 1
        r = float(r)
        err = err + math.pow((r - numpy.dot(Q[i,:], P[u,:])),2)
        
    err = err + lam * (sumPL2Norm + sumQL2Norm)
    
    return err   

def t_error(P, Q, RFile):
    data = open(RFile,"r")
    err = 0.0
    
    for line in data:
        [u, i, r] = line.split("\t")
        u = int(u) - 1
        i = int(i) - 1
        r = float(r)
        err = err + math.pow((r - numpy.dot(Q[i,:], P[u,:])),2)
    
    return err 



def stochastic_gradient_descent(P, Q, RFile, iter, lam, eta):
    print "Running SGD for %s iterations." % iter
    boom = False
    error = []
    for j in xrange(iter):
       if boom: break
       print "Iteration :",j, "...",
       data = open(RFile,"r")
       for line in data: 
           # Get u , m , r
           [u, i, r] = line.split("\t")
           # Numpy indexes array from 0
           u = int(u) - 1
           i = int(i) - 1
           r = float(r)

           if math.isnan(numpy.dot(Q[i,:], P[u,:])):
               errorTxt =  "(Iteration %s :Nan at u: %s i: %s r: %s )" \
                                % (j,u,i,r)
               boom = True
               raise Exception(errorTxt)
              
           # Calculate Error
           e = r - numpy.dot(Q[i,:], P[u,:]) 

           #Update q
           tmp_q = Q[i,:] + eta*(e*P[u,:] - lam*Q[i,:])

           #Update p

           tmp_p = P[u,:] + eta*(e*Q[i,:] - lam*P[u,:])

       
           Q[i,:] = tmp_q
           P[u,:] = tmp_p
           
       data.close() 
       print "done.",
       error.append(total_error(P,Q,RFile,lam))
       print "Error:[ %s ]" % (error[j])
    return P, Q, error

def stochastic_gradient_descent2(P, Q, RFile, iter, lam, eta, k):
    print "Running SGD for %s iterations." % iter
    boom = False
    error = []
    for f in xrange(k):
        for j in xrange(iter):
            if boom: break
            print "Iteration :",j, "...",
            data = open(RFile,"r")
            for line in data:
                # Get u , m , r
                [u, i, r] = line.split("\t")
                # Numpy indexes array from 0
                u = int(u) - 1
                i = int(i) - 1
                r = float(r)

            	if math.isnan(numpy.dot(Q[i,:], P[u,:])):
            		errorTxt =  "(Iteration %s :Nan at u: %s i: %s r: %s )" % (j,u,i,r)
                	boom = True
					#raise Exception(errorTxt)

            	# Calculate Error
            	e = r - numpy.dot(Q[i,:], P[u,:]) 

            	#Update q
            	tmp_q = Q[i,f] + eta*(e*P[u,f] - lam*Q[i,f])

             	#Update p
             	tmp_p = P[u,f] + eta*(e*Q[i,f] - lam*P[u,f])
            	
            	Q[i,f] = tmp_q
            	P[u,f] = tmp_p

            data.close()
            print "done."
        error.append(total_error(P,Q,RFile,lam))
        print "Error:[ %s ]" % (error[f])
    return P, Q, error
        


(u,m) = get_matrix_dims(TRAIN_FILE)
print u,m

print 'Part b'
k = 20
iter = 40
lam = 0.2

eta = 0.03
P = create_random_matrix(u, k)
Q = create_random_matrix(m, k) 
# P1, Q1, e1 = stochastic_gradient_descent(P,Q,TRAIN_FILE,iter,lam,eta)
P1,Q1,e1 = stochastic_gradient_descent2(P,Q,TRAIN_FILE,iter,lam,eta,k)
print P1[:10][:10]
print Q1[:10][:10]

f0 = pylab.figure()
p1 = f0.add_subplot(111)
p1.plot([i for i in xrange(iter)],e1) 
p1.set_xlabel("Iteration")
p1.set_ylabel("Error")
p1.set_title("Gradient Descent : Error vs. Iterations")
pylab.savefig(path+"plot-b.png")