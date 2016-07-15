import numpy as np
import math
import sys

def matrix_factorization(training_set,init_val,N,M,rank,learning_rate=0.002,regularizer=0.02,MAX_EPOCHS=100):
	training_size = len(training_set)
	# ufeats n x rank
	ufeats = np.array([[init_val for i in range(rank)] for i in range(N)])
	# ifeats m x rank
	ifeats = np.array([[init_val for i in range(rank)] for i in range(M)])

	for feat in xrange(rank):
		prev_sqrerr = 100.0
		for epochs in xrange(MAX_EPOCHS):
			sqr_err = 0.0
			for i in xrange(training_size):
				user = training_set[i][0] - 1
				item = training_set[i][1] - 1
				
				err = training_set[i][2] - np.dot(ufeats[user],ifeats[item])

				ufeats[user][feat] += learning_rate*(err*ifeats[item][feat] - regularizer*ufeats[user][feat])
				ifeats[item][feat] += learning_rate*(err*ufeats[user][feat] - regularizer*ifeats[item][feat])

				sqr_err += pow(training_set[i][2] - np.dot(ufeats[user],ifeats[item]),2) 

			ufeatsL2norm = sum([np.dot(i,i) for i in ufeats])
			ifeatsL2norm = sum([np.dot(i,i) for i in ifeats])
			sqr_err += regularizer*(ufeatsL2norm + ifeatsL2norm)

			if(abs(sqr_err - prev_sqrerr) < 1e-5):
				#print "Training: Epochs=%d feat=%d "%(epochs,feat)
				break;
			prev_sqrerr = sqr_err
		#print ufeats
		#print ifeats
		#raw_input()
	return ufeats,ifeats

def PARAGON(training_set,rank,U,V,learning_rate=0.002,regularizer=0.02,MAX_EPOCHS=200):
	training_size = len(training_set)

	for feat in xrange(rank):
		prev_sqrerr = 100.0
		for epochs in xrange(MAX_EPOCHS):
			sqr_err = 0.0
			for i in xrange(training_size):
				user = training_set[i][0]
				item = training_set[i][1]
				
				err = training_set[i][2] - np.dot(U[user],V[item]) 

				U[user][feat] += learning_rate*(err*V[item][feat] - regularizer*U[user][feat])
				V[item][feat] += learning_rate*(err*U[user][feat] - regularizer*V[item][feat])
				sqr_err += pow(training_set[i][2] - np.dot(U[user],V[item]),2) 

			U_L2norm = sum([np.dot(i,i) for i in U])
			V_L2norm = sum([np.dot(i,i) for i in V])
			sqr_err += regularizer*(U_L2norm + V_L2norm)

			if(abs(sqr_err - prev_sqrerr) < 1e-5):
				#print "Training: Epochs=%d feat=%d "%(epochs,feat)
				break;
			prev_sqrerr = sqr_err
		#print ufeats
		#print ifeats
		#raw_input()
	return U,V.T

def process_newuser(in_user,V,M,rank,learning_rate=0.002,regularizer=0.002,MAX_EPOCHS=200):
	new_user_feat = np.array([0.1 for i in xrange(rank)])
	
	for feat in xrange(rank):
		prev_sqrerr = 100.0
		for epochs in xrange(MAX_EPOCHS):
			sqr_err=0.0
			for j in xrange(M):
				if in_user[j] > 0:
					err = in_user[j] - np.dot(new_user_feat,V[j])
					sqr_err += pow(err,2)

					new_user_feat[feat] += learning_rate*(err*V[j][feat] - regularizer*new_user_feat[feat])
					#V[j][feat] += learning_rate*(err*V[j][feat] - regularizer*new_user_feat[feat])
			if(abs(sqr_err - prev_sqrerr) < 1e-5):
				#print "New User: Epochs=%d feat=%d "%(epochs,feat)
				break;
			prev_sqrerr = sqr_err

	return new_user_feat,V

def find_minimum_rank(s):
	# retain a percentage of the energy
	energy = 0.0
	rank = 0
	total_energy = sum(pow(s,2))
	while energy < 0.9:
		rank += 1
		energy = sum(pow(s[:rank],2)) / total_energy
	
	return rank 

def movielens(N,M):
	training_set = []
	training_size = 0	
	fp = open("ml-100k/u.data","r")
	while(True):
		line = fp.readline()
		if not line:
			break
		else:
			line = line.split('\t')
			if (int(line[0]) <= N) and (int(line[1]) <= M):
				training_set.append((int(line[0]),int(line[1]),float(line[2])))
				training_size += 1
	fp.close()

	test_set = []
	test_size = 0
	fp = open("ml-100k/u1.test","r")
	while(True):
		line = fp.readline()
		if not line:
			break
		else:
			line = line.split("\t")
			if (int(line[0]) <= N) and (int(line[1]) <= M):
				test_set.append((int(line[0]),int(line[1]),float(line[2])))
				test_size += 1
	fp.close()

	return training_set,training_size,test_set,test_size


if __name__ == '__main__':

	if len(sys.argv) < 2:
		#print "run %s and give rank" % sys.argv[0]
		#sys.exit(1)
		pass
	else:
		#rank given by the user
		K = int(sys.argv[1])


	#A = np.array([[5.,2.,4.,4.,3.],[3.,1.,2.,4.,1.],[2.,0.,3.,1.,4.,],[2.,5.,4.,3.,5.,],[4.,4.,5.,4.,0.]])
	#A =np.array([[5.,3.,0.,1.],[4.,0.,0.,1.],[1.,1.,0.,5.],[1.,0.,0.,4.],[0.,1.,5.,4.]])
	N = 10
	M = 10
	A = np.zeros((N,M))
	training_set = []
	training_size = 0
	for i in xrange(N):
		if i == N-1:
			training_percent = 0.4
		else:
			training_percent = 1.0
		for j in xrange(M):
			die = np.random.rand()
			if (die < training_percent):
				rate = float(np.random.randint(1,6))
				training_set.append((i,j,rate))
				A[i][j] = rate
				training_size += 1
			else:
				training_set.append((i,j,0.0))
				A[i][j] = 0.0


	u,s,v = np.linalg.svd(A,full_matrices=0)

	# find K based on the rule "retain minimum 90% of Energy"
	K = find_minimum_rank(s)
	print "SVD approximate rank = ",K
	
	# keep rank K of svd
	u = u[:,:K]
	s = s[:K]
	v = v[:K]

	U1 = np.dot(u,np.diag(s**0.5))
	V1 = np.dot(np.diag(s**0.5),v)
	R1 = np.dot(U1,V1)
	
	R1 = np.array([[round(j,3) for j in i] for i in R1])
	print R1

	U2,V2 = PARAGON(training_set,K,U1,V1.T)

	R2 = np.dot(U2,V2)
	R2 = np.array([[round(j,3) for j in i] for i in R2])
	print R2

	raw_input()

	#print "SVD"
	#print R1

	#training_set,training_size,test_set,test_size = movielens(N,M)

	#init_val = math.sqrt((sum(sum(A))/float(np.count_nonzero(A))) / K)
	init_val = 0.1

	# UV decomposition
	# U2,V2 = matrix_factorization(training_set,init_val,N,M,K)
	# R2 = np.dot(U2,V2.T)
	# R2 = np.array([[round(j,3) for j in i] for i in R2])
	#print "UV decomposition"
	#print R2

	# sqr_error = 0.0
	# for original in test_set:
	# 	prediction = np.dot(U[original[0]-1],V[original[1]-1])
	# 	sqr_error += pow(original[2] - prediction,2)
	# rmse = math.sqrt(sqr_error/test_size)
	# print rmse

	sqr_error1 = 0.0
	#sqr_error2 = 0.0
	for i in xrange(N):
		for j in xrange(M):
			prediction1 = np.dot(U1[i],V1.T[j])
	#		prediction2 = np.dot(U2[i],V2[j]) 
			sqr_error1 += pow(A[i][j]-prediction1,2)
	#		sqr_error2 += pow(A[i][j]-prediction2,2)
	rmse1 = math.sqrt(sqr_error1/(N*M))
	print "RMSE from SVD: %f"%rmse1	
	#rmse2 = math.sqrt(sqr_error2/(N*M))
	#print "RMSE from UV decomposition: %f"%rmse2

	known_usr = N-1
	new_user = A[known_usr]#np.random.randint(1,6,M).astype(float)

	in_user = np.zeros(M).astype(float)
	train_size = 0
	for i in xrange(M):
		die = np.random.random(1)
		if die < 0.7:
			in_user[i] = new_user[i]
			train_size += 1

	
	print "Percentage of train size for new user: %.2f %% " % ((float(train_size)/M)*100.0)


	new_user_feat,V_tmp = process_newuser(in_user,V1.T,M,K)

	#print np.dot(new_user_feat,V1)

	sqr_error_u = 0.0
	u_test_size = M - train_size
	
	for j in xrange(M):
		if in_user[j] == 0.0:
			prediction_u = np.dot(new_user_feat,V1.T[j])
			sqr_error_u += pow(new_user[j]-prediction_u,2)
		#print "Original: %f \t Prediction: %f %f " % (new_user[j],prediction_u,np.dot(U1[known_usr],V1.T[j])) + str(known)

	rmse_u = math.sqrt(sqr_error_u/u_test_size)
	print "SVD: RMSE for new user= %f"%rmse_u

	# for j in xrange(M):
	# 	if in_user[j] > 0.0:
	# 		print "Difference %f" % (np.dot(U1[known_usr],V1.T[j])-np.dot(U1[known_usr],V_tmp[j]))

	# new_user_feat2 = process_newuser(in_user,V2,M,K)

	# print np.dot(new_user_feat2,V2.T)

	# sqr_error_u = 0.0
	# u_training_size = 0
	# for j in xrange(M):
	# 	if in_user[j] > 0:
	# 		prediction_u = np.dot(new_user_feat2,V2[j])
	# 		sqr_error_u += pow(in_user[j]-prediction_u,2)
	# 		u_training_size += 1
	# rmse_u = math.sqrt(sqr_error_u/u_training_size)
	# print "UV: RMSE for new user= %f"%rmse_u
