import numpy as np
import math

def matrix_factorization(A,U,V,K,alpha=0.001,beta=0.02,max_epochs=120):
	V = V.T
	N = len(A)
	M = len(A[0])
	rmse = 1.0
	
	''' Training from the non zero values '''
	e = 0
	for i in xrange(N):
		for j in xrange(M):
			if A[i][j] > 0:
				for l in xrange(max_epochs):
					eij = A[i][j] - np.dot(U[i,:],V[:,j])
					for k in xrange(K):
						U[i][k] = U[i][k] + alpha * (2*eij * V[k][j] - beta * U[i][k])
						V[k][j] = V[k][j] + alpha * (2*eij * U[i][k] - beta * V[k][j])
				e += pow(A[i][j] - np.dot(U[i,:],V[:,j]),2)

	tmp = 0
	for k in xrange(K):
		tmp += sum(pow(U[:][k],2))+sum(pow(V[k][:],2))
	e = e + (beta/2)*tmp
	
	rmse = math.sqrt(e/np.count_nonzero(A))

	print "RMSE=%.8f" % rmse
	return U,V.T			

def recommendation_svd_users(A,u,s,v,nUsr):
	pred = []
	for user in nUsr:
		new_uspace = np.dot(np.dot(user,v.T),np.diag(s**-1))
		#u = np.vstack((u,new_uspace))
		print "projection in concept space: ", new_uspace
		
		u_sims = np.zeros(len(u))
		for i in xrange(len(u)):
			u_sims[i] = similarity_fun(new_uspace,u[i])
		
		print "User similarity vector ", u_sims

		avg_rates = [sum(usr)/len(usr) for usr in A]
		print avg_rates
		#idx_max_2 = np.argsort(u_sims)[::-1][:2]
		for (idx,rate) in enumerate(user):
			if rate == 0:
				other_rates = [i[idx] for i in A]
				user[idx] = sum(np.subtract(other_rates,avg_rates) * u_sims) / sum(u_sims)
		pred.append(user)
	return pred
			
def similarity_fun(u1,u2):
	#cosine similarity
	return (np.dot(u1,u2) / (np.linalg.norm(u1) * np.linalg.norm(u2)))

def factorization2(A,U,V,rank,non_z,max_epochs=100):
	N = len(A)
	M = len(A[0])
	V = V.T
	rmse = 100
	for epochs in xrange(max_epochs):
		for s in xrange(rank):
			for d1 in xrange(N):
				tmp11 = 0.0
				tmp12 = 0.0
				for j in xrange(M):
					if A[d1][j] > 0:
						tmp11 += V[s][j]*(A[d1][j] - (np.dot(U[d1],V.T[j]) - U[d1][s]*V[s][j]))
						tmp12 += pow(V[s][j],2)
				U[d1][s] = tmp11 / tmp12
				
			for d2 in xrange(M):
				tmp21 = 0.0
				tmp22 = 0.0
				for i in xrange(N):
					if A[i][d2] > 0:
						tmp21 += U[i][s]*(A[i][d2] - (np.dot(U[i],V.T[d2])- U[i][s]*V[s][d2]))
						tmp22 += pow(U[i][s],2)
				V[s][d2] = tmp21 / tmp22
		
		#calculate rmse for early break
		prev_rmse = rmse
		rmse = 0.0
		for i in xrange(N):
			for j in xrange(M):
				if A[i][j] > 0:
					rmse += pow(A[i][j] - np.dot(U[i],V.T[j]), 2)
		rmse = math.sqrt(rmse/non_z)
		if (abs(rmse - prev_rmse) < 1e-5):
			print prev_rmse,rmse,epochs
			break
		
	#returns V transposed
	return U,V,rmse

if __name__ == '__main__':
	
	R = [[5.,2.,4.,4.,3.],[3.,1.,2.,4.,1.],[2.,0.,3.,1.,4.,],[2.,5.,4.,3.,5.,],[4.,4.,5.,4.,0.]]
	#R =[[5.,3.,0.,1.],[4.,0.,0.,1.],[1.,1.,0.,5.],[1.,0.,0.,4.],[0.,1.,5.,4.]]
	A = np.array(R)
	#A = np.random.randint(0,6,(100,100)).astype(float)
	# A = np.array([[0., 2., 2., 0., 5., 4., 3., 1., 2., 5.],
 # 				[5., 5., 2., 0., 1., 3., 0., 0., 0., 5.],
 # 				[4., 3., 0., 5., 4., 2., 1., 1., 2., 3.],
 # 				[3., 2., 1., 5., 3., 2., 4., 4., 0., 3.],
 # 				[2., 2., 4., 1., 3., 5., 4., 0., 4., 1.],
 # 				[3., 5., 0., 4., 5., 1., 3., 0., 4., 1.],
 # 				[0., 2., 0., 1., 3., 4., 0., 3., 2., 1.],
 # 				[1., 4., 0., 0., 5., 1., 3., 0., 4., 4.],
 # 				[2., 4., 0., 5., 3., 1., 5., 0., 2., 4.],
 # 				[2., 0., 0., 5., 4., 5., 5., 4., 5., 4.]])
	
	K = 3
	
	U = np.empty((len(A),K))
	V = np.empty((len(A[0]),K))
	init_val = math.sqrt(sum(sum(A)) / float(np.count_nonzero(A)) / K)
	
	# U[:] = init_val
	# V[:] = init_val

	# nU,nV = matrix_factorization(A,U,V,K)

	# predA = np.dot(nU,nV.T)
	# print "Using UV factorization:\n", predA
	# print "\n"
	# print "My array: \n",A

	# rmse = 0.0
	# for i in xrange(len(A)):
	# 	for j in xrange(len(A[0])):
	# 		if A[i][j] > 0:
	# 			rmse += pow(A[i][j] - predA[i][j], 2)

	# rmse = math.sqrt(rmse/np.count_nonzero(A))
	# print "RMSE = ",rmse

	U[:]=init_val
	V[:]=init_val

	non_zero = np.count_nonzero(A)

	U2,V2,rmse = factorization2(A,U,V,K,non_zero)
	predA2 = np.array([[round(i,3) for i in j] for j in np.dot(U2,V2)])
	print predA2
	rmse2 = 0.0
	for i in xrange(len(A)):
		for j in xrange(len(A[0])):
			if A[i][j] > 0:
				rmse2 += pow(A[i][j] - predA2[i][j], 2)
	
	rmse2 = math.sqrt(rmse2/np.count_nonzero(A))
	print "RMSE2 = %f , RMSE = %f" % (rmse2,rmse)
	# A2 = np.vstack((A[0],A[1],A[3]))
	# newUsers = np.vstack((A[2],A[4]))
	# u,s,v = np.linalg.svd(A2)
	# k=len(s)
	# preds = recommendation_svd_users(A2,u[:,:k],s[:k],v[:k,:],newUsers)
	# predA = np.vstack((A2,preds))
	# print "Using similarities: \n", predA

	






