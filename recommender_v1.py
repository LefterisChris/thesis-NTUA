import csv
import numpy as np
import math
import sys

def usage(program):
	print "Please run as: ./%s rank[>0] feedback[0|1] output[0|1]" % program

def get_rating(infile,Mflops):
	ratings = np.zeros(2880)
	with open(infile,"rb") as csvfile:
		myreader = csv.reader(csvfile,delimiter=',')
		myreader.next()
		for (i,row) in enumerate(myreader):
			""" 
			Configuration,grade1,CPI,BW/s((GB/sec),GFLOPS/s,Vectorization,Power(W),Temperature(C),Time(sec)
			initial test: take CPI 
			"""
			conf = row[0][1:-1].split(",")
			threads = int(conf[7][2:-1])
			cores = int(conf[6][2:-1])
			cpi = float(row[2])
			power = float(row[6])
			bw = float(row[3])
			flops = float(row[4])
			rtime = sum(map(float,row[8].strip('[]').split(','))) / 4.0
			# ratings[i] = 1000.0*(threads[i] / cpi[i]) / power[i]
			# ratings[i] = 1000.0*(cpi[i] / threads[i]) / power[i]
			# ratings[i] = (Mflops/rtime[i])
			ratings[i] = 10.0*(Mflops/rtime)/power
	return ratings


def get_permutation():
	permutation = []

	libhugetlbfs = ["","LD_PRELOAD=/home/echristof/hugetlbfs/obj64/libhugetlbfs.so HUGETLB_MORECORE=yes "]
	opts = ["-O2 ", "-O3 "]
	prefetch = ["-opt-prefetch=0 ","-opt-prefetch=2 ","-opt-prefetch=3 ","-opt-prefetch=4 "]
	streaming_stores1 = ["-opt-streaming-stores never ","-opt-streaming-stores always "]
	streaming_stores2 = ["-opt-streaming-cache-evict=0 ","-opt-streaming-cache-evict=1 ",
					"-opt-streaming-cache-evict=2 ","-opt-streaming-cache-evict=3 "]
	unroll = ["-unroll=0 ","-unroll "]

	for f1 in opts:
		for env1 in libhugetlbfs:
			for f2 in prefetch:
				for f3 in streaming_stores1:
					if f3 == "-opt-streaming-stores never ":
						f4 = ""
						for f5 in unroll:
							for affinity in ['scatter','balanced']:
								for cores in [19,38,57]:
									for threads in [2,3,4]:
										permutation.append([f1,f2,f3,f4,f5,env1,str(cores),str(threads),affinity])
					else:
						for f4 in streaming_stores2:
							for f5 in unroll:
								for affinity in ['scatter','balanced']:
									for cores in [19,38,57]:
										for threads in [2,3,4]:
											permutation.append([f1,f2,f3,f4,f5,env1,str(cores),str(threads),affinity])
	return permutation

def get_rating2(infile):
	permutation = get_permutation()
	ratings = np.zeros(2880)

	with open(infile,"rb") as csvfile:
		myreader = csv.reader(csvfile,delimiter=',')
		myreader.next()
		for row in myreader:
			i = 0
			while(str(permutation[i]) != row[0]):
				i += 1

			""" initial test: take CPI """
			ratings[i] = row[3]
			
	csvfile.close()
	return ratings

def get_rating3(infile):
	permutation = get_permutation()
	ratings = np.zeros(2880)

	with open(infile,"rb") as csvfile:
		myreader = csv.reader(csvfile,delimiter=',')
		myreader.next()
		for row in myreader:
			if '-unroll-aggressive' in row[0]:
				continue
			elif '-opt-prefetch=1' in row[0]:
				continue
			i = 0
			but_last = len(row[0]) - 1
			tmp = row[0][:but_last]+", 'balanced']"
			while(str(permutation[i]) != tmp):
				i += 1

			""" initial test: take CPI """
			ratings[i] = row[2]
			
	csvfile.close()
	return ratings

def get_configurations(infile):
	configurations = []
	with open(infile,"rb") as csvfile:
		myreader = csv.reader(csvfile,delimiter=',')
		myreader.next()
		for row in myreader:
			configurations.append(row[0])

	csvfile.close()
	return configurations

def process_newuser(in_user,V,M,rank,learning_rate=0.00005,regularizer=0.0005,MAX_EPOCHS=200):
	new_user_feat = np.array([0.1 for i in xrange(rank)])
	
	for feat in xrange(rank):
		prev_sqrerr = 100.0
		for epochs in xrange(MAX_EPOCHS):
			for j in xrange(M):
				if in_user[j] > 0:
					err = in_user[j] - np.dot(new_user_feat,V[j])
					new_user_feat[feat] += learning_rate*(err*V[j][feat] - regularizer*new_user_feat[feat])
			sqr_err=0.0
			for j in xrange(M):
				if in_user[j] > 0:		
					sqr_err += pow(in_user[j] - np.dot(new_user_feat,V[j]),2)
			sqr_err += regularizer * np.linalg.norm(new_user_feat)
			
			if(abs(sqr_err - prev_sqrerr) < 1e-4):
				#print "New User: Epochs=%d feat=%d "%(epochs,feat)
				break;
			prev_sqrerr = sqr_err

	return new_user_feat

def find_minimum_rank(s):
	# retain a percentage of the energy
	energy = 0.0
	rank = 0
	total_energy = sum(pow(s,2))
	while energy < 0.999:
		rank += 1
		energy = sum(pow(s[:rank],2)) / total_energy
	
	return rank 


if __name__ == '__main__':
	
	if len(sys.argv) < 4:
		usage(sys.argv[0])
		sys.exit(1)
	else:
		rank = int(sys.argv[1])
		if rank <= 0:
			usage(sys.argv[0])
			sys.exit(1)
		feedback = int(sys.argv[2])
		if feedback != 0 and feedback != 1:
			usage(sys.argv[0])
			sys.exit(1)
		output = int(sys.argv[3])
		if output != 0 and output != 1:
			usage(sys.argv[0])
			sys.exit(1)

	training_files = [
						("rodinia/lud/lud_8000.csv", 350950.0), ("rodinia/sradv2/srad_v2_6000.csv",151200.0),
						("rodinia/sradv1/srad_v1.csv", 103462.0), ("rodinia/pre_euler/pre_euler3d_097K.csv",168371.0),
						("rodinia/backprop/backprop_4194304.csv",469.8), ("rodinia/hotspot/hotspot_1024.csv", 3144.5),
						("NPB/lu_A.csv",119280.0), ("NPB/mg_C.csv",155700.0), ("NPB/cg_B.csv",54700.0)
					]
	test_files = [
					("rodinia/nn/nn_filelist1G_8.csv",182.4), ("rodinia/kmeans/kmeans.csv",63492.0), 
				 	("rodinia/cfd/cfd_097K.csv", 157347.4), ("rodinia/hotspot3D/hotspot3D_512_4.csv",3770.0), 
					("rodinia/streamcluster/streamcluster.csv",1716.0), ("rodinia/myocyte/myocyte_30_228.csv",2331.2), 
					("rodinia/lavaMD/lavaMD_20.csv", 14720.0), ("rodinia/heartwall/heartwall_30.csv",175.9),
					("NPB/bt_A.csv",168300.0), ("NPB/ft_B.csv",92050.0), ("NPB/sp_A.csv",85000.0)
				]

	data_set = []
	for f in training_files:
		data_set.append(get_rating(f[0],f[1]))

	#configurations = get_configurations(training_files[0])

	A = np.array(data_set)
	N = len(A)
	M = len(A[0]) 
	# print "NxM = %dx%d" % (N,M)
	

	u,s,v = np.linalg.svd(A,full_matrices=False)
	# print "Initial rank A %d" % len(s)
	# print s

	''' find K based on the rule retain minimum 90% of Energy '''
	K = rank #find_minimum_rank(s)
	# print "SVD approximate rank = ",K

	# keep rank K of svd
	u = u[:,:K]
	s = s[:K]
	v = v[:K]
	# u: NxK , s: KxK, v: KxM 
	#print s

	# transform in UV form, U:NxK, V:KxM
	U = np.dot(u,np.diag(s**0.5))
	V = np.dot(np.diag(s**0.5),v)

	# Calculate RMSE of the rank-k SVD
	# sqr_error = 0.0
	# for i in xrange(N):
	# 	for j in xrange(M):
	# 		prediction = np.dot(U[i],V.T[j])
	# 		sqr_error += pow(A[i][j]-prediction,2)
	# rmse = math.sqrt(sqr_error/(N*M))
	# print "RMSE from rank-%d SVD: %f"%(K,rmse)


	''' test a single user '''	
	# test_user = get_rating(test_files[0])
	# train_user = np.zeros(M).astype(float)
	# train_size = 0
	# for i in xrange(M):
	# 	die = np.random.random(1)
	# 	if die < 0.2:
	# 		train_user[i] = test_user[i]
	# 		train_size += 1

	# print "Percentage of test user's train size : %.2f %% " % ((float(train_size)/M)*100.0)

	# New_User_fspace = process_newuser(train_user,V.T,M,K)

	# sqr_error = 0.0
	# test_size = M - train_size
	
	# for j in xrange(M):
	# 	if train_user[j] == 0.0:
	# 		prediction = np.dot(New_User_fspace,V.T[j])
	# 		sqr_error += pow(test_user[j]-prediction,2)

	# rmse = math.sqrt(sqr_error/test_size)
	# print "SVD: RMSE for test user= %f" % rmse

	# sys.exit(0)

	'''
	Testing new incoming users recommendations 
	based on our training set.
	'''
	for test_file in test_files:
		test_user = get_rating(test_file[0],test_file[1])

		train_user = np.zeros(M).astype(float)
		train_size = 0
		for i in xrange(M):
			die = np.random.random(1)
			if die < 0.1:
				train_user[i] = test_user[i]
				train_size += 1

		# print "Percentage of test user's train size : %.2f %% " % ((float(train_size)/M)*100.0)

		'''
			Main processing of the new user
		'''
		New_User_fspace = process_newuser(train_user,V.T,M,K)

		
		sqr_error = 0.0
		test_size = M - train_size
		for j in xrange(M):
			if train_user[j] == 0.0:
				prediction = np.dot(New_User_fspace,V.T[j])
				sqr_error += pow(test_user[j]-prediction,2)

		rmse = math.sqrt(sqr_error/test_size)
		# print "SVD: RMSE for test user= %f" % rmse
		print "%.6f" % rmse

		if output:
			'''
			Output rates to file.
			'''
			file = 'recommender_'+test_file.split('/')[1]
			with open('evaluation/'+file+'_'+str(feedback)+'.csv','a') as fp:
				for j in xrange(M):
					prediction = np.dot(New_User_fspace,V.T[j])
					fp.write(str(j)+','+str(test_user[j])+','+str(prediction)+'\n')

		if feedback:
			'''
			Add feedback - reconstruct our training set using derived user space.
			'''
			newU = np.vstack((U,New_User_fspace))
			newA = np.dot(newU,V)
			# newA = np.vstack((A,test_user))
			u,s,v = np.linalg.svd(newA)
			K = find_minimum_rank(s)
			#print "SVD approximate rank = ",K
			u = u[:,:K]
			s = s[:K]
			v = v[:K]
			#print s
			U = np.dot(u,np.diag(s**0.5))
			V = np.dot(np.diag(s**0.5),v)
		

	
	'''
	Use half profiled benchmarks
	'''
	# sten2d9pt = get_rating2("sten2d9pt/sten2d9pt_pad.csv")
	# train_size = np.count_nonzero(sten2d9pt)
	# print "Percentage of train size for sten2d9pt: %.2f %% " % ((float(train_size)/M)*100.0)

	# sten2d9pt_fspace = process_newuser(sten2d9pt,V.T,M,K)

	# sqr_error = 0.0
	# test_size = train_size
	
	# for j in xrange(M):
	# 	if sten2d9pt[j] > 0.0:
	# 		prediction = np.dot(sten2d9pt_fspace,V.T[j])
	# 		sqr_error += pow(sten2d9pt[j]-prediction,2)

	# rmse = math.sqrt(sqr_error/test_size)
	# print "SVD: RMSE for sten2d9pt= %f" % rmse

	# BS_pricing = get_rating3("BS_pricing_omp.csv")
	# train_size = np.count_nonzero(BS_pricing)
	# print train_size
	# print "Percentage of train size for BS pricing: %.2f %% " % ((float(train_size)/M)*100.0)

	# BS_pricing_fspace = process_newuser(BS_pricing,V.T,M,K)

	# sqr_error = 0.0
	# test_size = train_size
	
	# for j in xrange(M):
	# 	if BS_pricing[j] > 0.0:
	# 		prediction = np.dot(BS_pricing_fspace,V.T[j])
	# 		sqr_error += pow(BS_pricing[j]-prediction,2)

	# rmse = math.sqrt(sqr_error/test_size)
	# print "SVD: RMSE for BS pricing= %f" % rmse