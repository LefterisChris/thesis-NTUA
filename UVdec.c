#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "UVdec.h"

#ifndef MAX_EPOCHS
#define MAX_EPOCHS 200
#endif

#define RD_BYTES 512

#define INIT_VAL 0.1

REAL l1,l2,l3,l4,r1,r2,r3,r4;
float TRAINING_PERCENT;

typedef struct {
	int u;
	int i;
	REAL rate;
}data_struct;

int usage(char *name){
	fprintf(stderr, "Please run: %s rank [>0] training percent (0.0,1.0) feedback [0|1]\n",name);
	exit(1);
}

void *safe_malloc(size_t size){
	void *p;
	if ((p = malloc(size)) == NULL) {
		fprintf(stderr, "Out of memory, failed to allocate %zd bytes\n",size);
		exit(1);
	}
	return p;
}

REAL dot_product(REAL **X, REAL **Y, int row, int col, int start, int end){
	int k;
	REAL tmp = 0.0;
	for(k=start; k < end; k++)
		tmp += X[row][k]*Y[k][col];
	return tmp;
}

REAL summedsquaredL2norms(REAL **M, int rows, int cols, int transposed){
	int i,j;
	REAL tmp1 = 0.0, tmp2;
	/* 1 for transposed, 0 for normal */
	if(transposed)
		for(j=0;j<cols;j++){
			tmp2 = 0.0;
			for(i=0;i<rows;i++)
				tmp2 += pow(M[i][j],2);
			tmp1 += tmp2;
		}
	else
		for(i=0;i<rows;i++){
			tmp2 = 0.0;
			for(j=0;j<cols;j++)
				tmp2 += pow(M[i][j],2);
			tmp1 += tmp2;
		}
	return tmp1;
}

REAL summedsquaredL2norms2(REAL *M, int size){
	int i;
	REAL tmp = 0.0;
	for(i=0;i<size;i++)
		tmp += pow(M[i],2);
	return tmp;
}

REAL rating_function(REAL cpi, int cores, int threads, REAL power, REAL flops, REAL bw, REAL rtime, REAL base_rate, REAL Mflops){
	REAL rate1 = 1000.0*(((REAL)threads)/cpi)/power; //ipc per watt
	// REAL rate2 = (10.0*(Mflops/rtime)/power);
	REAL rate2 = 10*(base_rate / (power*rtime)); // MFlops/sec / Watt normalized
	REAL rate3 = 100*(base_rate/rtime);
	return rate1;
}

REAL get_base_rate(char *bfile){
	FILE *fp;
	REAL base_rate;
	char buf[RD_BYTES], *tmp, *rest, tmp2[32];
	int option;
	REAL pwr, rtime;

	fp = fopen(bfile,"r");
	if (fp == NULL){
		fprintf(stderr, "Error opening file %s\n",bfile);
		exit(1);
	}
	/* Ommit the first line */
	if(fgets(buf,RD_BYTES,fp) == NULL){
		fprintf(stderr, "Error reading file %s\n",bfile);
		exit(1);
	}
	/* Get the second line with the metrics */
	if(fgets(buf,RD_BYTES,fp) == NULL){
		fprintf(stderr, "Error reading file %s\n",bfile);
		exit(1);
	}

	tmp = strtok_r(buf,",",&rest);
	option = 1;
	pwr = rtime = 0.0;
	while(tmp){
		if (option == 7)
			pwr = strtod(tmp,NULL);
		if (option == 8)
			break; // parse time 
		option++;
		tmp = strtok_r(NULL,",",&rest);
	};
	sscanf(rest," \"[%[^]]\"",tmp2);
	tmp = strtok_r(tmp2,",",&rest);
	while(tmp){
		rtime += strtod(tmp,NULL);
		tmp = strtok_r(NULL,",",&rest);
	};
	rtime = rtime / 4.0;

	fclose(fp);
	return (pwr*rtime);
	// return rtime;
}

REAL **get_data(int N, int M){
	int i,j,option;
	FILE *fp;
	char buf[RD_BYTES];
	char *tmp,*conf,*rest,tmp2[32];
	REAL cpi,pwr,flops,bw,rtime, base_rate;
	int cores, threads;
	size_t conf_len;
	REAL **A;
	
	A = safe_malloc(N * sizeof(*A));
	for(i=0;i<N;i++)
		A[i] = safe_malloc(M*sizeof(**A));

	for(i=0; i<N; i++){
		base_rate = get_base_rate(training_base[i].base);

		fp = fopen(training_base[i].file,"r");
		if (fp == NULL){
			fprintf(stderr, "Error opening file %s\n",training_base[i].file);
			exit(1);
		}
		/* Ommit the first line */
		if(fgets(buf,RD_BYTES,fp) == NULL){
			fprintf(stderr, "Error reading file %s\n",training_base[i].file);
			exit(1);
		}
		j = 0;
		while(fgets(buf,RD_BYTES,fp)) {
			/* format: 
			 * Configuration[9 options],grade1,CPI,BW/s((GB/sec),GFLOPS/s,Vectorization,Power(W),Temperature(C),Time(sec) 
			 * Configuration: "['-O2 ', '-opt-prefetch=0 ', '-opt-streaming-stores always ', '-opt-streaming-cache-evict=1 ', '-unroll=0 ', 'LD_PRELOAD=/home/echristof/hugetlbfs/obj64/libhugetlbfs.so HUGETLB_MORECORE=yes ', '57', '4', 'scatter']"
			 */
			
			sscanf(buf," \"[%m[^]\"]",&conf); //sscanf(buf," \"[%*[^]\"]");
			conf_len = strlen(conf);
			tmp = strtok_r(conf,",",&rest);
			/* get cores(6) and threads(7) */
			option = 1;
			while(tmp) {
				sscanf(tmp," '%[^']'",tmp2); 
				if (option == 7)
					cores = (int) strtol(tmp2, NULL, 10);
				if (option == 8)
					threads = (int) strtol(tmp2, NULL, 10); 
				option++;
				tmp = strtok_r(NULL,",",&rest);
				memset(tmp2,'\0',sizeof(tmp2));
			};
			
			tmp = strtok_r(buf+conf_len+5,",",&rest);
			option = 1;
			cpi = pwr = flops = bw = rtime = 0.0;
			while(tmp){
				if (option == 2)
					cpi = strtod(tmp,NULL);
				if (option == 3)
					bw = strtod(tmp,NULL);
				if (option == 4)
					flops = strtod(tmp,NULL);
				if (option == 6)
					pwr = strtod(tmp,NULL);
				if (option == 7)
					break; // parse time 
				option++;
				tmp = strtok_r(NULL,",",&rest);
			};
			sscanf(rest," \"[%[^]]\"",tmp2);
			tmp = strtok_r(tmp2,",",&rest);
			while(tmp){
				rtime += strtod(tmp,NULL);
				tmp = strtok_r(NULL,",",&rest);
			};
			rtime = rtime / 4.0;
			
			if ((cpi < 1e-10) || (pwr < 1e-10) || (flops < 1e-10) || (bw < 1e-10) || (rtime < 1e-10)){
				fprintf(stderr, "Error in parsing. Zero value cpi or power. Exiting...\n");
				exit(1);
			}else
				A[i][j] = rating_function(cpi,cores,threads,pwr,flops,bw,rtime,base_rate,training_base[i].Mflops); 
			
			j++;
			free(conf);
		}
		fclose(fp);
	}
	return A;
}

REAL *get_new_user(int usr_idx, int M){
	int j,option;
	FILE *fp;
	char buf[RD_BYTES];
	char *tmp,*conf,*rest,tmp2[32];
	REAL cpi,pwr,flops,bw,rtime,base_rate;
	int cores, threads;
	size_t conf_len;
	REAL *A;
	
	A = safe_malloc(M*sizeof(*A));

	base_rate = get_base_rate(test_base[usr_idx].base);

	fp = fopen(test_base[usr_idx].file,"r");
	if (fp == NULL){
		fprintf(stderr, "Error opening file %s\n",test_base[usr_idx].file);
		exit(1);
	}
	/* Ommit the first line */
	if(fgets(buf,RD_BYTES,fp) == NULL){
		fprintf(stderr, "Error reading file %s\n",test_base[usr_idx].file);
		exit(1);
	}
	j = 0;
	while(fgets(buf,RD_BYTES,fp)) {
		/* format: 
		 * Configuration[9 options],grade1,CPI,BW/s((GB/sec),GFLOPS/s,Vectorization,Power(W),Temperature(C),Time(sec) 
		 */
		sscanf(buf," \"[%m[^]\"]",&conf); //sscanf(buf," \"[%*[^]\"]");
		conf_len = strlen(conf);
		tmp = strtok_r(conf,",",&rest);
		/* get cores(6) and threads(7) */
		option = 1;
		while(tmp) {
			sscanf(tmp," '%[^']'",tmp2); 
			if (option == 7)
				cores = (int) strtol(tmp2, NULL, 10);
			if (option == 8)
				threads = (int) strtol(tmp2, NULL, 10); 
			option++;
			tmp = strtok_r(NULL,",",&rest);
			memset(tmp2,'\0',sizeof(tmp2));
		};

		tmp = strtok_r(buf+conf_len+5,",",&rest);
		option = 1;
		cpi = pwr = flops = bw = rtime =0;
		while(tmp){
			if (option == 2)
				cpi = strtod(tmp,NULL);
			if (option == 3)
				bw = strtod(tmp,NULL);
			if (option == 4)
				flops = strtod(tmp,NULL);
			if (option == 6)
				pwr = strtod(tmp,NULL);
			if (option == 7)
				break; // parse time 
			option++;
			tmp = strtok_r(NULL,",",&rest);
		};
		sscanf(rest," \"[%[^]]\"",tmp2);
		tmp = strtok_r(tmp2,",",&rest);
		while(tmp){
			rtime += strtod(tmp,NULL);
			tmp = strtok_r(NULL,",",&rest);
		};
		rtime = rtime / 4.0;
		
		if ((cpi < 1e-10) || (pwr < 1e-10) || (flops < 1e-10) || (bw < 1e-10) || (rtime < 1e-10)){
			fprintf(stderr, "Error in parsing. Zero value cpi or power. Exiting...\n");
			exit(1);
		}else
			A[j] = rating_function(cpi,cores,threads,pwr,flops,bw,rtime,base_rate,test_base[usr_idx].Mflops); 
		
		j++;
		free(conf);
	}
	fclose(fp);
	return A;
}
/* for UV factorization */
void get_training_test_data(REAL **A, int N, int M, data_struct **training_set, int *training_size, data_struct **test_set, int *test_size) {
	int i,j;
	float training_percent = 0.2;
	data_struct *tmp_training, *tmp_test;
	tmp_training = tmp_test = NULL;
	
	*training_size = *test_size = 0;
	srand(time(NULL));
	for(i = 0; i<N; i++) {
		//training_percent = (i == 0) ? 0.1 : 1.0;
		for(j=0;j<M; j++)
			if(((float)rand() / RAND_MAX) < training_percent){
				(*training_size)++;
				tmp_training = realloc(*training_set,(*training_size)*sizeof(data_struct));
				if(!tmp_training){
					fprintf(stderr, "Error (re)allocating memory! Exiting...\n");
					exit(1);
				}
				*training_set = tmp_training;
				(*training_set)[*training_size - 1].u = i;
				(*training_set)[*training_size - 1].i = j;
				(*training_set)[*training_size - 1].rate = A[i][j];
			}
			else{
				(*test_size)++;
				tmp_test = realloc(*test_set,(*test_size)*sizeof(data_struct));
				if(!tmp_test){
					fprintf(stderr, "Error (re)allocating memory! Exiting...\n");
					exit(1);
				}
				*test_set = tmp_test;
				(*test_set)[*test_size - 1].u = i;
				(*test_set)[*test_size - 1].i = j;
				(*test_set)[*test_size - 1].rate = A[i][j];
			}
	}
}

void get_training_data(REAL **A, int N, int M, data_struct **training_set, int *training_size){
	int i,j;
	data_struct *tmp_training;
	tmp_training = NULL;
	
	*training_size = 0;
	for(i = 0; i<N; i++)
		for(j=0;j<M; j++){
			(*training_size)++;
			tmp_training = realloc(*training_set,(*training_size)*sizeof(data_struct));
			if(!tmp_training){
				fprintf(stderr, "Error (re)allocating memory! Exiting...\n");
				exit(1);
			}
			*training_set = tmp_training;
			(*training_set)[*training_size - 1].u = i;
			(*training_set)[*training_size - 1].i = j;
			(*training_set)[*training_size - 1].rate = A[i][j];
		}
}

void update_training_set_get_test_set(REAL *A, int num_new_user, int M, data_struct **training_set, int *training_size, data_struct **test_set, int *test_size){
	
	int j;
	// float training_percent = 0.1;
	data_struct *tmp_training, *tmp_test;
	tmp_training = tmp_test = NULL;
	
	//srand(42);
	srand(time(NULL));
	(*test_size) = 0;
	for(j=0;j<M; j++)
		if(((float)rand() / RAND_MAX) < TRAINING_PERCENT){
			(*training_size)++;
			tmp_training = realloc(*training_set,(*training_size)*sizeof(data_struct));
			if(!tmp_training){
				fprintf(stderr, "Error (re)allocating memory! Exiting...\n");
				exit(1);
			}
			*training_set = tmp_training;
			(*training_set)[*training_size - 1].u = num_new_user;
			(*training_set)[*training_size - 1].i = j;
			(*training_set)[*training_size - 1].rate = A[j];
		}
		else{
			(*test_size)++;
			tmp_test = realloc(*test_set,(*test_size)*sizeof(data_struct));
			if(!tmp_test){
				fprintf(stderr, "Error (re)allocating memory! Exiting...\n");
				exit(1);
			}
			*test_set = tmp_test;
			(*test_set)[*test_size - 1].u = num_new_user;
			(*test_set)[*test_size - 1].i = j;
			(*test_set)[*test_size - 1].rate = A[j];
		}
}

void update_training_set(data_struct **training_set, int *training_size, REAL **U, REAL **Vt, int rank, REAL *bu, REAL *bi, REAL glb_avg, data_struct *test_set, int test_size){
	
	int j;
	data_struct *tmp_training;
	tmp_training = NULL;
	REAL pred;

	for(j=0;j<test_size;j++){
		(*training_size)++;
		pred = glb_avg + bu[test_set[j].u] + bi[test_set[j].i] + dot_product(U,Vt,test_set[j].u,test_set[j].i,0,rank);
		tmp_training = realloc(*training_set,(*training_size)*sizeof(data_struct));
		if(!tmp_training){
			fprintf(stderr, "Error (re)allocating memory! Exiting...\n");
			exit(1);
		}
		*training_set = tmp_training;
		(*training_set)[*training_size - 1].u = test_set[j].u;
		(*training_set)[*training_size - 1].i = test_set[j].i;
		(*training_set)[*training_size - 1].rate = pred;
	}
}

void initialize(REAL **U, REAL **Vt, int N, int M, int rank, REAL init_val){
	int i,k;
	/* Initialize U,Vt */
	for(i=0;i<N;i++)
		for(k=0;k<rank;k++)
			U[i][k] = init_val;
	for(k=0;k<rank;k++)
		for(i=0;i<M;i++)
			Vt[k][i] = init_val;
}

/* initial value for UV suggested by Ullman */
REAL calculate_initial_value(data_struct *TS,int size,int rank){
	REAL sum = 0.0;
	int i;
	for(i=0;i<size;i++)
		sum += TS[i].rate;
	return sqrt((sum/size)/rank);
}

/* 
 *	Simple implementation: Prediction r[u,i] = Sigma U[u,k]* Vt[k,i] over f  
 */
void matrix_factorization(data_struct *training_set, REAL **U, REAL **Vt, int N, int M, int rank, int training_size){
	REAL lambda = 0.00002, regularizer=0.001;
	int epochs,i,f,user,item;
	REAL error,sq_err,prev_sqerr = 100.0;
	REAL pred, tmp, *cached_prods;

	cached_prods = safe_malloc(N*M*sizeof(*cached_prods));

	for(f=0; f<rank; f++){
		for(epochs=0; epochs < 800; epochs++){
			for(i=0; i<training_size; i++){
				user = training_set[i].u;
				item = training_set[i].i;
				if (f == 0)
					pred = dot_product(U,Vt,user,item,f,rank);
				else
					pred = cached_prods[user*M+item] + dot_product(U,Vt,user,item,f,rank);
				
				error = training_set[i].rate - pred;

				tmp = U[user][f];
				U[user][f] += lambda * (error * Vt[f][item] - regularizer * tmp);
				Vt[f][item] += lambda * (error * tmp - regularizer * Vt[f][item]);
			}
			sq_err = 0.0;
			for(i=0; i<training_size; i++){
				user = training_set[i].u;
				item = training_set[i].i;
				error = training_set[i].rate - dot_product(U,Vt,user,item,0,rank);
				sq_err += pow(error,2);
			}
			if(fabs(sq_err - prev_sqerr) < 1e-5){
				printf("Feat: %d Epochs: %d\n",f,epochs);
				break;
			}
			prev_sqerr = sq_err;
		}
		for(i=0; i<training_size; i++){
			user = training_set[i].u;
			item = training_set[i].i;
			if (f == 0)
				cached_prods[user*M+item] = U[user][f]*Vt[f][item];
			else
				cached_prods[user*M+item] += U[user][f]*Vt[f][item];
		}
	}
	free(cached_prods);
}

/*
 *	SVD++: Prediction r[u,i] = global average + b[u] + b[i] + Sigma U[u,k]* Vt[k,i] over f
 */
void SVD_plus(data_struct *training_set, int training_size, REAL **U, REAL **Vt, int rank, REAL *bu, REAL *bi, REAL glb_avg, int N, int M){
	int i,f,epochs,user,item;
	// REAL lambda1 = 0.003, lambda2 = 0.003, reg1 = 0.05, reg2 = 0.04;
	// REAL lambda1 = 0.001, lambda2 = 0.003, reg1 = 0.01, reg2 = 0.02;
	REAL sqerr, prev_sqerr = 100.0, error;
	REAL tmp_user, tmp_item, pred, *cached_prods;

	cached_prods = safe_malloc(N*M*sizeof(*cached_prods));

	for(f=0; f<rank; f++){
		for(epochs=0; epochs < MAX_EPOCHS; epochs++){
			sqerr = 0.0;
			for(i=0; i<training_size; i++){
				user = training_set[i].u;
				item = training_set[i].i;
				
				pred = glb_avg + bu[user] + bi[item];
				if (f == 0)
					pred += dot_product(U,Vt,user,item,f,rank);
				else
					pred += cached_prods[user*M+item] + dot_product(U,Vt,user,item,f,rank);

				error = training_set[i].rate - pred;
				sqerr += pow(error,2);

				bu[user] += l1 * (error - r1 * bu[user]);
				bi[item] += l2 * (error - r2 * bi[item]);

				tmp_user = U[user][f];
				tmp_item = Vt[f][item];

				U[user][f] += l3 * (error * tmp_item - r3 * tmp_user);
				Vt[f][item] += l4 * (error * tmp_user - r4 * tmp_item);
			}
			sqerr += r3 * summedsquaredL2norms(U,N,rank,0) + r4 * summedsquaredL2norms(Vt,rank,M,1) + r1 * summedsquaredL2norms2(bu,N) + r2 * summedsquaredL2norms2(bi,M); // not sure if it is needed
			if(fabs(sqerr - prev_sqerr) < 1e-5){
				// printf("Feature: %d Epochs: %d\n",f,epochs);
				break;
			}
			prev_sqerr = sqerr;
		}
		// cache calculated residuals
		for(i = 0; i<training_size; i++){
			user = training_set[i].u;
			item = training_set[i].i;
			if (f == 0)
				cached_prods[user*M+item] = U[user][f]*Vt[f][item];
			else
				cached_prods[user*M+item] += U[user][f]*Vt[f][item];
		}
	}
	free(cached_prods);
}

int main(int argc, char *argv[]){
	/* Give rank */
	int rank,feedback;
	if (argc < 4)
		usage(argv[0]);
	else{
		rank = atoi(argv[1]);
		if (rank <= 0)
			usage(argv[0]);
		TRAINING_PERCENT = atof(argv[2]);
		if ((TRAINING_PERCENT <= 0.0) | (TRAINING_PERCENT >= 1.0))
			usage(argv[0]);
		feedback = atoi(argv[3]);
		if ((feedback != 0) & (feedback != 1))
			usage(argv[0]);
	}

	int len;
	// len = sizeof(training_files) / sizeof(*training_files);
	len = sizeof(training_base) / sizeof(*training_base);

	int i,j,N,M;
	N = len;
	M = 2880; //total number of configurations
	// printf("N = %d\n",N );

	REAL **data;
	/* we take ipc, data is a 2D matrix with the known ratings */
	data = get_data(N,M);

	// REAL min = 99999999.9, max = -1.0;
	// for(i=0;i<N;i++){
	// 	for(j=0;j<M;j++){
	// 		printf("%f ",data[i][j]);
	// 		max = (data[i][j] > max)? data[i][j] : max;
	// 		min = (data[i][j] < min)? data[i][j] : min;
	// 	}
	// 	printf("\n");
	// }
	// printf("%f %f\n",min,max );
	// return 1;

	data_struct *training_set = NULL, *test_set = NULL;
	int training_size, test_size;
	/* 
	 *	choose training set from data_set, for a sparse initial matrix 
	 */
	//get_training_test_data(data,N,M,&training_set,&training_size,&test_set,&test_size);
	
	/*
	 *	Create the training set 
	 */
	get_training_data(data,N,M,&training_set,&training_size);

	/*REAL *bias_user, *bias_item, global_average=0.0;
	bias_user = safe_malloc(N*sizeof(REAL));
	bias_item = safe_malloc(M*sizeof(REAL));
	
	for(i=0;i<N;i++)
		bias_user[i] = INIT_VAL;
	for(i=0;i<M;i++)
		bias_item[i] = INIT_VAL;

	for(i=0;i<training_size;i++)
		global_average += training_set[i].rate / training_size;
	
	/* 
	 * U: N x rank matrix
	 * Vt: rank x M matrix 
	 */
/*	REAL **U,**Vt;

	U = safe_malloc(N*sizeof(*U));
	for(i=0;i<N;i++)
		U[i] = safe_malloc(rank*sizeof(REAL));	

	Vt = safe_malloc(rank*sizeof(*Vt));
	for(i=0;i<rank;i++)
		Vt[i] = safe_malloc(M*sizeof(REAL));

	initialize(U,Vt,N,M,rank,INIT_VAL);

	l1 = 0.001, r1 = 0.01, l2 = 0.001, r2 = 0.01, l3 = 0.005, r3 = 0.01, l4 = 0.005, r4 = 0.01;

	SVD_plus(training_set,training_size,U,Vt,rank,bias_user,bias_item,global_average,N,M);

	REAL rmse, sq_error = 0.0, pred;
	for(i=0; i<training_size; i++){
		pred = global_average + bias_user[training_set[i].u] + bias_item[training_set[i].i] + dot_product(U,Vt,training_set[i].u,training_set[i].i,0,rank);
		sq_error += pow(training_set[i].rate - pred,2);
	}
	rmse = sqrt(sq_error/training_size);
	printf("RMSE = %lf\n",rmse);

	return 0;*/

	/*
	 *		Values for ipc/watt factorization
	 */
	l1 = 0.002, r1 = 0.01, l2 = 0.001, r2 = 0.01, l3 = 0.005, r3 = 0.01, l4 = 0.005, r4 = 0.01;

	/*
	 *		Values for (Mflops/sec) / Watt factorization
	 */

	// l1 = 0.002, l2 = 0.002, l3 = 0.002 , l4 = 0.003, r1 = 0.01, r2 = 0.01, r3 = 0.02, r4 = 0.03;

	// l1 = 0.002, l2 = 0.002, l3 = 0.0005 , l4 = 0.0005, r1 = 0.005, r2 = 0.005, r3 = 0.01, r4 = 0.01;

	REAL *new_user;
	int num_testfiles;
	// num_testfiles = sizeof(test_files) / sizeof(*test_files);
	num_testfiles = sizeof(test_base) / sizeof(*test_base);

	REAL avg=0.0; 
	/*
	 *	Begin predictions for every test file
	 */
	for(j=0;j<num_testfiles;j++){
		test_set = NULL;
		/* incoming user 1xM */
		new_user = get_new_user(j,M);
		update_training_set_get_test_set(new_user,N,M,&training_set,&training_size,&test_set,&test_size);

		N = N + 1;
		//printf("Training size = %.2f%% Test size = %.2f%%\n",training_size*100.0/(N*M),test_size*100.0/M );

		REAL *bias_user, *bias_item, global_average=0.0;
		bias_user = safe_malloc(N*sizeof(REAL));
		bias_item = safe_malloc(M*sizeof(REAL));
		
		for(i=0;i<N;i++)
			bias_user[i] = INIT_VAL;
		for(i=0;i<M;i++)
			bias_item[i] = INIT_VAL;

		for(i=0;i<training_size;i++)
			global_average += training_set[i].rate / training_size;
		
		/* 
		 * U: N x rank matrix
		 * Vt: rank x M matrix 
		 */
		REAL **U,**Vt;

		U = safe_malloc(N*sizeof(*U));
		for(i=0;i<N;i++)
			U[i] = safe_malloc(rank*sizeof(REAL));	

		Vt = safe_malloc(rank*sizeof(*Vt));
		for(i=0;i<rank;i++)
			Vt[i] = safe_malloc(M*sizeof(REAL));

		initialize(U,Vt,N,M,rank,INIT_VAL);
		
		/*
		 *	Starting matrix factorization 
		 */
		//matrix_factorization(training_set,U,Vt,N,M,rank,training_size);
		SVD_plus(training_set,training_size,U,Vt,rank,bias_user,bias_item,global_average,N,M);
		
#ifdef OUTPUT
		/* 
		 *	Output results to file for evaluation
		 */
		FILE *out_fp;
		char output_file[64], *tmp, *infile;
		infile = strdup(test_base[j].file);
		
		if(!(tmp = strtok(infile,"/"))){
			printf("Error with strtok, %d\n",__LINE__);
			exit(1);
		}
		if(!(tmp = strtok(NULL,"/"))){
			printf("Error with strtok, %d\n",__LINE__);
			exit(1);	
		}

		sprintf(output_file,"evaluation/UVdec_%s_%.3f_%d.csv",tmp,TRAINING_PERCENT,feedback);
		out_fp = fopen(output_file,"a");
		REAL pred2;
		for(i=0;i<M;i++){
			pred2 = global_average + bias_user[N-1] + bias_item[i] + dot_product(U,Vt,N-1,i,0,rank);
			fprintf(out_fp,"%d,%f,%f\n",i,new_user[i],pred2);
		}
		fclose(out_fp);
		free(infile);
#endif
		/*
		 *	Calculate the approximation error (RMSE,MAE)
		 */
		REAL rmse, mae, sq_error = 0.0, pred, abs_acc = 0.0;
		for(i=0; i<test_size; i++){
			pred = global_average + bias_user[test_set[i].u] + bias_item[test_set[i].i] + dot_product(U,Vt,test_set[i].u,test_set[i].i,0,rank);
			sq_error += pow(test_set[i].rate - pred,2);
			abs_acc += fabs(test_set[i].rate - pred);
		}
		mae = abs_acc / test_size;
		rmse = sqrt(sq_error/test_size);
		printf("RMSE = %lf MAE = %lf\n",rmse,mae);
		avg += rmse/5;
		// printf("%lf\n",rmse);

		for(i=0;i<N;i++)
			free(U[i]);
		free(U);
		for(i=0;i<rank;i++)
			free(Vt[i]);
		free(Vt);
		
		free(bias_user);
		free(bias_item);
		free(new_user);
		free(test_set);

		/* Feedback: Update training set with known values from the test file. */
		if(!feedback){
			/* No feedback */
			N = N - 1;
			free(training_set);
			training_set = NULL;
			get_training_data(data,N,M,&training_set,&training_size);
		}
	}
	// printf("%f\n",avg );

	/* Deallocate memory */
	free(training_set);
	for(i=0;i<len;i++)
		free(data[i]);
	free(data);
	return 0;
}