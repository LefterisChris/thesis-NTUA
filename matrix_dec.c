#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>


#ifndef MAX_EPOCHS
#define MAX_EPOCHS 120
#endif

struct training_struct{
	int u;
	int i;
	float rate;
};

float lambda1,lambda2,lambda3,lambda4,reg1,reg2,reg3,reg4;

int usage(char *name){
	fprintf(stderr, "Please run: %s rank>0\n",name);
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

float dot_product(float **X, float **Y, int row, int col,int rank){
	int k;
	float tmp = 0.0;
	for(k=0; k < rank; k++)
		tmp += X[row][k]*Y[k][col];

	return tmp;
}

float dot_product2(float **X, float **Y, int row, int col, int start, int end){
	int k;
	float tmp = 0.0;
	for(k=start; k < end; k++)
		tmp += X[row][k]*Y[k][col];

	return tmp;
}

float summedsquaredL2norms(float **M, int rows, int cols, int transposed){
	int i,j;
	float tmp1 = 0.0, tmp2;
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

float summedsquaredL2norms2(float *M, int size){
	int i;
	float tmp = 0.0;
	for(i=0;i<size;i++)
		tmp += pow(M[i],2);
	return tmp;
}

float calculate_initial_value(float **A, int N,int M,int rank){
	/* 
	 * Initial value derives as the square root 
	 * of the global average divided by the given rank.
	 */
	float sum = 0.0;
	int non_zero = 0;
	int i,j;
	for(i=0;i<N;i++)
		for(j=0;j<M;j++)
			if(A[i][j] > 0){
				sum += A[i][j];
				non_zero++;
			}
	return (sqrtf((sum/non_zero)/rank));
}

float calculate_initial_value2(struct training_struct *TS,int size,int rank){
	float sum = 0.0;
	int i;
	for(i=0;i<size;i++)
		sum += TS[i].rate;
	return sqrt((sum/size)/rank);
}

void initialize(float **U, float **Vt, int N, int M, int rank, float init_val){
	int i,j,k;
	/* Initialize U,Vt */
	for(k=0;k<rank;k++){
		for(i=0;i<N;i++)
			U[i][k] = init_val;
		for(j=0;j<M;j++)
			Vt[k][j] = init_val;
	}
}

/*
 *	Implementation from analytical solution of Ullman's UV decomposition	
 */
float matrix_factorization(float **A,float **U, float **Vt, int N, int M, int rank){
	int epochs,i,j,f;
	float prev_sqerr,sq_err=100.0;
	float tmp1,tmp2;

	for(epochs=0; epochs<MAX_EPOCHS; epochs++){
		for(f=0; f<rank; f++){
			for(i=0; i<N; i++){
				tmp1=tmp2=0.0;
				for(j=0; j<M; j++)
					if (A[i][j] > 0){
						tmp1 += Vt[f][j]*(A[i][j] - (dot_product(U,Vt,i,j,rank) - U[i][f]*Vt[f][j]));
						tmp2 += pow(Vt[f][j],2);
					}
				U[i][f] = tmp1/tmp2;
			}
			for(j=0; j<M; j++){
				tmp1=tmp2=0.0;
				for(i=0; i<N; i++)
					if(A[i][j] > 0){
						tmp1 += U[i][f]*(A[i][j] - (dot_product(U,Vt,i,j,rank) - U[i][f]*Vt[f][j]));
						tmp2 += pow(U[i][f],2);
					}
				Vt[f][j] = tmp1/tmp2;
			}
		}
		/* Calculate squared error */
		prev_sqerr = sq_err;
		sq_err = 0.0;
		for(i=0; i < N; i++)
			for(j=0; j< M; j++)
				if(A[i][j] > 0)
					sq_err += pow(A[i][j] - dot_product(U,Vt,i,j,rank),2);
		
		if(fabs(sq_err - prev_sqerr) < 1e-5){
			printf("Epochs passed: %d \n",epochs);
			printf("previous = %f\n",prev_sqerr );
			break;
		}
	}

	return sq_err;
}

/*
 *	Simon Funk's incremental SVD implementation (simple factorization)
 */
float matrix_factorization2(struct training_struct *training_set,float **U, float **Vt, int rank,int training_size,int N,int M){
	float lambda = 0.002, regularizer=0.02;
	int epochs,i,f,user,item;
	float sq_err,prev_sqerr = 100.0, error;
	float pred, tmp, *cached_prods;

	cached_prods = safe_malloc(N*M*sizeof(*cached_prods));
	memset(cached_prods,0,sizeof(cached_prods));

	for(f=0; f<rank; f++){
		for(epochs=0; epochs < MAX_EPOCHS; epochs++){
			sq_err = 0.0;
			for(i=0; i<training_size; i++){
				user = training_set[i].u;
				item = training_set[i].i;
				if (f == 0)
					pred = dot_product2(U,Vt,user,item,f,rank);
				else
					pred = cached_prods[user*M+item] + dot_product2(U,Vt,user,item,f,rank);
				
				error = training_set[i].rate - pred;

				tmp = U[user][f];
				U[user][f] += lambda * (error * Vt[f][item] - regularizer * tmp);
				Vt[f][item] += lambda * (error * tmp - regularizer * Vt[f][item]);

				sq_err += pow(error,2);
			}
			if(fabs(sq_err - prev_sqerr) < 1e-5){
				printf("Feature: %d Epochs: %d\n",f,epochs);
				break;
			}
			prev_sqerr = sq_err;
		}
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
	return (1.0);
}

/*
 *	Baseline predictor, empirical forms
 */
void baseline_predictor(struct training_struct *training_set, int training_size, float *bu, float *bi, float glb_avg, int N, int M){
	int i,user,item;
	int lambda_u, lambda_i;
	lambda_u = 24;
	lambda_i = 1;
	int *user_rates, *item_rates;
	user_rates = calloc(N,sizeof(*user_rates));
	item_rates = calloc(M,sizeof(*item_rates));
	float rate;

	for(i=0;i<training_size;i++){
		user = training_set[i].u;
		rate = training_set[i].rate;
		bu[user] += rate - glb_avg;
		user_rates[user]++; 
	}
	for(i=0;i<N;i++)
		bu[i] = bu[i] / (lambda_u + user_rates[i]);

	for(i=0;i<training_size;i++){
		user = training_set[i].u;
		item = training_set[i].i;
		rate = training_set[i].rate;
		bi[item] += rate - glb_avg - bu[user];
		item_rates[item]++; 
	}
	for(i=0;i<M;i++)
		bi[i] = bi[i] / (lambda_i + item_rates[i]);

	free(user_rates);
	free(item_rates);
}

/*
 *	Baseline predictor using gradient descent
 */
void baseline_predictor_gdc(struct training_struct *training_set, int training_size, float *bu, float *bi, float glb_avg, int N, int M){
	int i,epochs,user,item;
	float lambda = 0.005, regularizer = 0.02;
	float sq_err, prev_sqerr = 100.0, error;
	int MAX_EPOCHS2 = 53;

	for(epochs = 0; epochs < MAX_EPOCHS2; epochs++){
		for(i=0;i<training_size;i++){
			user = training_set[i].u;
			item = training_set[i].i;
			error = training_set[i].rate - glb_avg - bu[user] - bi[item];
			bu[user] += lambda * (error - regularizer * bu[user]);
			bi[item] += lambda * (error - regularizer * bi[item]);
		}
		sq_err = 0.0;
		for(i=0;i<training_size;i++){
			user = training_set[i].u;
			item = training_set[i].i;
			error = training_set[i].rate - glb_avg - bu[user] - bi[item];
			sq_err += pow(error,2);
		}
		sq_err += regularizer * (summedsquaredL2norms2(bu,N)+summedsquaredL2norms2(bi,M));
		if(fabs(sq_err - prev_sqerr) < 1e-5){
			printf("Epochs: %d\n",epochs);
			break;
		}
		prev_sqerr = sq_err;
	}
}

/*
 * SGD normal approach. (stanford) NOTE: A feature vector has the same elements if initialized to a standard value.
 * So I cannot infer the SVD form.  
 */
void SVD_plus_revised(struct training_struct *training_set, int training_size, float **U, float **Vt, int rank, 
	float *bu, float *bi, float glb_avg, int N, int M){
	
	int i,f,epochs,user,item;
	float lambda1 = 0.001, lambda2 = 0.001, reg1 = 0.01, reg2 = 0.04;
	float tmp,error;

	for(epochs=0; epochs < 500; epochs++){
		for(i=0; i<training_size; i++){
			user = training_set[i].u;
			item = training_set[i].i;
			error = training_set[i].rate - glb_avg - bu[user] - bi[item] - dot_product(U,Vt,user,item,rank);

			bu[user] += lambda1 * (error - reg1 * bu[user]);
			bi[item] += lambda1 * (error - reg1 * bi[item]);
			for(f=0; f<rank; f++){
				tmp = U[user][f];
				U[user][f] += lambda2 * (error * Vt[f][item] - reg2 * tmp);
				Vt[f][item] += lambda2 * (error * tmp - reg2 * Vt[f][item]);
			}
		}
	}
}


/*
 *	Incremental SVD with biases. 	
 */
void SVD_plus(struct training_struct *training_set, int training_size, float **U, float **Vt, int rank, 
	float *bu, float *bi, float glb_avg, int N, int M){
	
	int i,f,epochs,user,item;
	float lambda1 = 0.001, lambda2 = 0.003, reg1 = 0.05, reg2 = 0.05;
	float sqerr, prev_sqerr = 100.0, error;
	float tmp_user, tmp_item, pred, *cached_prods;

	cached_prods = safe_malloc(N*M*sizeof(*cached_prods));

	for(f=0; f<rank; f++){
		for(epochs=0; epochs < MAX_EPOCHS; epochs++){
			sqerr = 0.0;
			for(i=0; i<training_size; i++){
				user = training_set[i].u;
				item = training_set[i].i;

				pred = glb_avg + bu[user] + bi[item];
				if (f == 0)
					pred += dot_product2(U,Vt,user,item,f,rank);
				else
					pred += cached_prods[user*M+item] + dot_product2(U,Vt,user,item,f,rank);

				error = training_set[i].rate - pred;
				sqerr += pow(error,2);

				bu[user] += lambda1 * (error - reg1 * bu[user]);
				bi[item] += lambda1 * (error - reg1 * bi[item]);

				tmp_user = U[user][f];
				tmp_item = Vt[f][item];

				U[user][f] += lambda2 * (error * tmp_item - reg2 * tmp_user);
				Vt[f][item] += lambda2 * (error * tmp_user - reg2 * tmp_item);
			}
			sqerr += reg2 * (summedsquaredL2norms(U,N,rank,0) + summedsquaredL2norms(Vt,rank,M,1)) + reg1 * (summedsquaredL2norms2(bu,N) + summedsquaredL2norms2(bi,M));
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

/*
 *	Incremental SVD, with biases calculated by analytical form.
 */
void SVD_plus2(struct training_struct *training_set, int training_size, float **U, float **Vt, int rank, 
	float *bu, float *bi, float glb_avg, int N, int M){
	
	int i,f,epochs,user,item;
	float lambda = 0.003, reg = 0.05;
	float sq_err, prev_sqerr = 100.0, error;

	for(f=0; f<rank; f++){
		for(epochs=0; epochs < MAX_EPOCHS; epochs++){
			for(i=0; i<training_size; i++){
				user = training_set[i].u;
				item = training_set[i].i;
				error = training_set[i].rate - glb_avg - bu[user] - bi[item] - dot_product(U,Vt,user,item,rank);

				U[user][f] += lambda * (error * Vt[f][item] - reg * U[user][f]);
				Vt[f][item] += lambda * (error * U[user][f] - reg * Vt[f][item]);
			}
			sq_err = 0.0;
			for(i=0;i<training_size;i++){
				user = training_set[i].u;
				item = training_set[i].i;
				error = training_set[i].rate - glb_avg - bu[user] - bi[item] - dot_product(U,Vt,user,item,rank);
				sq_err += pow(error,2);
			}
			sq_err += reg * (summedsquaredL2norms(U,N,rank,0) + summedsquaredL2norms(Vt,rank,M,1));
			if(fabs(sq_err - prev_sqerr) < 1e-5){
				printf("Feature: %d Epochs: %d\n",f,epochs);
				break;
			}
			prev_sqerr = sq_err;
		}
	}
}

void get_data_movielens(struct training_struct *TS,int size, char * infile){
	FILE *fp;
	fp = fopen(infile,"r");
	int i,usr,itm;
	float rt;
	for(i=0;i<size;i++){
		fscanf(fp,"%d %d %f %*d",&usr,&itm,&rt);
		TS[i].u = usr - 1;
		TS[i].i = itm - 1;
		TS[i].rate = rt;
	}
	fclose(fp);
}

float verify(float **U,float **Vt,int N, int M, int rank,char *infile){
	FILE *fp;
	fp = fopen(infile,"r");
	int i,usr,itm,test_size=20000;
	float rt, rmse=0.0, prediction;
	for(i=0;i<test_size;i++){
		fscanf(fp,"%d %d %f %*d",&usr,&itm,&rt);
		prediction = dot_product(U,Vt,usr-1,itm-1,rank);
		rmse += pow(rt-prediction,2);
	}
	fclose(fp);
	rmse = sqrt(rmse/test_size);
	return rmse;
}

float verify_base(float *bu, float *bi, float glb_avg){
	FILE *fp;
	fp = fopen("ml-100k/u1.test","r");
	int i,usr,itm,test_size=20000;
	float rt, rmse=0.0, prediction;
	for(i=0;i<test_size;i++){
		fscanf(fp,"%d %d %f %*d",&usr,&itm,&rt);
		prediction = glb_avg + bu[usr-1] + bi[itm-1];
		rmse += pow(rt-prediction,2);
	}
	fclose(fp);
	rmse = sqrt(rmse/test_size);
	return rmse;
}

float verify_SVDplus(float **U, float **Vt, int rank, float *bu, float *bi, float glb_avg, char * infile){
	FILE *fp;
	fp = fopen(infile,"r");
	int i,usr,itm,test_size=20000;
	float rt, rmse=0.0, prediction;
	for(i=0;i<test_size;i++){
		fscanf(fp,"%d %d %f %*d",&usr,&itm,&rt);
		prediction = glb_avg + bu[usr-1] + bi[itm-1] + dot_product(U,Vt,usr-1,itm-1,rank);
		rmse += pow(rt-prediction,2);
	}
	fclose(fp);
	rmse = sqrt(rmse/test_size);
	return rmse;
}

int main(int argc, char *argv[]){
	/* Give rank */
	int rank;
	if (argc < 2)
		usage(argv[0]);
	else
		rank = atoi(argv[1]);


	int N,M;
	int i,j,training_size = 0;

	training_size = 80000;
	struct training_struct *training_set;
	training_set = safe_malloc(training_size*sizeof(struct training_struct));
	get_data_movielens(training_set,training_size,"ml-100k/u1.base");
	N = 943; M=1682;

	float init_val;
	//init_val = calculate_initial_value2(training_set,training_size,rank);
	init_val = 0.1;
	// printf("Initial value=%f\n",init_val );
	
	/* 
	 * U: N x rank matrix
	 * Vt: rank x M matrix 
	 */
	float **U,**Vt;

	U = safe_malloc(N*sizeof(*U));
	for(i=0;i<N;i++)
		U[i] = safe_malloc(rank*sizeof(float));	

	Vt = safe_malloc(rank*sizeof(*Vt));
	for(i=0;i<rank;i++)
		Vt[i] = safe_malloc(M*sizeof(float));

	initialize(U,Vt,N,M,rank,init_val);

	float sq_error;
	// sq_error = matrix_factorization(A,U,Vt,N,M,rank);

/*	printf("Starting matrix factorization\n");
	sq_error = matrix_factorization2(training_set,U,Vt,rank,training_size,N,M);
*/
	// printf("RMSE from training data = %f \n",sqrt(sq_error/training_size));


	/* verify */
	float rmse;
/*	rmse = verify(U,Vt,N,M,rank,"ml-100k/u1.test");
	printf("Matrix factorization: RMSE from test data = %f\n",rmse);
*/
	/* Baseline predictor */
	float *bias_user, *bias_item, global_average=0.0;
	bias_user = safe_malloc(N*sizeof(*bias_user));
	bias_item = safe_malloc(M*sizeof(*bias_item));
	for(i=0;i<N;i++)
		bias_user[i] = 0.1;
	for(i=0;i<M;i++)
		bias_item[i] = 0.1;
	
	for(i=0;i<training_size;i++)
		global_average += training_set[i].rate / training_size;
/*				
	baseline_predictor(training_set,training_size,bias_user,bias_item,global_average,N,M);	
	rmse = verify_base(bias_user, bias_item, global_average);
		
	printf("Baseline: RMSE from test data = %f\n",rmse);
	
	initialize(U,Vt,N,M,rank,init_val);
	SVD_plus2(training_set,training_size,U,Vt,rank,bias_user,bias_item,global_average,N,M);

	rmse = verify_SVDplus(U,Vt,rank,bias_user,bias_item,global_average);
	printf("SVD plus(no gdc): RMSE from test data = %f\n",rmse);


	for(i=0;i<N;i++)
		bias_user[i] = 0.1;
	for(i=0;i<M;i++)
		bias_item[i] = 0.1;

	baseline_predictor_gdc(training_set,training_size,bias_user,bias_item,global_average,N,M);

	rmse = verify_base(bias_user, bias_item, global_average,"ml-100k/u1.test");
		
	printf("Baseline(gdc): RMSE from test data = %f\n",rmse);
*/
	/*float best_rmse=1000.0, best_l1, best_r1,best_l2, best_r2;
	int a,b,c,d;

	float l1[] = {0.001,0.002,0.003,0.004,0.005};
	float l2[] = {0.001,0.002,0.003,0.004,0.005};
	float r1[] = {0.01,0.02,0.03,0.04,0.05};
	float r2[] = {0.01,0.02,0.03,0.04,0.05};
	for(a=0;a<5;a++)
		for(b=0;b<5;b++)
			for(c=0;c<5;c++)
				for(d=0;d<5;d++){
					lambda1 = l1[a]; lambda2 = l2[b];
					reg1 = r1[c]; reg2 = r2[d];
					
					for(i=0;i<N;i++)
						bias_user[i] = 0.1;
					for(i=0;i<M;i++)
						bias_item[i] = 0.1;
					initialize(U,Vt,N,M,rank,init_val);
					
					SVD_plus(training_set,training_size,U,Vt,rank,bias_user,bias_item,global_average,N,M);
					rmse = verify_SVDplus(U,Vt,rank,bias_user,bias_item,global_average,"ml-100k/u1.test");
					
					if (rmse < best_rmse){
						best_rmse = rmse;
						best_l1 = lambda1; best_l2 = lambda2;
						best_r1 = reg1; best_r2 = reg2;
					}
				}
	printf("Best rmse=%f for lambda1=%f lambda2=%f reg1=%f reg2=%f\n",best_rmse,best_l1,best_l2,best_r1,best_r2 );
	*/
	/* SVD matrix factorization: 
	 * pred = global_average + bias_user + bias_item + U * Vt 
	 */
	SVD_plus(training_set,training_size,U,Vt,rank,bias_user,bias_item,global_average,N,M);
	rmse = verify_SVDplus(U,Vt,rank,bias_user,bias_item,global_average,"ml-100k/u1.test");
	printf("SVD plus: RMSE from test data = %f\n",rmse);

	/*for(i=0;i<10;i++){
		for(j=0;j<5;j++)
			printf("%f ",U[i][j]);
		printf("\n");
	}

	for(i=0;i<5;i++){
		for(j=0;j<10;j++)
			printf("%f ",Vt[i][j]);
		printf("\n");
	}
*/

	free(bias_user);
	free(bias_item);

	free(training_set);
	for(i=0;i<N;i++)
		free(U[i]);
	free(U);
	for(j=0;j<rank;j++)
		free(Vt[j]);
	free(Vt);

	return 0;

}