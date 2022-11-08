#include "cake.h"



// randomized sparse Normal(0,1) matrix with sparsity % of values determined by sigma (std dev)
void rand_sparse_gaussian1(float* mat, int r, int c, float mu, float sigma) {
	int nnz = 0;
	for(int i = 0; i < r*c; i++) {
		float x = normalRandom()*sigma+mu;
		if(fabs(x) <= 4.26) { // 2 sigmas i.e. 95% sparse
			mat[i] = 0;
		} else {
			mat[i] =  x;
			nnz++;
		}
	}	
	printf("nnz = %d\n", nnz);
}


int main( int argc, char** argv ) {
	 // run_tests();

    if(argc < 3) {
        printf("Enter M, K, and N\n");
        exit(1);
    }

	int M, K, N, p, write_result;
	struct timespec start, end;
	long nanoseconds, seconds;
	double diff_t;

	M = atoi(argv[1]);
	K = atoi(argv[2]);
	N = atoi(argv[3]);
	p = atoi(argv[4]);
	float sp = atof(argv[5]);
	write_result = atoi(argv[6]);

	printf("M = %d, K = %d, N = %d, cores = %d, sparsity = %f\n", M,K,N,p, ((float) sp) / 100.0);

	float* A = (float*) malloc(M * K * sizeof( float ));
	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));
	rand_sparse(A, M, K, ((float) sp) / 100.0);
	// rand_sparse_gaussian1(A, M, K, 0, 4.26);
	// rand_init(A, M, K);
	// print_array(A, M*K);
	// exit(1);
	rand_init(B, K, N);




	cake_cntx_t* cake_cntx = cake_query_cntx();
	// update_mr_nr(cake_cntx, 30, 16);
	int iters = atoi(argv[7]);
	// if(M < 1792) {
	// 	iters = 20;
	// }
	double ret = 0;

	float ressss;
	float tttmp[18];
	int flushsz=100000000;

	for(int i = 0; i < iters; i++) {

        float *dirty = (float *)malloc(flushsz * sizeof(float));
        #pragma omp parallel for
        for (int dirt = 0; dirt < flushsz; dirt++){
            dirty[dirt] += dirt%100;
            tttmp[dirt%18] += dirty[dirt];
        }

        for(int ii =0; ii<18;ii++){
            ressss+= tttmp[ii];
        }


		ret += cake_sp_sgemm(A, B, C, M, N, K, p, cake_cntx, ((float) sp) / 100.0, NULL);


        free(dirty);

	}

	printf("sp_sgemm time: %f \n", ret/iters); 

	if(write_result) {
	    char fname[50];
	    snprintf(fname, sizeof(fname), "results");
	    FILE *fp;
	    fp = fopen(fname, "a");
	    fprintf(fp, "rosko,%d,%d,%d,%d,%f,%f\n",M,K,N,p,sp, ret/iters);
	    fclose(fp);
	}

	// cake_sgemm_checker(A, B, C, N, M, K);
	
	free(A);
	free(B);
	free(C);

	return 0;
}




