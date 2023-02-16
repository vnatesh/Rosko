#include "rosko.h"




int main( int argc, char** argv ) {
	 // run_tests();

    if(argc < 3) {
        printf("Enter M, K, and N\n");
        exit(1);
    }

	int M, K, N, p, sp, write_result, ntrials;
	struct timespec start, end;
	double diff_t;

	M = atoi(argv[1]);
	K = atoi(argv[2]);
	N = atoi(argv[3]);
	p = atoi(argv[4]);
	write_result = atoi(argv[5]);
	sp = atoi(argv[6]);
	ntrials = atoi(argv[7]);

	printf("M = %d, K = %d, N = %d, cores = %d, sparsity = %f\n", M,K,N,p, ((float) sp) / 100.0);

	float* A = (float*) malloc(M * K * sizeof( float ));
	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));
	rand_sparse(A, M, K, ((float) sp) / 100.0);
	// rand_sparse_gaussian(A, M, K, 0, 1);
	// rand_init(A, M, K);
	// print_array(A, M*K);
	// exit(1);
	rand_init(B, K, N);

	cake_cntx_t* cake_cntx = cake_query_cntx();
	
	double ret = 0;
	clock_gettime(CLOCK_REALTIME, &start);

	for(int i = 0; i < ntrials; i++) {
		ret += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx);
	}

    clock_gettime(CLOCK_REALTIME, &end);
    long seconds = end.tv_sec - start.tv_sec;
    long nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	printf("rosko sp_sgemm time: %f \n", ret/ntrials); 

	if(write_result) {
	    char fname[50];
	    snprintf(fname, sizeof(fname), "result_ablate_arm");
	    FILE *fp;
	    fp = fopen(fname, "a");
	    fprintf(fp, "rosko,%d,%d,%d,%d,%f\n",M,K,N,sp,ret/ntrials);
	    fclose(fp);
	}
	// cake_sgemm_checker(A, B, C, N, M, K);
	
	free(A);
	free(B);
	free(C);

	return 0;
}



