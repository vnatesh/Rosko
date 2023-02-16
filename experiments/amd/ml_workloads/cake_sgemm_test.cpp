#include "rosko.h"




int main( int argc, char** argv ) {


	int M, K, N, p, ntrials;
	struct timespec start, end;
	double diff_t, ans;
	float *A, *B, *C;
	long seconds, nanoseconds;
    cake_cntx_t* cake_cntx = cake_query_cntx();

	char fname[50];
	snprintf(fname, sizeof(fname), "ml_workloads");

    FILE *fp;


	M = atoi(argv[1]);
	K = atoi(argv[2]);
	N = atoi(argv[3]);
    p = atoi(argv[4]);
	ntrials = atoi(argv[5]);

    srand(time(NULL));

	for(int i = 0; i < ntrials; i++) {

		A = (float*) malloc(M * K * sizeof( float ));
		B = (float*) malloc(K * N * sizeof( float ));
		C = (float*) calloc(M * N , sizeof( float ));

		rand_init(A, M, K);
		rand_init(B, K, N);
		
	    clock_gettime(CLOCK_REALTIME, &start);

		cake_sgemm(A, B, C, M, N, K, p, cake_cntx);

	    clock_gettime(CLOCK_REALTIME, &end);
	    seconds = end.tv_sec - start.tv_sec;
	    nanoseconds = end.tv_nsec - start.tv_nsec;
	    diff_t = seconds + nanoseconds*1e-9;

		fp = fopen(fname, "a");
		fprintf(fp, "cake,%d,%d,%d,%d,%f\n",M,K,N,p,diff_t);
		fclose(fp);
	
		free(A);
		free(B);
		free(C);





		A = (float*) malloc(M * K * sizeof( float ));
		B = (float*) malloc(K * N * sizeof( float ));
		C = (float*) calloc(M * N , sizeof( float ));

		float density_val = 0.05;

		rand_sparse(A, M, K, 0.95);
		rand_init(B, K, N);
		
	    clock_gettime(CLOCK_REALTIME, &start);

        rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density_val);

	    clock_gettime(CLOCK_REALTIME, &end);
	    seconds = end.tv_sec - start.tv_sec;
	    nanoseconds = end.tv_nsec - start.tv_nsec;
	    diff_t = seconds + nanoseconds*1e-9;

	    fp = fopen(fname, "a");
		fprintf(fp, "rosko,%d,%d,%d,%d,%f\n",M,K,N,p,diff_t);
	    fclose(fp);
	
		free(A);
		free(B);
		free(C);
	}


	free(cake_cntx);

	return 0;
}







