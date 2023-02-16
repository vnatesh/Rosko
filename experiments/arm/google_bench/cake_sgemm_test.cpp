#include "rosko.h"
#include <string.h>

float rand_gen();

float rand_gen() {
   // return a uniformly distributed random value
   return ( (float)(rand()) + 1. )/( (float)(((float) RAND_MAX)) + 1. );
}


int main( int argc, char** argv ) {
	 // run_tests();

	int M, K, N, p, nz, ntrials = atoi(argv[3]), write_result = atoi(argv[4]);
	struct timespec start, end;
	double diff_t;

	N = 2048; // fix N dimension for now (batch size = 8, seq len = 256)
	p = 4; // 4 cores on rasbpi 4

	int id = atoi(argv[2]);

	// read in sparse matrix A from google DNN benchmark
	FILE *fptr;
	char *line = NULL;
	size_t len = 0;
	ssize_t nread;

	fptr = fopen(argv[1], "r");
	if (fptr == NULL) {
	   perror("fopen");
	   exit(EXIT_FAILURE);
	}

	nread = getline(&line, &len, fptr);
	M = atoi(strtok(line," "));
	K = atoi(strtok(NULL, " "));
	nz = atoi(strtok(NULL, " "));

	float* A = (float*) malloc(M * K * sizeof( float ));

	printf("M = %d K = %d nz = %d  N = %d, cores = %d\n", M, K, nz, N, p);

	nread = getline(&line, &len, fptr);
	nread = getline(&line, &len, fptr);

	int i = 0, j = 0, prev = 0;
	char* tok;
	tok = strtok(line," \n");

	while (tok != NULL) {

		j = atoi(tok);

		if(j < prev) {
			i++;
		}

		prev = j;
		A[i*K + j] = rand_gen();
		tok = strtok(NULL, " \n");
	}

   	free(line);
   	fclose(fptr);


	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));
	 // rand_sparse(A, M, K, 0.80);
	//rand_sparse_gaussian(A, M, K, 0, 1);
	//rand_init(A, M, K);
	// print_array(A, M*K);
	// exit(1);
	rand_init(B, K, N);

	cake_cntx_t* cake_cntx = cake_query_cntx();
	
//	cake_sgemm(A, B, C, M, N, K, p, cake_cntx);

	// double ret = rosko_sgemm(A, B, C, M, N, K, p, cake_cntx);	
	double ans = 0;
	for(int i = 0; i < ntrials; i++) {
		ans += cake_sgemm(A, B, C, M, N, K, p, cake_cntx);
	}

	if(write_result) {
	    char fname[50];
	    snprintf(fname, sizeof(fname), "result_dlmc");
	    FILE *fp;
	    fp = fopen(fname, "a");
	    fprintf(fp, "CAKE,%d,%d,%d,%d,%d,%f\n",M,K,N,nz,id,ans/ntrials);
	    fclose(fp);
	}


	// cake_sgemm_checker(A, B, C, N, M, K);
	
	free(A);
	free(B);
	free(C);

	return 0;
}


