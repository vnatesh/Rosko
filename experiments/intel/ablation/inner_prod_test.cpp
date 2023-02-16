#include "rosko.h"
// #include <stdio.h>
// #include <stdlib.h>
// #include <time.h>


int main( int argc, char** argv ) {
	 // run_tests();

    if(argc < 3) {
        printf("Enter M, K, and N\n");
        exit(1);
    }

	int M, K, N, p;
	struct timespec start, end;
	double diff_t;

	M = atoi(argv[1]);
	K = atoi(argv[2]);
	N = atoi(argv[3]);
	p = atoi(argv[4]);

	printf("M = %d, K = %d, N = %d, cores = %d\n", M,K,N,p);

	float* A = (float*) malloc(M * K * sizeof( float ));
	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));
	rand_sparse(A, M, K, 0.95);
	// // rand_sparse_gaussian(A, M, K, 0, 1);
	// // rand_init(A, M, K);
	// // print_array(A, M*K);
	// // exit(1);
	rand_init(B, K, N);
	
	float sum;
	clock_gettime(CLOCK_REALTIME, &start);

	for(int i = 0; i < M; i++) {
		for(int j = 0; j < N; j++) {
			// #pragma omp parallel for simd schedule (static,16)
			sum = 0.0; 

			// #pragma omp for simd reduction(+:sum) //schedule(simd: static, 16)
			#pragma omp parallel for simd schedule (static,16)
			for(int k = 0; k < K; k++) {
				sum += A[i*K + k]*B[k*N + j];
				// int q = k*8;
				// sum += A[i*K + q++]*B[q*N + j];
				// sum += A[i*K + q++]*B[q*N + j];
				// sum += A[i*K + q++]*B[q*N + j];
				// sum += A[i*K + q++]*B[q*N + j];
				// sum += A[i*K + q++]*B[q*N + j];
				// sum += A[i*K + q++]*B[q*N + j];
				// sum += A[i*K + q++]*B[q*N + j];
				// sum += A[i*K + q++]*B[q*N + j];
			}

			C[i*N + j] = sum;
		}
	}

    clock_gettime(CLOCK_REALTIME, &end);
    long seconds = end.tv_sec - start.tv_sec;
    long nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	printf("inner product time: %f \n", diff_t); 

	// cake_sgemm_checker(A, B, C, N, M, K);
	
	free(A);
	free(B);
	free(C);

	return 0;
}



