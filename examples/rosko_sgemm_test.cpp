#include "rosko.h"


int main( int argc, char** argv ) {
	// run_tests();
	// run_tests_sparse_test();
	// run_tests_sparse();

// exit(1);
	int M, K, N, p, nz, mr, nr, ntrials, alg, warmup = 10;
	struct timespec start, end;
	double diff_t;
	float density, sp;

	M = atoi(argv[1]);
	K = atoi(argv[2]);
	N = atoi(argv[3]);
	p = atoi(argv[4]);
	sp = atof(argv[5]);
	ntrials = atoi(argv[6]);

	printf("M = %d, K = %d, N = %d, cores = %d, sparsity = %f\n", M,K,N,p, ((float) sp) / 100.0);

	float* A = (float*) malloc(M * K * sizeof( float ));
	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));
	rand_sparse(A, M, K, ((float) sp) / 100.0);
	rand_init(B, K, N);

	density = (100.0 - sp) / 100.0;

	cake_cntx_t* cake_cntx = cake_query_cntx();

	char fname[50];
	snprintf(fname, sizeof(fname), "results_new_sp");
	FILE *fp;
	fp = fopen(fname, "a");

    float ressss;
    float tttmp[18];
    int flushsz = 2*cake_cntx->L3 / sizeof(float);
    diff_t = 0.0;
    
    printf("alg = %d, %d\n", alg, flushsz);

    for(int i = 0; i < (ntrials + warmup); i++) {

        float* dirty = (float *) malloc(flushsz * sizeof(float));
        #pragma omp parallel for
        for (int dirt = 0; dirt < flushsz; dirt++){
            dirty[dirt] += dirt%100;
            tttmp[dirt%18] += dirty[dirt];
        }

        for(int ii =0; ii<18;ii++){
            ressss+= tttmp[ii];
        }


		// diff_t += rosko_sgemm_online_BC(A, B, C, M, N, K, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, alg);
		// diff_t += rosko_sgemm_compressed(argv[8], B, C, M, N, K, p, cake_cntx, 
		// 							density, NULL, sp_pack, 1, 0, 1, 0, KMN, alg);
		// diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, 3);
		// diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, 3, 120, 274, 1024);
		
		if(i < warmup) {
			// diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, alg, 60, 974, 512);
			float y = rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, alg);
			printf("sss %f\n", y);
		} else {
			float y = rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, alg);
			printf("sss %f\n", y);
			diff_t += y;
		}

        free(dirty);
    }

	printf("%d,%f,%d,%d,%d,%f\n", alg, sp, M, K, N, diff_t / ntrials);
	fprintf(fp, "%d,%f,%d,%d,%d,%f\n", alg, sp, M, K, N, diff_t / ntrials);
	fclose(fp);

	cake_sgemm_checker(A, B, C, N, M, K);

	// free(A);
	free(B);
	free(C);
	free(cake_cntx);

	return 0;
}












// #include "rosko.h"



// typedef double rosko_sgemm_tester(float* A, float* B, float* C, int M, int N, int K, int p, 
// 	cake_cntx_t* cake_cntx, float density, char* argv[], 
// 	bool packedA, sp_pack_t* sp_pack, bool packedB, 
// 	float alpha, float beta, enum sched sch, int alg, int mcu, int kcu, int ncu);


// const static int num_funcs = 2;

// static rosko_sgemm_tester* test_funcs[num_funcs] = 
// {
// 	rosko_sgemm,
// 	rosko_sgemm_test
// 	// rosko_sgemm_online_B,
// 	// rosko_sgemm_online_BC,
// 	// rosko_sgemm_online
// };



// void square_test(int func, int ntrials, int p, int s, int e, int step, float sp) {

// 	int M, K, N, alg, warmup = 10;
// 	struct timespec start, end;
//     long seconds, nanoseconds;
// 	double diff_t;
// 	float density;

// 	for(int t = s; t < (e + 1); t += step) {

// 		M = t;
// 		K = t;
// 		N = t;

// 		printf("M = %d, K = %d, N = %d, cores = %d\n", M,K,N,p);

// 		float* A = (float*) malloc(M * K * sizeof( float ));
// 		float* B = (float*) malloc(K * N * sizeof( float ));
// 		float* C = (float*) calloc(M * N , sizeof( float ));

// 	    srand(time(NULL));
// 		rand_sparse(A, M, K, ((float) sp) / 100.0);
// 		rand_init(B, K, N);

// 		density = (100.0 - sp) / 100.0;
// 		cake_cntx_t* cake_cntx = cake_query_cntx();
//         update_mr_nr(cake_cntx, 30, 128);

// 		if(density > 0.05) {
// 	        // update_mr_nr(cake_cntx, 30, 128);
// 	        alg = 0;
// 	    } else {
// 	        // update_mr_nr(cake_cntx, 6, 16);
// 	        alg = 2;
// 	    }



// 	    float ressss;
// 	    float tttmp[18];
// 	    int flushsz = 2*cake_cntx->L3 / sizeof(float);
// 	    diff_t = 0.0;


// 	    for(int i = 0; i < (ntrials + warmup); i++) {


// 	        float *dirty = (float *)malloc(flushsz * sizeof(float));
// 	        #pragma omp parallel for
// 	        for (int dirt = 0; dirt < flushsz; dirt++){
// 	            dirty[dirt] += dirt%100;
// 	            tttmp[dirt%18] += dirty[dirt];
// 	        }

// 	        for(int ii =0; ii<18;ii++){
// 	            ressss+= tttmp[ii];
// 	        }

// 			if(i < warmup) {
// 				// diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, alg, 60, 974, 512);
// 				test_funcs[func](A, B, C, M, N, K, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, alg, 0, 0, 0);
// 			} else {
// 				diff_t += test_funcs[func](A, B, C, M, N, K, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, alg, 0, 0, 0);
// 			}


// 	        free(dirty);
// 	    }


// 	    // printf("cake_sgemm time: %f \n", diff_t / ntrials); 
// 	    // int write_result = atoi(argv[13]);
// 	    // if(write_result) {
// 		char fname[50];
// 		snprintf(fname, sizeof(fname), "results");
// 		FILE *fp;
// 		fp = fopen(fname, "a");
// 		fprintf(fp, "%d,%d,%d,%d,%d,%f,%f\n", func, p, M, K, N, sp, diff_t / ntrials);
// 		fclose(fp);
// 	    // }


		
// 		free(A);
// 		free(B);
// 		free(C);
// 		free(cake_cntx);
	
// 	}
// }



// int main( int argc, char** argv ) {


// 	int ntrials = atoi(argv[1]);
// 	int p = atoi(argv[2]);
// 	int start = atoi(argv[3]);
// 	int end = atoi(argv[4]);
// 	int step = atoi(argv[5]);
// 	int sp = atof(argv[6]);

// 	for(int i = 0; i < num_funcs; i++) {
// 		square_test(i, ntrials, p, start, end, step, sp);
// 	}

// 	return 0;
// }
















// #include "rosko.h"


// int main( int argc, char** argv ) {
// 	// run_tests();
// 	// run_tests_sparse_test();
// 	// run_tests_sparse();

// // exit(1);
// 	int M, K, N, p, nz, mr, nr, ntrials, alg;
// 	struct timespec start, end;
// 	double diff_t;
// 	float density, sp;
// 	csr_t* csr;

// 	M = atoi(argv[1]);
// 	K = atoi(argv[2]);
// 	N = atoi(argv[3]);
// 	p = atoi(argv[4]);
// 	sp = atof(argv[5]);
// 	ntrials = atoi(argv[6]);
// 	alg = atoi(argv[7]);

// 	printf("M = %d, K = %d, N = %d, cores = %d, sparsity = %f\n", M,K,N,p, ((float) sp) / 100.0);

// 	float* A = (float*) malloc(M * K * sizeof( float ));
// 	float* B = (float*) malloc(K * N * sizeof( float ));
// 	float* C = (float*) calloc(M * N , sizeof( float ));

// 	// initialize A and B
//     srand(time(NULL));
// 	rand_sparse(A, M, K, ((float) sp) / 100.0);
// 	rand_init(B, K, N);

// 	nz = mat_to_csr_file(A, M, K, argv[8]);
// 	printf("nz = %d\n", nz);
// 	density = ((float) nz) / ((float) (((float) M) * ((float) K)));

// 	cake_cntx_t* cake_cntx = cake_query_cntx();
// 	update_mr_nr(cake_cntx, 30, 128);

// 	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
// 	init_sparse_block_dims(M, N, K, p, x, cake_cntx, KMN, NULL, density, 4, alg);
// 	csr = file_to_csr(argv[8]);
// 	sp_pack_t* sp_pack = malloc_sp_pack(M, K, nz, x, cake_cntx);


//  	printf("time = %f, mc = %d, kc = %d, nc = %d, mr = %d, nr = %d,  alpha = %f, nnz = %d, density = %f\n", 
//  		diff_t / ntrials, x->m_c, x->k_c, x->n_c, cake_cntx->mr, cake_cntx->nr, cake_cntx->alpha_n, csr->rowptr[M], density);

// 	pack_A_csr_to_sp_k_first(csr, M, K, csr->rowptr[M], p, sp_pack, x, cake_cntx);

// 	printf("done packing\n");

// 	char fname[50];
// 	snprintf(fname, sizeof(fname), "results_new_sp");
// 	FILE *fp;
// 	fp = fopen(fname, "a");

//     float ressss;
//     float tttmp[18];
//     int flushsz = cake_cntx->L3 / sizeof(float);
//     diff_t = 0.0;
    
//     for(int i = 0; i < ntrials; i++) {

//         float* dirty = (float *) malloc(flushsz * sizeof(float));
//         #pragma omp parallel for
//         for (int dirt = 0; dirt < flushsz; dirt++){
//             dirty[dirt] += dirt%100;
//             tttmp[dirt%18] += dirty[dirt];
//         }

//         for(int ii =0; ii<18;ii++){
//             ressss+= tttmp[ii];
//         }


// 		// diff_t += rosko_sgemm_compressed(argv[8], B, C, M, N, K, p, cake_cntx, 
// 		// 							density, NULL, sp_pack, 1, 0, 1, 0, KMN, alg);
// 		diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, alg);

//         free(dirty);
//     }

// 	printf("%d,%f,%d,%d,%d,%f\n", alg, sp, M, K, N, diff_t / ntrials);
// 	fprintf(fp, "%d,%f,%d,%d,%d,%f\n", alg, sp, M, K, N, diff_t / ntrials);
// 	fclose(fp);
// 	free(x);
// 	free_sp_pack(sp_pack);

// 	// cake_sgemm_checker(A, B, C, N, M, K);

// 	// free(A);
// 	free(B);
// 	free(C);
// 	free(cake_cntx);

// 	return 0;
// }


