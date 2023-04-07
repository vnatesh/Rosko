#include "rosko.h"


int main( int argc, char** argv ) {
	// run_tests();
	// run_tests_sparse_test();
	// run_tests_sparse();

// exit(1);
	int M, K, N, p, nz, mr, nr, ntrials, alg, A_sz;
	struct timespec start, end;
	double diff_t;
	float density, sp;
	float* A_p;
	long seconds, nanoseconds;

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

	density = (100 - sp) / 100.0;
	cake_cntx_t* cake_cntx = cake_query_cntx();
	
	if(density > 0.05) {
		update_mr_nr(cake_cntx, 20, 72);
		alg = 3;
	} else {
		update_mr_nr(cake_cntx, 20, 72);
		alg = 2;
	}


	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	init_sparse_block_dims(M, N, K, p, x, cake_cntx, KMN, NULL, density, 4, alg);
	omp_set_num_threads(p);


	clock_gettime(CLOCK_REALTIME, &start);

	A_sz = cake_sgemm_packed_A_size(M, K, p, x, cake_cntx, KMN) / sizeof(float);
	A_p = (float*) calloc(A_sz, sizeof(float));
	sp_pack_t* sp_pack = (sp_pack_t*) malloc(sizeof(sp_pack_t));
	pack_A_sp_k_first(A, A_p, M, K, p, sp_pack, x, cake_cntx);

	clock_gettime(CLOCK_REALTIME, &end);
	seconds = end.tv_sec - start.tv_sec;
	nanoseconds = end.tv_nsec - start.tv_nsec;
	diff_t = seconds + nanoseconds*1e-9;
	printf("done packing\n");


	char fname[50];
	snprintf(fname, sizeof(fname), "results");
	FILE *fp;
	fp = fopen(fname, "a");
	fprintf(fp, "%s,pack,%d,%d,%d,%f,%f\n", argv[7], M, K, N, sp, diff_t);


 	// printf("time = %f, mc = %d, kc = %d, nc = %d, mr = %d, nr = %d,  alpha = %f, density = %f\n", 
 	// 	diff_t / ntrials, x->m_c, x->k_c, x->n_c, cake_cntx->mr, cake_cntx->nr, cake_cntx->alpha_n, density);


    float ressss;
    float tttmp[18];
    int flushsz = cake_cntx->L3 / sizeof(float);
    diff_t = 0.0;
    
    for(int i = 0; i < ntrials; i++) {

        float* dirty = (float *) malloc(flushsz * sizeof(float));
        #pragma omp parallel for
        for (int dirt = 0; dirt < flushsz; dirt++){
            dirty[dirt] += dirt%100;
            tttmp[dirt%18] += dirty[dirt];
        }

        for(int ii =0; ii<18;ii++){
            ressss+= tttmp[ii];
        }


		// diff_t += rosko_sgemm_compressed(argv[8], B, C, M, N, K, p, cake_cntx, 
		// 							density, NULL, sp_pack, 1, 0, 1, 0, KMN, alg);
		diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, NULL, 1, sp_pack, 0, 1, 0, KMN, alg);

        free(dirty);
    }

	printf("%d,%f,%d,%d,%d,%f\n", alg, sp, M, K, N, diff_t / ntrials);
	fprintf(fp, "%s,rosko,%d,%d,%d,%f,%f\n", argv[7], M, K, N, sp, diff_t / ntrials);
	fclose(fp);
	free(x);
	free_sp_pack(sp_pack);

	// cake_sgemm_checker(A, B, C, N, M, K);

	// free(A);
	free(B);
	free(C);
	free(cake_cntx);

	return 0;
}



