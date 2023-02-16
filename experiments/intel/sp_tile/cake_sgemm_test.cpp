#include "rosko.h"



int main( int argc, char** argv ) {
// 	run_tests();
// 	run_tests_sparse_test();
// 	run_tests_sparse();

	int M, K, N, p, nz, ntrials;
	struct timespec start, end;
	double diff_t;
	float density, sp;

	M = atoi(argv[1]);
	K = atoi(argv[2]);
	N = atoi(argv[3]);
	p = atoi(argv[4]);
	sp = atof(argv[5]);
	ntrials = atoi(argv[6]);

	float* A = (float*) malloc(M * K * sizeof( float ));
	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));
	rand_sparse(A, M, K, sp / 100.0);
	rand_init(B, K, N);
	density = 1.0 - (sp / 100.0);
	cake_cntx_t* cake_cntx = cake_query_cntx();

	printf("M = %d, K = %d, N = %d, cores = %d, density = %f\n", M,K,N,p, density);

	for(int mr = 6; mr < 21; mr += 2) {
		for(int nr = 16; nr < 97; nr += 16) {

			blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
			init_block_dims(M, N, K, p, x, cake_cntx, KMN, NULL, density);
		 	printf("mc = %d, kc = %d, nc = %d, mr = %d, nr = %d, density = %f\n", 
		 		x->m_c, x->k_c, x->n_c, mr, nr, density);

			update_mr_nr(cake_cntx, mr, nr);
		 	// printf("time = %f, mc = %d, kc = %d, nc = %d, mr = %d, nr = %d,  alpha = %f, nnz = %d, density = %f\n", 
		 	// 	diff_t / ntrials, x->m_c, x->k_c, x->n_c, cake_cntx->mr, cake_cntx->nr, cake_cntx->alpha_n, csr->rowptr[M], density);
			char fname[50];
			snprintf(fname, sizeof(fname), "results_mr_nr");
			FILE *fp;
			fp = fopen(fname, "a");

		    float ressss;
		    float tttmp[18];
		    int flushsz = cake_cntx->L3 / sizeof(float);
		    diff_t = 0.0;

		    for(int i = 0; i < ntrials; i++) {

		        float *dirty = (float*) malloc(flushsz * sizeof(float));

		        #pragma omp parallel for
		        for (int dirt = 0; dirt < flushsz; dirt++){
		            dirty[dirt] += dirt%100;
		            tttmp[dirt%18] += dirty[dirt];
		        }

		        for(int ii =0; ii<18;ii++){
		            ressss+= tttmp[ii];
		        }

				diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density);
		        free(dirty);

		    }

			fprintf(fp, "rosko_new,%f,%d,%d,%d,%d,%d,%f\n", sp, mr, nr, M, K, N, diff_t / ntrials);
			fclose(fp);
		}
	}



	// cake_sgemm_checker(A, B, C, N, M, K);


	free(A);
	free(B);
	free(C);
	free(cake_cntx);

	return 0;
}




