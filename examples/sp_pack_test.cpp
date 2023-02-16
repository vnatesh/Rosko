#include "rosko.h"



int main( int argc, char** argv ) {
// 	run_tests();
// 	run_tests_sparse_test();
// 	run_tests_sparse();

// exit(1);
	int M, K, N, p, sp, nz, mr, nr, ntrials;
	struct timespec start, end;
	double diff_t;
	float density;
	csr_t* csr;

	M = atoi(argv[1]);
	K = atoi(argv[2]);
	N = atoi(argv[3]);
	p = atoi(argv[4]);
	sp = atoi(argv[5]);
	ntrials = atoi(argv[10]);

	printf("M = %d, K = %d, N = %d, cores = %d, sparsity = %f\n", M,K,N,p, ((float) sp) / 100.0);

	float* A = (float*) malloc(M * K * sizeof( float ));
	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));
	rand_sparse(A, M, K, ((float) sp) / 100.0);
	rand_init(B, K, N);
	// print_mat(A,M,K);

	nz = mat_to_csr_file(A, M, K, argv[9]);
	// nz = 114615892;
	density = ((float) nz) / ((float) (((float) M) * ((float) K)));
	cake_cntx_t* cake_cntx = cake_query_cntx();
	update_mr_nr(cake_cntx, 20, 96);

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	init_block_dims(M, N, K, p, x, cake_cntx, KMN, NULL, density);
	csr = file_to_csr(argv[9]);
	sp_pack_t* sp_pack = malloc_sp_pack(M, K, nz, x, cake_cntx);

 	printf("time = %f, mc = %d, kc = %d, mr = %d, nr = %d, nnz = %d\n", 
 		diff_t / ntrials, x->m_c, x->k_c, cake_cntx->mr, cake_cntx->nr, csr->rowptr[M]);

	char fname[50];
	snprintf(fname, sizeof(fname), "sp_pack");
	pack_A_csr_to_sp_k_first(csr, M, K, csr->rowptr[M], p, sp_pack, x, cake_cntx);
	sp_pack_to_file(sp_pack, fname);

	printf("done packing\n");

	free(x);
	free_sp_pack(sp_pack);


	free(A);
	free(B);
	free(C);
	free(cake_cntx);

	return 0;
}



