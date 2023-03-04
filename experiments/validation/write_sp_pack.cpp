#include "rosko.h"


int main( int argc, char** argv ) {

// exit(1);
	int M, K, N, p, alg;
	struct timespec start, end;
	double diff_t;
	float density, sp;
	enum sched sch = KMN;

	M = atoi(argv[1]);
	K = atoi(argv[2]);
	N = atoi(argv[3]);
	p = atoi(argv[4]);
	sp = atof(argv[5]);

	omp_set_num_threads(p);

	// printf("M = %d, K = %d, N = %d, cores = %d, sparsity = %f\n", M,K,N,p, ((float) sp) / 100.0);

	density = (100.0 - sp) / 100.0;

	float* A = (float*) malloc(M * K * sizeof( float ));
	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));
	rand_sparse(A, M, K, sp / 100.0);
	// rand_init(B, K, N);

	FILE *fptr1 = fopen(argv[10], "wb");
	fwrite(A, sizeof(float), M*K, fptr1);
	fclose(fptr1);


	cake_cntx_t* cake_cntx = cake_query_cntx();

	if(density > 0.0001) {
		update_mr_nr(cake_cntx, MR_MAX, NR_MAX);
		if(MR_MIN == 6 && NR_MIN == 16) {
			alg = 0;
		} else {
			alg = 3;
		}
	} else {
		update_mr_nr(cake_cntx, MR_MIN, NR_MIN);
		if(MR_MIN == 6 && NR_MIN == 16) {
			alg = 2;
		} else {
			alg = 3;
		}
	}

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	init_sparse_block_dims(M, N, K, p, x, cake_cntx, sch, NULL, density, 4, alg);
	size_t A_sz = cake_sgemm_packed_A_size(M, K, p, x, cake_cntx, sch) / sizeof(float);
    float* A_p = (float*) calloc(A_sz, sizeof(float));
	sp_pack_t* sp_pack = (sp_pack_t*) malloc(sizeof(sp_pack_t));
	pack_A_sp(A, A_p, M, K, p, sp_pack, x, cake_cntx, sch);
	sp_pack_to_file2(sp_pack, cake_cntx, x->M_padded, K, argv[9]);

	free(sp_pack->loc_m); 
	free(sp_pack->nnz_outer); 
	free(sp_pack->k_inds); 
	free(sp_pack->A_sp_p);
	free(sp_pack);
	free(A);
	free(x);
	free(cake_cntx);
	return 0;
}

