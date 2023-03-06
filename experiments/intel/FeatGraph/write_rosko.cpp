#include "rosko.h"


// make; ./cake_sgemm_test 512 10 10 ogbn-proteins  

int main( int argc, char** argv ) {

	int M, K, N, p, nz, mr, nr;
	struct timespec start, end;
	double diff_t;
	float density;
	enum sched sch = KMN;
	csr_t* csr;

	N = atoi(argv[1]);
	p = atoi(argv[2]);

	csr = file_to_csr(argv[3]);
	M = csr->M;
	K = csr->K;
	nz = csr->rowptr[M]; // 114615892
	density = ((float) nz) / ((float) (((float) M) * ((float) K)));


	cake_cntx_t* cake_cntx = cake_query_cntx();
	if(N < 128) {
		update_mr_nr(cake_cntx, MR_MAX, N);
	} else {
		update_mr_nr(cake_cntx, MR_MAX, NR_MAX);
	}
	
	int alg = 2;
	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	init_sparse_block_dims(M, N, K, p, x, cake_cntx, sch, NULL, density, 4, alg);
	sp_pack_t* sp_pack = malloc_sp_pack(M, K, nz, x, cake_cntx);

	printf("M = %d, K = %d, N = %d, cores = %d, nz = %d\n", M,K,N,p, nz);
 	printf("mc = %d, kc = %d, nc = %d, mr = %d, nr = %d, nnz = %d\n", 
 		x->m_c, x->k_c, x->n_c, cake_cntx->mr, cake_cntx->nr, nz);

	pack_A_csr_to_sp_k_first(csr, M, K, nz, p, sp_pack, x, cake_cntx);
	sp_pack_to_file(sp_pack, argv[4]);
	printf("done packing\n");

	free(x);
	free_sp_pack(sp_pack);
	free(cake_cntx);
	free_csr(csr);

	return 0;
}




