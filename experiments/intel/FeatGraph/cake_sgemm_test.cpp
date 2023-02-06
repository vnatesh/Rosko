#include "cake.h"



int main( int argc, char** argv ) {

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

	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));
	rand_init(B, K, N);

	csr = file_to_csr(argv[9]);
	nz = csr->rowptr[M]; // 114615892
	M = csr->M;
	K = csr->K;
	density = ((float) nz) / ((float) (((float) M) * ((float) K)));

	cake_cntx_t* cake_cntx = cake_query_cntx();
	update_mr_nr(cake_cntx, 20, 96);

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	init_block_dims(M, N, K, p, x, cake_cntx, KMN, NULL, density);
	sp_pack_t* sp_pack = malloc_sp_pack(M, K, nz, x, cake_cntx);


 	printf("time = %f, mc = %d, kc = %d, mr = %d, nr = %d, nnz = %d\n", 
 		diff_t / ntrials, x->m_c, x->k_c, cake_cntx->mr, cake_cntx->nr, nz);

	pack_A_csr_to_sp_k_first(csr, M, K, nz, p, sp_pack, x, cake_cntx);

	printf("done packing\n");


    float ressss;
    float tttmp[18];
    int flushsz=10000000;
    diff_t = 0.0;
    
    for(int i = 0; i < ntrials; i++) {


        float *dirty = (float *)malloc(flushsz * sizeof(float));
        #pragma omp parallel for
        for (int dirt = 0; dirt < flushsz; dirt++){
            dirty[dirt] += dirt%100;
            tttmp[dirt%18] += dirty[dirt];
        }

        for(int ii =0; ii<18;ii++){
            ressss+= tttmp[ii];
        }


		diff_t += cake_sp_sgemm_testing(argv[9], B, C, M, N, K, p, cake_cntx, 
									density, NULL, sp_pack, 1, 0, 1, 0, KMN);
		// diff_t += cake_sp_sgemm(A, B, C, M, N, K, p, cake_cntx, density);
		// diff_t += cake_sgemm(A, B, C, M, N, K, p, cake_cntx);

        free(dirty);
    }

 	printf("time = %f, mc = %d, kc = %d, mr = %d, nr = %d\n", 
 		diff_t / ntrials, x->m_c, x->k_c, cake_cntx->mr, cake_cntx->nr);
 	// printf("time = %f\n", diff_t / ntrials);
	free(x);
	free_sp_pack(sp_pack);



	// cake_sgemm_checker(A, B, C, N, M, K);


	// free(A);
	free(B);
	free(C);
	free(cake_cntx);

	return 0;
}




