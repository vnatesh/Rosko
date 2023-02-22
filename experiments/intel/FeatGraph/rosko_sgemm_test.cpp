#include "rosko.h"


// make; ./cake_sgemm_test 512 10 10 ogbn-proteins  

int main( int argc, char** argv ) {

	int M, K, N, p, nz, mr, nr, ntrials;
	struct timespec start, end;
	double diff_t;
	float density;
	csr_t* csr;

	N = atoi(argv[1]);
	p = atoi(argv[2]);
	ntrials = atoi(argv[3]);

	csr = file_to_csr(argv[4]);
	M = csr->M;
	K = csr->K;
	nz = csr->rowptr[M]; // 114615892
	density = ((float) nz) / ((float) (((float) M) * ((float) K)));


	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));
	// initialize B
    srand(time(NULL));
	rand_init(B, K, N);


	cake_cntx_t* cake_cntx = cake_query_cntx();
	update_mr_nr(cake_cntx, 6, 16);

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	init_block_dims(M, N, K, p, x, cake_cntx, KMN, argv, density);
	sp_pack_t* sp_pack = malloc_sp_pack(M, K, nz, x, cake_cntx);

	printf("M = %d, K = %d, N = %d, cores = %d, nz = %d\n", M,K,N,p, nz);
 	printf("time = %f, mc = %d, kc = %d, nc = %d, mr = %d, nr = %d, nnz = %d\n", 
 		diff_t / ntrials, x->m_c, x->k_c, x->n_c, cake_cntx->mr, cake_cntx->nr, nz);

	pack_A_csr_to_sp_k_first(csr, M, K, nz, p, sp_pack, x, cake_cntx);

	printf("done packing\n");

	int alg = 2;

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


		diff_t += rosko_sgemm_compressed(argv[4], B, C, M, N, K, p, cake_cntx, 
									density, argv, sp_pack, 1, 0, 1, 0, KMN, alg);
		// diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density);
		// diff_t += cake_sgemm(A, B, C, M, N, K, p, cake_cntx);

        free(dirty);
    }

 	printf("time = %f, mc = %d, kc = %d, mr = %d, nr = %d\n", 
 		diff_t / ntrials, x->m_c, x->k_c, cake_cntx->mr, cake_cntx->nr);
 	// printf("time = %f\n", diff_t / ntrials);
	free(x);
	free_sp_pack(sp_pack);


    if(write_result) {
        char fname[50];
        snprintf(fname, sizeof(fname), "result_gnn");
        FILE *fp;
        fp = fopen(fname, "a");
        fprintf(fp, "rosko,%s,%d,%d,%d,%d,%f,%f\n",argv[4],M,K,N,p,(1-density)*100,diff_t / ntrials);
        fclose(fp);
    }



	// cake_sgemm_checker(A, B, C, N, M, K);


	// free(A);
	free(B);
	free(C);
	free(cake_cntx);

	return 0;
}




