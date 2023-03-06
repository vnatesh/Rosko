#include "rosko.h"


// make; ./cake_sgemm_test 512 10 10 ogbn-proteins  

int main( int argc, char** argv ) {
// run_tests_sparse_test();
	int M, K, N, p, nz, mr, nr, ntrials, dram, setup;
	struct timespec start, end;
	double diff_t;
	float density;
	enum sched sch = KMN;

	N = atoi(argv[1]);
	p = atoi(argv[2]);
	ntrials = atoi(argv[3]);
	dram = atoi(argv[5]);
	setup = atoi(argv[6]);


	cake_cntx_t* cake_cntx = cake_query_cntx();

	if(N < 128)
		update_mr_nr(cake_cntx, MR_MAX, N);
	else
		update_mr_nr(cake_cntx, MR_MAX, NR_MAX);

	int alg = 2;
	sp_pack_t* sp_pack = file_to_sp_pack(N, p, cake_cntx, sch, alg, argv[4]);
	M = sp_pack->M;
	K = sp_pack->K;
	nz = sp_pack->nnz;
	density = ((float) nz) / ((float) (((float) M) * ((float) K)));

	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));
	// initialize B
    srand(time(NULL));
	// rand_init(B, K, N);

	printf("M = %d, K = %d, N = %d, cores = %d, nz = %d\n", M,K,N,p, nz);
 	printf("mr = %d, nr = %d\n", 
 		cake_cntx->mr, cake_cntx->nr);


    float ressss;
    float tttmp[18];
    int flushsz=10000000;
    diff_t = 0.0;
    
    if(setup) exit(1);

    if(dram) {
	    for(int i = 0; i < ntrials; i++) {
			diff_t += rosko_sgemm_compressed(argv[4], B, C, M, N, K, p, cake_cntx, 
										density, NULL, sp_pack, 1, 0, 1, 0, sch, alg);
	    }
    } else {

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
										density, NULL, sp_pack, 1, 0, 1, 0, sch, alg);
			// diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density);
			// diff_t += cake_sgemm(A, B, C, M, N, K, p, cake_cntx);

	        free(dirty);
	    }
	}

 	printf("time = %f\n", diff_t / ntrials);

    if(!dram) {
        char fname[50];
        snprintf(fname, sizeof(fname), "result_gnn");
        FILE *fp;
        fp = fopen(fname, "a");
        fprintf(fp, "rosko,%s,%d,%d,%d,%d,%f,%f\n",argv[4],M,K,N,p,(1-density)*100,diff_t / ntrials);
        fclose(fp);
    }

	// cake_sgemm_checker(A, B, C, N, M, K);

	free_sp_pack(sp_pack);
	free(B);
	free(C);
	free(cake_cntx);

	return 0;
}




