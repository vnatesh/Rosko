#include "rosko.h"





float cake_cpu_DRAM_accesses(int M1, int K1, int N1, int p, char* argv[]) {
	

	cake_cntx_t* cake_cntx = cake_query_cntx();
	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	init_block_dims(M1, N1, K1, p, x, cake_cntx, KMN, argv);

	int mr = cake_cntx->mr, nr = cake_cntx->nr, mc = x->m_c, Mb = x->Mb, Nb = x->Nb, Kb = x->Kb;
	float alpha = cake_cntx->alpha_n; 

 	printf("\nmc = %d, mc1 = %d, Mb = %d, Nb = %d, Kb = %d, kc = %d, nc = %d, mr = %d, nr = %d,  alpha = %f\n", 
 		x->m_c, x->m_c1, Mb, Nb, Kb, x->k_c, x->n_c, mr, nr, alpha);

	float M = (float) M1, K = (float) K1, N = (float) N1; 

	// float dram_acc = ((( ((float) (M*N*K)) / (alpha*p*mc) + ((float) (M*N*K)) / (p*mc)) + 4.0*(M*N) + 2.0*(M*K + K*N)) / 1e9)*4.0;
	float dram_acc = (((M*K*Nb + N*K*Mb) + 4.0*(M*N) + 2.0*(M*K + K*N)) / 1e9)*4.0;

 	free(x);
	free(cake_cntx);
	return dram_acc;
}


float rosko_cpu_DRAM_accesses(int M1, int K1, int N1, int p, float d, char* argv[]) {
	
	int alg;

	cake_cntx_t* cake_cntx = cake_query_cntx();

	if(d > 0.0001) {
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
	init_sparse_block_dims(M1, N1, K1, p, x, cake_cntx, KMN, argv, d, 4, alg);
	
	int mr = cake_cntx->mr, nr = cake_cntx->nr;
	int mc = x->m_c, kc = x->k_c, nc = x->n_c, Mb = x->Mb, Nb = x->Nb, Kb = x->Kb;
	float alpha = cake_cntx->alpha_n; 

 	printf("\nmc = %d, mc1 = %d, Mb = %d, Nb = %d, Kb = %d, kc = %d " 
 		"nc = %d, mr = %d, nr = %d,  alpha = %f, density = %f, cores = %d\n", 
 		x->m_c, x->m_c1, Mb, Nb, Kb, x->k_c, x->n_c, mr, nr, alpha, d, p);

	float M = (float) M1, K = (float) K1, N = (float) N1; 


	float c_size = alpha*p*mc*nc;
	float b_size = kc*nc;
	float a_size = 2.5*d*p*mc*kc;
	
	printf("ab = %f, abc = %f, L2 = %f\n", 
		a_size + b_size, c_size + a_size + b_size, cake_cntx->L3 / 4.0);


	// float c_fact = ((a_size + b_size + c_size) / (cake_cntx->L3 / (4.0*2))) - 1.0;
	// c_fact = c_fact > 0 ? 1.0 : 0.0;

	// C stationary, IO caused by reading A/B during gemm, A metadata, and B/C packing
	// Assumes A is pre-packed
	// A = A_sp (4) + loc_m (1) + nnz_outer (1)+ k_inds (4) = 2.5 bytes/val on avg
	float dram_acc = (((2.5*d*M*K*Nb + N*K*Mb) + 4.0*M*N + 2.0*K*N) / 1e9)*4.0; // arm


	// On systems with high DRAM BW, 
	// We don't have to fit block in cache. We can use large blocks, allow cache misses and extra mem accesses
	// and we can do this if there's enough dram BW.  
	//IO Caused by read/write C, A/B reads during gemm, A metadata, and B/C packing
	dram_acc = (((2.5*d*M*K*Nb + N*K*Mb + 2.0*M*N*Kb) + 2.0*M*N + 2.0*K*N) / 1e9)*4.0; // intel

	free(x);
	free(cake_cntx);
	return dram_acc;
}



void test_cake(int M, int N, int K,  int p, int ntrials, float sp, char* layer) {


	int mr, nr, alg;
	struct timespec start, end;
	double diff_t;
	float density;
	long seconds, nanoseconds;

	float* A = (float*) malloc(M * K * sizeof( float ));
	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));
	rand_init(A, M, K);
	rand_init(B, K, N);

	cake_cntx_t* cake_cntx = cake_query_cntx();
	
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

		diff_t += cake_sgemm(A, B, C, M, N, K, p, cake_cntx);

        free(dirty);
    }



	printf("cake,%d,%f,%d,%d,%d,%f\n", alg, sp, M, K, N, diff_t / ntrials);
	


	float cake_dram = cake_cpu_DRAM_accesses(M, K, N, p, NULL);
	char fname[50];
	snprintf(fname, sizeof(fname), "results");
	FILE *fp;
	fp = fopen(fname, "a");
	fprintf(fp, "%s,cake,%d,%d,%d,%f,%f,%f\n", layer, M, K, N, sp, 0.0, diff_t / ntrials);
    fprintf(fp, "%s,cake_dram,%d,%d,%d,%f,%f,%f\n",layer, M, K, N, sp, 0.0, cake_dram);
	fclose(fp);

	// cake_sgemm_checker(A, B, C, N, M, K);

	free(A);
	free(B);
	free(C);
	free(cake_cntx);
}



void test_rosko(int M, int N, int K,  int p, int ntrials, float sp, char* layer) {


	int mr, nr, alg;
	struct timespec start, end;
	double diff_t;
	float density;
	long seconds, nanoseconds;

	float* A = (float*) malloc(M * K * sizeof( float ));
	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));
	rand_sparse(A, M, K, ((float) sp) / 100.0);
	rand_init(B, K, N);


	density = (100 - sp) / 100.0;
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
		diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, alg);

        free(dirty);
    }



	printf("rosko,%d,%f,%d,%d,%d,%f\n", alg, sp, M, K, N, diff_t / ntrials);
	

	float rosko_dram = rosko_cpu_DRAM_accesses(M, K, N, p, density, NULL);
	char fname[50];
	snprintf(fname, sizeof(fname), "results");
	FILE *fp;
	fp = fopen(fname, "a");
	fprintf(fp, "%s,rosko,%d,%d,%d,%f,%f,%f\n", layer, M, K, N, sp, 0.0, diff_t / ntrials);
    fprintf(fp, "%s,rosko_dram,%d,%d,%d,%f,%f,%f\n",layer, M, K, N, sp, 0.0, rosko_dram);
	fclose(fp);

	// cake_sgemm_checker(A, B, C, N, M, K);

	free(A);
	free(B);
	free(C);
	free(cake_cntx);
}




void test_svd_UV_Data(int M, int N, int K, int p, int ntrials, float sp, float svd, char* layer) {


	int k_svd, mr, nr, alg;
	struct timespec start, end;
	double diff_t;
	float density, *B_p;
	long seconds, nanoseconds;

	k_svd = (int) (K * ((100.0 - svd) / 100.0));

	float* U = (float*) malloc(M * k_svd * sizeof( float ));
	float* V = (float*) malloc(k_svd * K * sizeof( float ));
	float* A = (float*) malloc(M * K * sizeof( float ));
	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));
	rand_sparse(U, M, k_svd, ((float) sp) / 100.0);
	rand_init(V, k_svd, K);
	rand_init(B, K, N);


	density = (100 - sp) / 100.0;
	cake_cntx_t* cake_cntx = cake_query_cntx();
	

	diff_t = 0.0;
	clock_gettime(CLOCK_REALTIME, &start);

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	init_sparse_block_dims(M, N, K, p, x, cake_cntx, KMN, NULL, density, 4, 2);
	omp_set_num_threads(p);
    size_t B_sz = cake_sgemm_packed_B_size(K, N, p, x, cake_cntx);
	if(posix_memalign((void**) &B_p, 64, B_sz)) {
		printf("posix memalign error\n");
		exit(1);
	}
	pack_B(B, B_p, K, N, p, x, cake_cntx, KMN);

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
    printf("pack B time = %f\n", diff_t);

	printf("M = %d, k_svd = %d K = %d, N = %d, cores = %d, sparsity = %f, svd = %f\n", 
		M,k_svd, K,N,p, ((float) sp) / 100.0, svd);


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

		diff_t += rosko_sgemm(U, V, A, M, K, k_svd, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, alg);
		update_mr_nr(cake_cntx, 8, 12);
		diff_t += cake_sgemm(A, B, C, M, N, K, p, cake_cntx);

        free(dirty);
    }


	printf("svd1_rosko,%d,%f,%d,%d,%d,%f\n", alg, sp, M, K, N, diff_t / ntrials);
	

	float svd_1_dram = rosko_cpu_DRAM_accesses(M, K, k_svd, p, density, NULL)
					 + cake_cpu_DRAM_accesses(M, N, K, p, NULL);

	char fname[50];
	snprintf(fname, sizeof(fname), "results");
	FILE *fp;
	fp = fopen(fname, "a");
	fprintf(fp, "%s,svd_1_rosko,%d,%d,%d,%f,%f,%f\n", layer, M, K, N, sp, svd, diff_t / ntrials);
    fprintf(fp, "%s,svd_1_dram,%d,%d,%d,%f,%f,%f\n",layer, M, K, N, sp, svd, svd_1_dram);
	fclose(fp);

	// cake_sgemm_checker(A, B, C, N, M, K);


	free(U);
	free(V);
	free(A);
	free(B);
	free(C);
	free(cake_cntx);
}






void test_svd_U_VData(int M, int N, int K, int p, int ntrials, float sp, float svd, char* layer) {


	int k_svd, mr, nr, alg;
	struct timespec start, end;
	double diff_t;
	float density;
	long seconds, nanoseconds;

	k_svd = (int) (K * ((100.0 - svd) / 100.0));

	float* V = (float*) malloc(k_svd * K * sizeof( float ));
	float* B = (float*) malloc(K * N * sizeof( float ));
	float* A = (float*) malloc(k_svd * N * sizeof( float ));
	float* U = (float*) malloc(M * k_svd * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));
	rand_sparse(U, M, k_svd, ((float) sp) / 100.0);
	rand_init(V, k_svd, K);
	rand_init(B, K, N);


	density = (100 - sp) / 100.0;
	cake_cntx_t* cake_cntx = cake_query_cntx();
	


	printf("M = %d, k_svd = %d K = %d, N = %d, cores = %d, sparsity = %f, svd = %f\n", 
		M,k_svd, K,N,p, ((float) sp) / 100.0, svd);


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

		update_mr_nr(cake_cntx, 8, 12);

		diff_t += cake_sgemm(V, B, A, k_svd, N, K, p, cake_cntx);
		
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

		diff_t += rosko_sgemm(U, A, C, M, N, k_svd, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, alg);

        free(dirty);
    }



	printf("svd_2_rosko,%d,%f,%d,%d,%d,%f\n", alg, sp, M, K, N, diff_t / ntrials);
	


	float svd_2_dram = cake_cpu_DRAM_accesses(k_svd, N, K, p, NULL)
					 + rosko_cpu_DRAM_accesses(M, N, k_svd, p, density, NULL);

	char fname[50];
	snprintf(fname, sizeof(fname), "results");
	FILE *fp;
	fp = fopen(fname, "a");
	fprintf(fp, "%s,svd_2_rosko,%d,%d,%d,%f,%f,%f\n", layer, M, K, N, sp, svd, diff_t / ntrials);
	fprintf(fp, "%s,svd_2_dram,%d,%d,%d,%f,%f,%f\n", layer, M, K, N, sp, svd, svd_2_dram);
	fclose(fp);

	// cake_sgemm_checker(A, B, C, N, M, K);


	free(U);
	free(V);
	free(A);
	free(B);
	free(C);
	free(cake_cntx);
}



int main( int argc, char** argv ) {
	// run_tests();
	// run_tests_sparse_test();
	// run_tests_sparse();

// exit(1);
	int M, K, N, p, nz, mr, nr, ntrials, alg, A_sz, k_svd;
	struct timespec start, end;
	double diff_t;
	float density, sp;
	float svd, *A_p;
	long seconds, nanoseconds;

	M = atoi(argv[1]);
	K = atoi(argv[2]);
	N = atoi(argv[3]);
	p = atoi(argv[4]);
	sp = atof(argv[5]);
	ntrials = atoi(argv[6]);
	// svd = atof(argv[7]);
	svd = 85.0;

	test_svd_UV_Data(M, N, K, p, ntrials, sp, svd, argv[7]);
	test_svd_U_VData(M, N, K, p, ntrials, sp, svd, argv[7]);
	test_rosko(M, N, K, p, ntrials, sp, argv[7]);
	test_cake(M, N, K, p, ntrials, sp, argv[7]);

	return 0;
}



