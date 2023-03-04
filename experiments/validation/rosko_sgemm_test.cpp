#include "rosko.h"


int main( int argc, char** argv ) {

// exit(1);
	int M, K, N, p, dram, ntrials, alg;
	struct timespec start, end;
	double diff_t;
	float density, sp;
	enum sched sch = KMN;

	M = atoi(argv[1]);
	K = atoi(argv[2]);
	N = atoi(argv[3]);
	p = atoi(argv[4]);
	sp = atof(argv[5]);
	ntrials = atoi(argv[10]);
	dram = atoi(argv[11]);


	// clock_gettime(CLOCK_REALTIME, &start);


	omp_set_num_threads(p);

	// printf("M = %d, K = %d, N = %d, cores = %d, sparsity = %f\n", M,K,N,p, ((float) sp) / 100.0);

	density = (100.0 - sp) / 100.0;

	float* A = (float*) malloc(M * K * sizeof( float ));
	FILE *fptr1 = fopen(argv[12], "rb");
	fread(A, sizeof(float), M*K, fptr1);
	fclose(fptr1);
	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));
	// rand_sparse(A, M, K, sp / 100.0);
	// rand_init(B, K, N);

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

	// blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	// init_sparse_block_dims(M, N, K, p, x, cake_cntx, sch, NULL, density, 4, alg);
	// size_t A_sz = cake_sgemm_packed_A_size(M, K, p, x, cake_cntx, sch) / sizeof(float);
 //    float* A_p = (float*) calloc(A_sz, sizeof(float));
	// sp_pack_t* sp_pack = (sp_pack_t*) malloc(sizeof(sp_pack_t));
	// pack_A_sp(A, A_p, M, K, p, sp_pack, x, cake_cntx, sch);

	// read pre-packed A matrix into sp_pack
	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	init_sparse_block_dims(M, N, K, p, x, cake_cntx, sch, NULL, density, 4, alg);
	sp_pack_t* sp_pack = malloc_sp_pack2(x->M_padded, K, cake_cntx);
	file_to_sp_pack2(sp_pack, argv[9]);



 //    clock_gettime(CLOCK_REALTIME, &end);
 //    long seconds = end.tv_sec - start.tv_sec;
 //    long nanoseconds = end.tv_nsec - start.tv_nsec;
 //    diff_t = seconds + nanoseconds*1e-9;
 //    char fname[50];
 //    snprintf(fname, sizeof(fname), "results");
 //    FILE *fp;
 //    fp = fopen(fname, "a");
 //    fprintf(fp, "%s,setup,%d,%d,%d,%d,%f,%f\n",argv[9],M,K,N,p,sp,diff_t);
 //    fclose(fp);

	// exit(1);

	if(dram) {

		float* B_p;
	    size_t B_sz = cake_sgemm_packed_B_size(K, N, p, x, cake_cntx);
		if(posix_memalign((void**) &B_p, 64, B_sz)) {
			printf("posix memalign error\n");
			exit(1);
		}

		pack_B(B, B_p, K, N, p, x, cake_cntx, sch);


	    diff_t = 0.0;
	    for(int i = 0; i < ntrials; i++) {
			// diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density);
			diff_t += rosko_sgemm(A, B_p, C, M, N, K, p, cake_cntx, density, NULL, 1, sp_pack, 1, 1, 0, sch, alg);
	    }

	} else {

	    float ressss;
	    float tttmp[18];
	    int flushsz=cake_cntx->L3 / sizeof(float);
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

			// diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density);
			diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, NULL, 1, sp_pack, 0, 1, 0, sch, alg);
	        free(dirty);
	    }
	}

	// printf("rosko,%d,%d,%d,%f\n",M,K,N, diff_t / ntrials);


	if(!dram) {
	    char fname[50];
	    snprintf(fname, sizeof(fname), "results");
	    FILE *fp;
	    fp = fopen(fname, "a");
	    fprintf(fp, "%s,rosko,%d,%d,%d,%d,%f,%f\n",argv[9],M,K,N,p,sp,diff_t / ntrials);
	    fclose(fp);
	}
	// cake_sgemm_checker(A, B, C, N, M, K);

	free(sp_pack->loc_m); 
	free(sp_pack->nnz_outer); 
	free(sp_pack->k_inds); 
	free(sp_pack->A_sp_p);
	free(sp_pack);
	free(A);
	free(B);
	free(C);

	return 0;
}






// int main( int argc, char** argv ) {

// // exit(1);
// 	int M, K, N, p, dram, ntrials, alg;
// 	struct timespec start, end;
// 	double diff_t;
// 	float density, sp;
// 	enum sched sch = KMN;

// 	M = atoi(argv[1]);
// 	K = atoi(argv[2]);
// 	N = atoi(argv[3]);
// 	p = atoi(argv[4]);
// 	sp = atof(argv[5]);
// 	ntrials = atoi(argv[10]);
// 	dram = atoi(argv[11]);

// 	omp_set_num_threads(p);

// 	printf("M = %d, K = %d, N = %d, cores = %d, sparsity = %f\n", M,K,N,p, ((float) sp) / 100.0);

// 	density = (100.0 - sp) / 100.0;

// 	float* A = (float*) malloc(M * K * sizeof( float ));
// 	float* B = (float*) malloc(K * N * sizeof( float ));
// 	float* C = (float*) calloc(M * N , sizeof( float ));

// 	// initialize A and B
//     srand(time(NULL));
// 	rand_sparse(A, M, K, sp / 100.0);
// 	rand_init(B, K, N);

// 	cake_cntx_t* cake_cntx = cake_query_cntx();

// 	if(density > 0.0001) {
// 		update_mr_nr(cake_cntx, MR_MAX, NR_MAX);
// 		alg = 0;
// 	} else {
// 		update_mr_nr(cake_cntx, MR_MIN, NR_MIN);
// 		alg = 2;
// 	}


// 	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
// 	init_sparse_block_dims(M, N, K, p, x, cake_cntx, sch, argv, density, 4, alg);
// 	size_t A_sz = cake_sgemm_packed_A_size(M, K, p, x, cake_cntx, sch) / sizeof(float);
//     float* A_p = (float*) calloc(A_sz, sizeof(float));
// 	sp_pack_t* sp_pack = (sp_pack_t*) malloc(sizeof(sp_pack_t));
// 	pack_A_sp(A, A_p, M, K, p, sp_pack, x, cake_cntx, sch);


// 	if(dram) {

// 		float* B_p;
// 	    size_t B_sz = cake_sgemm_packed_B_size(K, N, p, x, cake_cntx);
// 		if(posix_memalign((void**) &B_p, 64, B_sz)) {
// 			printf("posix memalign error\n");
// 			exit(1);
// 		}

// 		pack_B(B, B_p, K, N, p, x, cake_cntx, sch);


// 	    diff_t = 0.0;
// 	    for(int i = 0; i < ntrials; i++) {
// 			// diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density);
// 			diff_t += rosko_sgemm(A, B_p, C, M, N, K, p, cake_cntx, density, NULL, 1, sp_pack, 1, 1, 0, sch, alg);
// 	    }

// 	} else {

// 	    float ressss;
// 	    float tttmp[18];
// 	    int flushsz=cake_cntx->L3 / sizeof(float);
// 	    diff_t = 0.0;
	    
// 	    for(int i = 0; i < ntrials; i++) {


// 	        float *dirty = (float *)malloc(flushsz * sizeof(float));
// 	        #pragma omp parallel for
// 	        for (int dirt = 0; dirt < flushsz; dirt++){
// 	            dirty[dirt] += dirt%100;
// 	            tttmp[dirt%18] += dirty[dirt];
// 	        }

// 	        for(int ii =0; ii<18;ii++){
// 	            ressss+= tttmp[ii];
// 	        }

// 			// diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density);
// 			diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, NULL, 1, sp_pack, 0, 1, 0, sch, alg);
// 	        free(dirty);
// 	    }
// 	}

// 	printf("rosko,%d,%d,%d,%f\n",M,K,N, diff_t / ntrials);



// 	if(!dram) {
// 		float rosko = rosko_cpu_DRAM_accesses(M, K, N, p, density, NULL);
// 		float cake = cake_cpu_DRAM_accesses(M, K, N, p, NULL);
// 	    char fname[50];
// 	    snprintf(fname, sizeof(fname), "results");
// 	    FILE *fp;
// 	    fp = fopen(fname, "a");
// 	    fprintf(fp, "%s,rosko_dram,%d,%d,%d,%d,%f,%f\n",argv[9],M,K,N,p,sp,rosko);
// 	    fprintf(fp, "%s,rosko,%d,%d,%d,%d,%f,%f\n",argv[9],M,K,N,p,sp,diff_t / ntrials);
// 	    fclose(fp);
// 	}

// 	cake_sgemm_checker(A, B, C, N, M, K);

// 	free(sp_pack->loc_m); 
// 	free(sp_pack->nnz_outer); 
// 	free(sp_pack->k_inds); 
// 	free(sp_pack->A_sp_p);
// 	free(sp_pack);
// 	free(A);
// 	free(B);
// 	free(C);

// 	return 0;
// }





// int main( int argc, char** argv ) {

// // exit(1);
// 	int M, K, N, p, dram, ntrials, alg;
// 	struct timespec start, end;
// 	double diff_t;
// 	float density, sp;
// 	enum sched sch = KMN;

// 	M = atoi(argv[1]);
// 	K = atoi(argv[2]);
// 	N = atoi(argv[3]);
// 	p = atoi(argv[4]);
// 	sp = atof(argv[5]);
// 	ntrials = atoi(argv[7]);
// 	dram = atoi(argv[8]);

// 	printf("M = %d, K = %d, N = %d, cores = %d, sparsity = %f\n", M,K,N,p, ((float) sp) / 100.0);

// 	density = (100.0 - sp) / 100.0;

// 	float* A = (float*) malloc(M * K * sizeof( float ));
// 	float* B = (float*) malloc(K * N * sizeof( float ));
// 	float* C = (float*) calloc(M * N , sizeof( float ));

// 	// initialize A and B
//     srand(time(NULL));
// 	rand_sparse(A, M, K, sp / 100.0);
// 	rand_init(B, K, N);

// 	cake_cntx_t* cake_cntx = cake_query_cntx();

// 	if(dram) {

// 	    diff_t = 0.0;
// 	    for(int i = 0; i < ntrials; i++) {
// 			// diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density);
// 			// diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, NULL, 1, sp_pack, 0, 1, 0, sch, alg);
// 			diff_t += cake_sgemm(A, B, C, M, N, K, p, cake_cntx);
// 	    }

// 	} else {

// 	    float ressss;
// 	    float tttmp[18];
// 	    int flushsz=200000;
// 	    diff_t = 0.0;
	    
// 	    for(int i = 0; i < ntrials; i++) {


// 	        float *dirty = (float *)malloc(flushsz * sizeof(float));
// 	        #pragma omp parallel for
// 	        for (int dirt = 0; dirt < flushsz; dirt++){
// 	            dirty[dirt] += dirt%100;
// 	            tttmp[dirt%18] += dirty[dirt];
// 	        }

// 	        for(int ii =0; ii<18;ii++){
// 	            ressss+= tttmp[ii];
// 	        }

// 			// diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density);
// 			// diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, NULL, 1, sp_pack, 0, 1, 0, sch, alg);
// 			diff_t += cake_sgemm(A, B, C, M, N, K, p, cake_cntx);
// 	        free(dirty);
// 	    }
// 	}

// 	printf("cake,%d,%d,%d,%f\n",M,K,N, diff_t / ntrials);



// 	if(!dram) {
// 		float rosko = rosko_cpu_DRAM_accesses(M, K, N, p, density);
// 		float cake = cake_cpu_DRAM_accesses(M, K, N, p);
// 	    char fname[50];
// 	    snprintf(fname, sizeof(fname), "results");
// 	    FILE *fp;
// 	    fp = fopen(fname, "a");
// 	    fprintf(fp, "%s,cake_dram,%d,%d,%d,%f,%f\n",argv[6],M,K,N,sp,cake);
// 	    fprintf(fp, "%s,cake,%d,%d,%d,%f,%f\n",argv[6],M,K,N,sp,diff_t / ntrials);
// 	    fclose(fp);
// 	}

// 	// cake_sgemm_checker(A, B, C, N, M, K);

// 	free(A);
// 	free(B);
// 	free(C);

// 	return 0;
// }



