#include "rosko.h"





double rosko_sgemm(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, float density, char* argv[], 
	bool packedA, sp_pack_t* sp_pack, bool packedB, 
	float alpha, float beta, enum sched sch) {


	size_t A_sz, B_sz, C_sz;	
	struct timespec start, end, start1, end1;
	long seconds, nanoseconds;
	double diff_t, times;
	float *A_p, *B_p, *C_p;

	sch = KMN;
	// sch = set_schedule(sch, M, N, K);

	if(cake_cntx == NULL) {
		cake_cntx = cake_query_cntx();
	}

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));


	clock_gettime(CLOCK_REALTIME, &start1);

	init_sparse_block_dims(M, N, K, p, x, cake_cntx, sch, argv, density);
	omp_set_num_threads(p);

	if(DEBUG) printf("m_r = %d, n_r = %d\n\n", cake_cntx->mr, cake_cntx->nr);
	if(DEBUG) printf("mc = %d, kc = %d, nc = %d, alpha_n = %f\n", x->m_c, x->k_c, x->n_c, cake_cntx->alpha_n);


	if(sp_pack == NULL) {

		clock_gettime(CLOCK_REALTIME, &start);

		A_sz = cake_sgemm_packed_A_size(M, K, p, x, cake_cntx, sch) / sizeof(float);
	    A_p = (float*) calloc(A_sz, sizeof(float));

		sp_pack = (sp_pack_t*) malloc(sizeof(sp_pack_t));

		pack_A_sp(A, A_p, M, K, p, sp_pack, x, cake_cntx, sch);

		clock_gettime(CLOCK_REALTIME, &end);
		seconds = end.tv_sec - start.tv_sec;
		nanoseconds = end.tv_nsec - start.tv_nsec;
		diff_t = seconds + nanoseconds*1e-9;
		if(DEBUG) printf("A sparse pack time: %f \n", diff_t ); 
	}

	if(packedB) {
		B_p = B;
	} else {

		clock_gettime(CLOCK_REALTIME, &start);

	    B_sz = cake_sgemm_packed_B_size(K, N, p, x, cake_cntx);
		if(posix_memalign((void**) &B_p, 64, B_sz)) {
			printf("posix memalign error\n");
			exit(1);
		}

		pack_B(B, B_p, K, N, p, x, cake_cntx, sch);

	    clock_gettime(CLOCK_REALTIME, &end);
	    seconds = end.tv_sec - start.tv_sec;
	    nanoseconds = end.tv_nsec - start.tv_nsec;
	    diff_t = seconds + nanoseconds*1e-9;
		if(DEBUG) printf("B pack time: %f \n", diff_t ); 
	}


	// C = alpha*A*B + beta*C. If beta is !=0, we must explicitly pack C, 
	// otherwise just allocate an empty C_p buffer
	if(beta != 0) {
		clock_gettime(CLOCK_REALTIME, &start);
	    C_sz = cake_sgemm_packed_C_size(M, N, p, x, cake_cntx, sch);
		if(posix_memalign((void**) &C_p, 64, C_sz)) {
			printf("posix memalign error\n");
			exit(1);
		}

		pack_C(C, C_p, M, N, p, x, cake_cntx, sch);

	    clock_gettime(CLOCK_REALTIME, &end);
	    seconds = end.tv_sec - start.tv_sec;
	    nanoseconds = end.tv_nsec - start.tv_nsec;
	    diff_t = seconds + nanoseconds*1e-9;
		if(DEBUG) printf("C pack time: %f \n", diff_t ); 

	} else {
	    C_sz = cake_sgemm_packed_C_size(M, N, p, x, cake_cntx, sch) / sizeof(float);
	    C_p = (float*) calloc(C_sz, sizeof(float));
	}

	clock_gettime(CLOCK_REALTIME, &start);

	schedule_sp(sp_pack, B_p, C_p, M, N, K, p, cake_cntx, x, sch);

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("GEMM time: %f \n", diff_t); 	// exit(1);
	// print_packed_C(C_p, M, N, m_c, n_c);
	// unpack_C(C, C_p, M, N, m_c, n_c, n_r, m_r, p);
	times = diff_t;

	clock_gettime(CLOCK_REALTIME, &start);

	// unpack_C_rsc(C, C_p, M, N, m_c, n_c, n_r, m_r, p, alpha_n); 
	unpack_C(C, C_p, M, N, p, x, cake_cntx, sch); 

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("unpacking time: %f \n", diff_t); 	// exit(1);



    clock_gettime(CLOCK_REALTIME, &end1);
    seconds = end1.tv_sec - start1.tv_sec;
    nanoseconds = end1.tv_nsec - start1.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("full gemm time: %f \n", diff_t); 	// exit(1);


	if(!packedA) {
		free(sp_pack->loc_m); 
		free(sp_pack->nnz_outer); 
		free(sp_pack->k_inds); 
		free(sp_pack->A_sp_p);
		free(sp_pack);
	}

	if(!packedB) free(B_p);
	free(C_p);
	free(x);

	return times;
}







double rosko_sgemm_compressed(char* fname, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, float density, char* argv[], sp_pack_t* sp_pack,
	bool packedA, bool packedB, float alpha, float beta, enum sched sch, int alg) {


	size_t B_sz, C_sz;	
	struct timespec start, end, start1, end1;
	long seconds, nanoseconds;
	double diff_t, times;
	float *B_p, *C_p;
	csr_t* csr;

	sch = KMN;
	// sch = set_schedule(sch, M, N, K);

	if(cake_cntx == NULL) {
		cake_cntx = cake_query_cntx();
	}

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	init_sparse_block_dims(M, N, K, p, x, cake_cntx, sch, argv, density, 4, alg);
	omp_set_num_threads(p);

    if(DEBUG) printf("m_r = %d, n_r = %d\n\n", cake_cntx->mr, cake_cntx->nr);
    if(DEBUG) printf("mc = %d, kc = %d, nc = %d, alpha_n = %f\n", x->m_c, x->k_c, x->n_c, cake_cntx->alpha_n);

	if(sp_pack == NULL) {

		csr = file_to_csr(fname);
		sp_pack_t* sp_pack = malloc_sp_pack(M, K, csr->rowptr[M], x, cake_cntx);
		pack_A_csr_to_sp_k_first(csr, M, K, csr->rowptr[M], p, sp_pack, x, cake_cntx);
		free_csr(csr);
	} 


	if(packedB) {
		B_p = B;
	} else {

		clock_gettime(CLOCK_REALTIME, &start);

	    B_sz = cake_sgemm_packed_B_size(K, N, p, x, cake_cntx);


	// printf("K = %d N = %d Mb = %d, Nb = %d, k_c = %d, k_c1 = %d, p = %d, mc = %d, nc = %d, mc1 = %d, nc1 = %d\n", 
	// 	K, N, x->Mb, x->Nb, x->k_c, x->k_c1, p, x->m_c, x->n_c, x->m_c1, x->n_c1);

		if(posix_memalign((void**) &B_p, 64, B_sz)) {
			printf("posix memalign error\n");
			exit(1);
		}

		pack_B(B, B_p, K, N, p, x, cake_cntx, sch);

	    clock_gettime(CLOCK_REALTIME, &end);
	    seconds = end.tv_sec - start.tv_sec;
	    nanoseconds = end.tv_nsec - start.tv_nsec;
	    diff_t = seconds + nanoseconds*1e-9;
		if(DEBUG) printf("B pack time: %f \n", diff_t ); 
	}


	// C = alpha*A*B + beta*C. If beta is !=0, we must explicitly pack C, 
	// otherwise just allocate an empty C_p buffer
	if(beta != 0) {
		clock_gettime(CLOCK_REALTIME, &start);
	    C_sz = cake_sgemm_packed_C_size(M, N, p, x, cake_cntx, sch);
		if(posix_memalign((void**) &C_p, 64, C_sz)) {
			printf("posix memalign error\n");
			exit(1);
		}

		pack_C(C, C_p, M, N, p, x, cake_cntx, sch);

	    clock_gettime(CLOCK_REALTIME, &end);
	    seconds = end.tv_sec - start.tv_sec;
	    nanoseconds = end.tv_nsec - start.tv_nsec;
	    diff_t = seconds + nanoseconds*1e-9;
		if(DEBUG) printf("C pack time: %f \n", diff_t ); 

	} else {
	    C_sz = cake_sgemm_packed_C_size(M, N, p, x, cake_cntx, sch) / sizeof(float);
	    C_p = (float*) calloc(C_sz, sizeof(float));
	}

	clock_gettime(CLOCK_REALTIME, &start);

	schedule_KMN_sp_test(sp_pack, B_p, C_p, M, N, K, p, cake_cntx, x);

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("GEMM time: %f \n", diff_t); 	// exit(1);
	// print_packed_C(C_p, M, N, m_c, n_c);
	// unpack_C(C, C_p, M, N, m_c, n_c, n_r, m_r, p);
	times = diff_t;

	clock_gettime(CLOCK_REALTIME, &start);

	// unpack_C_rsc(C, C_p, M, N, m_c, n_c, n_r, m_r, p, alpha_n); 
	unpack_C(C, C_p, M, N, p, x, cake_cntx, sch); 

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("unpacking time: %f \n", diff_t); 	// exit(1);



    clock_gettime(CLOCK_REALTIME, &end1);
    seconds = end1.tv_sec - start1.tv_sec;
    nanoseconds = end1.tv_nsec - start1.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("full gemm time: %f \n", diff_t); 	// exit(1);


	if(!packedA) {
		free_sp_pack(sp_pack);
	}

	if(!packedB) free(B_p);

	free(C_p);
	free(x);

	return times;
}



void schedule_sp(sp_pack_t* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x, enum sched sch) {

	switch(sch) {
		case KMN: {
			schedule_KMN_sp(A_p, B_p, C_p, M, N, K, p, cake_cntx, x); 
			break;
		}
		case MKN: {
			schedule_MKN_sp(A_p, B_p, C_p, M, N, K, p, cake_cntx, x); 
			break;
		}
		case NKM: {
			schedule_NKM_sp(A_p, B_p, C_p, M, N, K, p, cake_cntx, x); 
			break;
		}
		default: {
			printf("unknown schedule\n");
			exit(1);
		}	
	}
}

