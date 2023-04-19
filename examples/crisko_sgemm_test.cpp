#include "rosko.h"
#include <immintrin.h>



void crisko_sgemm_haswell_mxn(float* A, float* B, float* C, unsigned char* nnz_outer_A, unsigned char* nnz_outer_B, 
								int* k_inds_A, int* k_inds_B, unsigned char* loc_m, unsigned char* loc_n,
								int m, int n, int k, int k_A, int k_B) {

	int m_cnt, n_cnt, n_vec, n_vec1, n_rem, k_ind, ka = 0, kb = 0;
	__m256 a, b;

	while(ka < k_A && kb < k_B) {

		// when k_inds don't match, increment rosko packed arrays by num nnz in that vec
		if(k_inds_A[ka] < k_inds_B[kb]) {
			loc_m += nnz_outer_A[ka];
			A += nnz_outer_A[ka];
			ka++;
		} else if(k_inds_B[kb] < k_inds_A[ka]) {
			loc_n += nnz_outer_B[kb];
			B += nnz_outer_B[kb];
			kb++;

		// k_inds of A and B match
		} else {
			m_cnt = nnz_outer_A[ka];
			n_cnt = nnz_outer_B[kb];
			n_rem = n_cnt % 8;
			n_vec = n_cnt / 8; // n_cnt + (8 - n_rem);
			n_vec1 = n_vec + (n_rem ? 1 : 0);
			// k_ind = n*kb;
// printf("yoo %d %d %d\n", n_vec, n_rem, n_cnt);
			__m256 c[m_cnt*n_vec1];

			for(int i = 0; i < m_cnt; i++) {
				a = _mm256_broadcast_ss(A++);
				for(int j = 0; j < n_vec1; j++) {
					b = _mm256_load_ps(B + j*8);
					c[i*n_vec1 + j] =  _mm256_mul_ps(a, b);
				}
			}

			// write final outputs to correct C memory locations (scatter)
			for(int i = 0; i < m_cnt; i++) {
				for(int j = 0; j < n_vec; j++) {
					C[(*loc_m * n) + *loc_n++] += c[i*n_vec1 + j][0];
					C[(*loc_m * n) + *loc_n++] += c[i*n_vec1 + j][1];
					C[(*loc_m * n) + *loc_n++] += c[i*n_vec1 + j][2];
					C[(*loc_m * n) + *loc_n++] += c[i*n_vec1 + j][3];
					C[(*loc_m * n) + *loc_n++] += c[i*n_vec1 + j][4];
					C[(*loc_m * n) + *loc_n++] += c[i*n_vec1 + j][5];
					C[(*loc_m * n) + *loc_n++] += c[i*n_vec1 + j][6];
					C[(*loc_m * n) + *loc_n++] += c[i*n_vec1 + j][7];
				}

				// do remaining few n
				for(int j = 0; j < n_rem; j++) {
					C[(*loc_m * n) + *loc_n++] += c[i*n_vec1 + n_vec][j];
					// printf("%f ", c[i*n_vec1 + n_vec][j]);
				}
				// printf("\n");

				loc_m++;
				loc_n -= (n_vec*8 + n_rem);
			}

			loc_n += nnz_outer_B[kb];
			B += nnz_outer_B[kb];
			ka++;
			kb++;
		}
	}
}






void schedule_KMN_sp_sp(sp_pack_t* sp_pack_A, sp_pack_t* sp_pack_B, 
						float* C, float** C_p, int M, int N, int K, int p, 
						cake_cntx_t* cake_cntx, blk_dims_t* x) {

	// copy over block dims to local vars to avoid readibility ussiues with x->
	int m_r = cake_cntx->mr, n_r = cake_cntx->nr;
	int m_map = cake_cntx->m_map, n_map = cake_cntx->n_map;

	int m_c = x->m_c, k_c = x->k_c, n_c = x->n_c;
	int m_c1 = x->m_c1, k_c1 = x->k_c1, n_c1 = x->n_c1;
	int m_c1_last_core = x->m_c1_last_core;
	int mr_rem = x->mr_rem;
	int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad, n_pad = x->n_pad;
	int Mb = x->Mb, Kb = x->Kb, Nb = x->Nb;
	int M_padded = x->M_padded;

	int m, k, n; //, m_start, m_end, m_inc, k_start, k_end, k_inc;
	int m1, n1, m_cb, n_c_t, p_used, core, C_offset = 0;

    // rsc = 1; csc = m_r;
	float* A_p = sp_pack_A->mat_sp_p;
	unsigned char* nnz_outer_A = sp_pack_A->nnz_outer;
	int* k_inds_A = sp_pack_A->k_inds;
	unsigned char* loc_m = sp_pack_A->loc;
	int* num_col_tile = sp_pack_A->num_vec_tile;

	float* B_p = sp_pack_B->mat_sp_p;
	unsigned char* nnz_outer_B = sp_pack_B->nnz_outer;
	int* k_inds_B = sp_pack_B->k_inds;
	unsigned char* loc_n = sp_pack_B->loc;
	int* num_row_tile = sp_pack_B->num_vec_tile;


	for(n = 0; n < Nb; n++) {

		n_c_t = n_c;
		if((n == Nb - 1) && n_pad) {
			n_c_t = n_c1;
			n1 = (N - (N % n_c));
		} else {
			n_c_t = n_c;
			n1 = n*n_c;
		}

		for(m = 0; m < Mb; m++) {

			if((m == Mb - 1) && m_pad) {
				p_used = p_l;
				m_cb = m_r*mr_rem ; //M % (p*m_c);
				m1 = (M - (M % (p*m_c)));
			} else {
				p_used = p;
				m_cb = p_used*m_c;
				m1 = m*p*m_c;
			}


			// pragma omp here (i_c loop)
			#pragma omp parallel for private(core,k)
			for(core = 0; core < p_used; core++) {

				// These vars must be private to thread, 
				// otherwise out of bounds memory access possible
				int m_c_t, m_c_x, k_c_t, n_reg, m_reg;

				if((m == Mb - 1) && m_pad) {
					m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
					m_c_x = m_c1;
				} else {
					m_c_t = m_c;
					m_c_x = m_c; 
				}

				// pragma omp also here possible (j_r loop)
				// for(k = k_start; k != k_end; k += k_inc) {
				for(k = 0; k < Kb; k++) {
					
					k_c_t = k_c; 
					if((k == Kb - 1) && k_pad) {
						k_c_t = k_c1;
					}

					int a_ind = m*p*m_c*K + k*m_cb*k_c + core*m_c_x*k_c_t;
					int out_ind_a = a_ind / m_r;
					int b_ind = K*n*n_c + k*k_c*n_c_t;
					int out_ind_b = b_ind / n_r;
					int c_ind = n*M_padded*n_c + m*p*m_c*n_c_t + core*m_c_x*n_c_t;
					
					int tile_ind_a = (m*p*m_c*Kb + k*m_cb + core*m_c_x) / m_r;
					int tile_ind_b = (Kb*n*n_c + k*n_c_t) / n_r;

					for(n_reg = 0; n_reg < (n_c_t / n_r); n_reg++) {

						for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {							

							crisko_sgemm_haswell_mxn(&A_p[a_ind + m_reg*m_r*k_c_t], 
													&B_p[b_ind + n_reg*k_c_t*n_r], 
													&C_p[core][n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
													&nnz_outer_A[out_ind_a + m_reg*k_c_t],
													&nnz_outer_B[out_ind_b + n_reg*k_c_t],
													&k_inds_A[out_ind_a + m_reg*k_c_t], 
													&k_inds_B[out_ind_b + n_reg*k_c_t], 
													&loc_m[a_ind + m_reg*m_r*k_c_t],
													&loc_n[b_ind + n_reg*n_r*k_c_t],
													m_r, n_r, k_c_t, 
													num_col_tile[tile_ind_a + m_reg], 
													num_row_tile[tile_ind_b + n_reg]);
						}
					}
				}
			}


			C_offset = m*p*m_c*N + n*n_c;

			#pragma omp parallel for private(core)
			for(core = 0; core < p_used; core++) {

				int m_c_t, m_c_x;

				if((m == Mb - 1) && m_pad) {
					m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
					m_c_x = m_c1;
				} else {
					m_c_t = m_c;
					m_c_x = m_c;
				}

				unpack_ob_C_single_buf(&C[C_offset + core*m_c_x*N], C_p[core], 
					M, N, m1, n1, core*m_c_x, m_c_t, n_c_t, m_r, n_r);

				memset(C_p[core], 0, m_c * n_c * sizeof(float));
			}
		}
	}
}



double crisko_sgemm(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, float density = 0, char* argv[] = NULL, bool packedA = 0, 
	sp_pack_t* sp_pack_A = NULL, bool packedB = 0, sp_pack_t* sp_pack_B = NULL,
	float alpha = 1, float beta = 0, enum sched sch = NA, int alg = 2, int mcu = 0, int kcu = 0, int ncu = 0);


double crisko_sgemm(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, float density, char* argv[], 
	bool packedA, sp_pack_t* sp_pack_A, bool packedB, sp_pack_t* sp_pack_B,
	float alpha, float beta, enum sched sch, int alg, int mcu, int kcu, int ncu) {


	size_t A_sz, B_sz, C_sz;	
	struct timespec start, end, start1, end1;
	long seconds, nanoseconds;
	double diff_t, times;
	float *A_p, *B_p;

	sch = KMN;
	// sch = set_schedule(sch, M, N, K);

	if(cake_cntx == NULL) {
		cake_cntx = cake_query_cntx();
	}

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));

	init_sparse_block_dims(M, N, K, p, x, cake_cntx, sch, argv, density, 4, alg, mcu, kcu, ncu);
	omp_set_num_threads(p);

	if(DEBUG) printf("m_r = %d, n_r = %d\n\n", cake_cntx->mr, cake_cntx->nr);
	if(DEBUG) printf("mc = %d, kc = %d, nc = %d, alpha_n = %f\n", x->m_c, x->k_c, x->n_c, cake_cntx->alpha_n);


	if(sp_pack_A == NULL) {

		clock_gettime(CLOCK_REALTIME, &start);

		A_sz = cake_sgemm_packed_A_size(M, K, p, x, cake_cntx, sch) / sizeof(float);
	    A_p = (float*) calloc(A_sz, sizeof(float));
		sp_pack_A = (sp_pack_t*) malloc(sizeof(sp_pack_t));
		pack_A_sp_crisko(A, A_p, M, K, p, sp_pack_A, x, cake_cntx);

		clock_gettime(CLOCK_REALTIME, &end);
		seconds = end.tv_sec - start.tv_sec;
		nanoseconds = end.tv_nsec - start.tv_nsec;
		diff_t = seconds + nanoseconds*1e-9;
		if(DEBUG) printf("A sparse pack time: %f \n", diff_t ); 
	}


	clock_gettime(CLOCK_REALTIME, &start1);

	if(sp_pack_B == NULL) {

		clock_gettime(CLOCK_REALTIME, &start);

		B_sz = cake_sgemm_packed_B_size(K, N, p, x, cake_cntx) / sizeof(float);
		B_p = (float*) calloc(B_sz, sizeof(float));
		sp_pack_B = (sp_pack_t*) malloc(sizeof(sp_pack_t));
		pack_B_sp_k_first(B, B_p, K, N, p, sp_pack_B, x, cake_cntx);

		clock_gettime(CLOCK_REALTIME, &end);
		seconds = end.tv_sec - start.tv_sec;
		nanoseconds = end.tv_nsec - start.tv_nsec;
		diff_t = seconds + nanoseconds*1e-9;
		if(DEBUG) printf("B sparse pack time: %f \n", diff_t ); 
	}


	float *C_p[p];

	for(int i = 0; i < p; i++) {
		C_p[i] = (float*) calloc(x->m_c * x->n_c, sizeof(float));
	}

	clock_gettime(CLOCK_REALTIME, &start);
// printf("gkhjreiugheruighre\n");
	schedule_KMN_sp_sp(sp_pack_A, sp_pack_B, C, C_p, M, N, K, p, cake_cntx, x);

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("GEMM time: %f \n", diff_t); 	// exit(1);


    clock_gettime(CLOCK_REALTIME, &end1);
    seconds = end1.tv_sec - start1.tv_sec;
    nanoseconds = end1.tv_nsec - start1.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("full gemm time: %f \n", diff_t); 	// exit(1);


	for(int i = 0; i < p; i++) {
		free(C_p[i]);
	}

	times = diff_t;

	if(!packedA) {
		free(sp_pack_A->loc); 
		free(sp_pack_A->nnz_outer); 
		free(sp_pack_A->k_inds); 
		free(sp_pack_A->mat_sp_p);
		free(sp_pack_A);
		free(sp_pack_B->loc); 
		free(sp_pack_B->nnz_outer); 
		free(sp_pack_B->k_inds); 
		free(sp_pack_B->mat_sp_p);
		free(sp_pack_B);
	}

	free(x);

	return times;
}



int main( int argc, char** argv ) {
	// run_tests();
	// run_tests_sparse_test();
// 	run_tests_sparse();

// exit(1);
	int M, K, N, p, nz, mr, nr, ntrials, alg, warmup = 10;
	struct timespec start, end;
	double diff_t;
	float density, sp;

	M = atoi(argv[1]);
	K = atoi(argv[2]);
	N = atoi(argv[3]);
	p = atoi(argv[4]);
	sp = atof(argv[5]);
	ntrials = atoi(argv[6]);
	alg = atoi(argv[7]);

	printf("M = %d, K = %d, N = %d, cores = %d, sparsity = %f\n", M,K,N,p, ((float) sp) / 100.0);

	float* A = (float*) malloc(M * K * sizeof( float ));
	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));
	rand_sparse(A, M, K, ((float) sp) / 100.0);
	rand_sparse(B, K, N, ((float) sp) / 100.0);

	density = (100.0 - sp) / 100.0;

	cake_cntx_t* cake_cntx = cake_query_cntx();
    // update_mr_nr(cake_cntx, 108, 256);
    update_mr_nr(cake_cntx, 30, 128);

	if(density > 0.05) {
        alg = 0;
    } else {
        alg = 2;
    }

	char fname[50];
	snprintf(fname, sizeof(fname), "results_new_sp");
	FILE *fp;
	fp = fopen(fname, "a");

    float ressss;
    float tttmp[18];
    int flushsz = 2*cake_cntx->L3 / sizeof(float);
    diff_t = 0.0;
    
    printf("alg = %d, %d\n", alg, flushsz);



    enum sched sch = KMN;
	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	init_sparse_block_dims(M, N, K, p, x, cake_cntx, sch, NULL, density, 4, alg);

    // print_mat(B, K, N);
	size_t B_sz = cake_sgemm_packed_B_size(K, N, p, x, cake_cntx) / sizeof(float);
    float* B_p = (float*) calloc(B_sz, sizeof(float));
	sp_pack_t* sp_pack = (sp_pack_t*) malloc(sizeof(sp_pack_t));
	pack_B_sp_k_first(B, B_p, K, N, p, sp_pack, x, cake_cntx);

	// for(int i = 0; i < x->N_padded*K; i++) {
	// 	printf("%f ", sp_pack->mat_sp_p[i]);
	// }
	// printf("\n");

	// for(int i = 0; i < N*K / cake_cntx->nr; i++) {
	// 	printf("%d ", sp_pack->nnz_outer[i]);
	// }
	// printf("\n");

	// for(int i = 0; i < (x->N_padded / cake_cntx->nr)*x->Kb; i++) {
	// 	printf("%d ", sp_pack->num_vec_tile[i]);
	// }

	// printf("\n\n\n\n\n\n");

    // print_mat(A, M, K);
	size_t A_sz = cake_sgemm_packed_A_size(M, K, p, x, cake_cntx, sch) / sizeof(float);
    float* A_p = (float*) calloc(A_sz, sizeof(float));
	sp_pack_t* sp_pack_A = (sp_pack_t*) malloc(sizeof(sp_pack_t));
	pack_A_sp_crisko(A, A_p, M, K, p, sp_pack_A, x, cake_cntx);

	// for(int i = 0; i < x->M_padded*K; i++) {
	// 	printf("%f ", sp_pack_A->mat_sp_p[i]);
	// }
	// printf("\n");

	// for(int i = 0; i < M*K / cake_cntx->mr; i++) {
	// 	printf("%d ", sp_pack_A->nnz_outer[i]);
	// }
	// printf("\n");

	// for(int i = 0; i < (x->M_padded / cake_cntx->mr)*x->Kb; i++) {
	// 	printf("%d ", sp_pack_A->num_vec_tile[i]);
	// }
	// printf("\n");
 // //    float* f2 = (float*) malloc(8*sizeof(float));

	// __m256 a, b, c[6*2];
	// a = _mm256_broadcast_ss(A++);
	// b = _mm256_load_ps(B);
	// c[2] =  _mm256_mul_ps(a, b);
	// for(int i = 0; i < 8; i++) {
	// 	*f2++ = c[2][i];
	// }
	// f2-=8;
	// for(int i = 0; i < 8; i++) {
	// 	printf("%f ", f2[i]);
	// }
 //    printf("\n");


 //    float* f1 = (float*) malloc(8*sizeof(float));
 //    _mm256_storeu_ps(f1, c[2]);

 //    for(int i = 0; i < 8; i++) {
 //        printf("%f ", f1[i]);
 //    }
 //    printf("\n");

	// exit(1);

    for(int i = 0; i < (ntrials + warmup); i++) {

        float* dirty = (float *) malloc(flushsz * sizeof(float));
        #pragma omp parallel for
        for (int dirt = 0; dirt < flushsz; dirt++){
            dirty[dirt] += dirt%100;
            tttmp[dirt%18] += dirty[dirt];
        }

        for(int ii =0; ii<18;ii++){
            ressss+= tttmp[ii];
        }


		// diff_t += rosko_sgemm_online_BC(A, B, C, M, N, K, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, alg);
		// diff_t += rosko_sgemm_compressed(argv[8], B, C, M, N, K, p, cake_cntx, 
		// 							density, NULL, sp_pack, 1, 0, 1, 0, KMN, alg);
		// diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, 3);
		// diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, 3, 120, 274, 1024);
		
		if(i < warmup) {
			// diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, alg, 60, 974, 512);
			// float y = crisko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, 
			// 	NULL, 0, NULL, 0, NULL, 1, 0, KMN, alg, 108, 200, 1024);
			// float y = crisko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, 
			// 	NULL, 1, sp_pack_A, 1, sp_pack, 1, 0, KMN, alg);

			float y = rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, alg);


		} else {
			// float y = crisko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, 
			// 	NULL, 0, NULL, 0, NULL, 1, 0, KMN, alg, 108, 200, 1024);
			// float y = crisko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, 
			// 	NULL, 1, sp_pack_A, 1, sp_pack, 1, 0, KMN, alg);

			float y = rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, alg);


			diff_t += y;
		}

        free(dirty);
    }


	printf("%d,%f,%d,%d,%d,%f\n", alg, sp, M, K, N, diff_t / ntrials);
	fprintf(fp, "%d,%f,%d,%d,%d,%f\n", alg, sp, M, K, N, diff_t / ntrials);
	fclose(fp);

	// print_mat(C, M, N);
	cake_sgemm_checker(A, B, C, N, M, K);

	// free(A);
	free(B);
	free(C);
	free(cake_cntx);

	return 0;
}
