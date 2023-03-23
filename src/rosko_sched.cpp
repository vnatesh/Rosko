#include "rosko.h"





void schedule_KMN_sp_compressed(sp_pack_t* sp_pack, float* B_p, float* C, float** C_p, int M, int N, int K, int p, 
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
	int m_cb, n_c_t, p_used, core;
	int m1, n1, C_offset = 0;

    // rsc = 1; csc = m_r;
	float* A_p = sp_pack->A_sp_p;
	char* nnz_outer = sp_pack->nnz_outer;
	int* k_inds = sp_pack->k_inds;
	char* loc_m = sp_pack->loc_m;
	int* nnz_tiles = sp_pack->nnz_tiles;
	int* num_col_tile = sp_pack->num_col_tile;


	// printf("Mb = %d, Nb = %d, k_c = %d, k_c1 = %d, mc = %d, nc = %d, mc1 = %d, nc1 = %d\n", 
	// 	Mb, Nb, k_c, k_c1, m_c, n_c, m_c1, n_c1);


	for(n = 0; n < Nb; n++) {

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
				int nnz_tile, num_col;

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

					int tile_ind = (m*p*m_c*Kb + k*m_cb + core*m_c_x) / m_r;
					int b_ind = n*K*n_c + k*k_c*n_c_t;
					int c_ind = n*M_padded*n_c + m*p*m_c*n_c_t + core*m_c_x*n_c_t;


					for(n_reg = 0; n_reg < (n_c_t / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {							

							nnz_tile = nnz_tiles[tile_ind + m_reg];
							num_col = num_col_tile[tile_ind + m_reg];

							kernel_map_sp_new[m_map][n_map](
									&A_p[nnz_tile], 
									&B_p[b_ind + n_reg*k_c_t*n_r],
									&C_p[core][n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
									m_r, n_r, num_col_tile[tile_ind + m_reg + 1] - num_col, 
									&nnz_outer[num_col],
									&k_inds[num_col], 
									&loc_m[nnz_tile]
							);
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





void schedule_KMN_sp_online(float* A, float* B, float* C, float** A_p, 
	char** loc_m, int** k_inds, char** nnz_outer, float* B_p, float** C_p, 
	int M, int N, int K, int p, cake_cntx_t* cake_cntx, blk_dims_t* x) {

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
	int m_cb, n_c_t, p_used, core;

	int m1, n1, n2, C_offset = 0;

	float kappa = 1.0;
	int lda = 1;


	for(n = 0; n < Nb; n++) {

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

			for(k = 0; k < Kb; k++) {

				int k_c_t;

				k_c_t = k_c; 
				if((k == Kb - 1) && k_pad) {
					k_c_t = k_c1;
				}


		    	#pragma omp parallel for private(n2)
		      	for(n2 = 0; n2 < n_c_t; n2 += n_r) {
		        	 // bli_spackm_haswell_asm_16xk(n_r, k_c, &kappa, &B[n1 + k1*N + n2], lda, N, 
		         	//                            &B_p[ind1 + (k1/k_c)*k_c*n_c + n2*k_c], n_r);
		         	B_packing_kernel(n_r, k_c_t, &kappa, &B[n1 + k*k_c*N + n2], lda, N, 
		                                    &B_p[n2*k_c_t], n_r);
		      	}


				// pragma omp here (i_c loop)
				#pragma omp parallel for private(core)
				for(core = 0; core < p_used; core++) {

					// These vars must be private to thread, 
					// otherwise out of bounds memory access possible
					int m_c_t, m_c_x, n_reg, m_reg;
		            bool pad;

		            if((m == Mb - 1) && m_pad) {
		               m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
		               m_c_x = m_c1;
		               pad = (core == (p_l - 1) ? 1 : 0);
		            } else {
		               m_c_t = m_c;
		               m_c_x = m_c;
		               pad = 0;
		            }


		            pack_ob_A_sp(&A[m*p*m_c*K + k*k_c + core*m_c_x*K], 
		            	A_p[core], 
						nnz_outer[core], 
						k_inds[core], 
						loc_m[core], 
						M, K, m*p*m_c, core*m_c_x, m_c_t, k_c_t, m_r, pad
					);

					
					for(n_reg = 0; n_reg < (n_c_t / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {							

							kernel_map_sp[m_map][n_map](
								&A_p[core][m_reg*m_r*k_c_t], 
								&B_p[n_reg*k_c_t*n_r], 
								&C_p[core][n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
								m_r, n_r, k_c_t,
								&nnz_outer[core][m_reg*k_c_t],
								&k_inds[core][m_reg*k_c_t], 
								&loc_m[core][m_reg*m_r*k_c_t]
							);

						}
					}

					memset(nnz_outer[core], 0, m_c * k_c * sizeof(char) / m_r);
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








void schedule_KMN_sp_online_BC(sp_pack_t* sp_pack, float* B, float* B_p, float* C, float** C_p, int M, int N, int K, int p, 
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
	int m_cb, n_c_t, p_used, core;

	int m1, n1, n2, C_offset = 0;

	float kappa = 1.0;
	int lda = 1;

    // rsc = 1; csc = m_r;
	float* A_p = sp_pack->A_sp_p;
	char* nnz_outer = sp_pack->nnz_outer;
	int* k_inds = sp_pack->k_inds;
	char* loc_m = sp_pack->loc_m;


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

			// pragma omp also here possible (j_r loop)
			// for(k = k_start; k != k_end; k += k_inc) {
			for(k = 0; k < Kb; k++) {
				
				int k_c_t;

				k_c_t = k_c; 
				if((k == Kb - 1) && k_pad) {
					k_c_t = k_c1;
				}


		    	#pragma omp parallel for private(n2)
		      	for(n2 = 0; n2 < n_c_t; n2 += n_r) {
		        	 // bli_spackm_haswell_asm_16xk(n_r, k_c, &kappa, &B[n1 + k1*N + n2], lda, N, 
		         	//                            &B_p[ind1 + (k1/k_c)*k_c*n_c + n2*k_c], n_r);
		         	B_packing_kernel(n_r, k_c_t, &kappa, &B[n1 + k*k_c*N + n2], lda, N, 
		                                    &B_p[n2*k_c_t], n_r);
		      	}

				// pragma omp here (i_c loop)
				#pragma omp parallel for private(core)
				for(core = 0; core < p_used; core++) {

					// These vars must be private to thread, 
					// otherwise out of bounds memory access possible
					int m_c_t, m_c_x, n_reg, m_reg;

					if((m == Mb - 1) && m_pad) {
						m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
						m_c_x = m_c1;
					} else {
						m_c_t = m_c;
						m_c_x = m_c; 
					}


					int out_ind = (m*p*m_c*K + k*m_cb*k_c + core*m_c_x*k_c_t) / m_r;
					int a_ind = m*p*m_c*K + k*m_cb*k_c + core*m_c_x*k_c_t;
					int c_ind = n*M_padded*n_c + m*p*m_c*n_c_t + core*m_c_x*n_c_t;
					
					for(n_reg = 0; n_reg < (n_c_t / n_r); n_reg++) {

						for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {							

							// A_p_offset = nnz_outer_blk[a_ind + m_reg];
							kernel_map_sp[m_map][n_map](&A_p[a_ind + m_reg*m_r*k_c_t], 
													&B_p[n_reg*k_c_t*n_r], 
													&C_p[core][n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
													m_r, n_r, k_c_t, 
													&nnz_outer[out_ind + m_reg*k_c_t],
													&k_inds[out_ind + m_reg*k_c_t], 
													&loc_m[a_ind + m_reg*m_r*k_c_t]);


							// cake_spgemm_ukernel(&A_p[a_ind + m_reg*m_r*k_c_t], 
							// 						&B_p[b_ind + n_reg*k_c_t*n_r], 
							// 						&C_p[c_ind + n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
							// 						m_r, n_r, k_c_t, 
							// 						&nnz_outer[out_ind + m_reg*k_c_t],
							// 						&k_inds[out_ind + m_reg*k_c_t], 
							// 						&loc_m[a_ind + m_reg*m_r*k_c_t]);

						}
					}
					// ob_offset += nnz_ob[m*p*Kb + k*p_used + core];
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





void schedule_KMN_sp_online_B(sp_pack_t* sp_pack, float* B, float* B_p, float* C_p, int M, int N, int K, int p, 
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
	int m_cb, n_c_t, p_used, core;

	int m1, n1, n2, C_offset = 0;

	float kappa = 1.0;
	int lda = 1;

    // rsc = 1; csc = m_r;
	float* A_p = sp_pack->A_sp_p;
	char* nnz_outer = sp_pack->nnz_outer;
	int* k_inds = sp_pack->k_inds;
	char* loc_m = sp_pack->loc_m;


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
			} else {
				p_used = p;
				m_cb = p_used*m_c;
			}


			// pragma omp also here possible (j_r loop)
			// for(k = k_start; k != k_end; k += k_inc) {
			for(k = 0; k < Kb; k++) {
				
				int k_c_t;

				k_c_t = k_c; 
				if((k == Kb - 1) && k_pad) {
					k_c_t = k_c1;
				}


		    	#pragma omp parallel for private(n2)
		      	for(n2 = 0; n2 < n_c_t; n2 += n_r) {
		        	 // bli_spackm_haswell_asm_16xk(n_r, k_c, &kappa, &B[n1 + k1*N + n2], lda, N, 
		         	//                            &B_p[ind1 + (k1/k_c)*k_c*n_c + n2*k_c], n_r);
		         	B_packing_kernel(n_r, k_c_t, &kappa, &B[n1 + k*k_c*N + n2], lda, N, 
		                                    &B_p[n2*k_c_t], n_r);
		      	}

				// pragma omp here (i_c loop)
				#pragma omp parallel for private(core)
				for(core = 0; core < p_used; core++) {

					// These vars must be private to thread, 
					// otherwise out of bounds memory access possible
					int m_c_t, m_c_x, n_reg, m_reg;

					if((m == Mb - 1) && m_pad) {
						m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
						m_c_x = m_c1;
					} else {
						m_c_t = m_c;
						m_c_x = m_c; 
					}


					int out_ind = (m*p*m_c*K + k*m_cb*k_c + core*m_c_x*k_c_t) / m_r;
					int a_ind = m*p*m_c*K + k*m_cb*k_c + core*m_c_x*k_c_t;
					int c_ind = n*M_padded*n_c + m*p*m_c*n_c_t + core*m_c_x*n_c_t;
					
					for(n_reg = 0; n_reg < (n_c_t / n_r); n_reg++) {

						for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {							

							// A_p_offset = nnz_outer_blk[a_ind + m_reg];
							kernel_map_sp[m_map][n_map](&A_p[a_ind + m_reg*m_r*k_c_t], 
													&B_p[n_reg*k_c_t*n_r], 
													&C_p[c_ind + n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
													m_r, n_r, k_c_t, 
													&nnz_outer[out_ind + m_reg*k_c_t],
													&k_inds[out_ind + m_reg*k_c_t], 
													&loc_m[a_ind + m_reg*m_r*k_c_t]);


							// cake_spgemm_ukernel(&A_p[a_ind + m_reg*m_r*k_c_t], 
							// 						&B_p[b_ind + n_reg*k_c_t*n_r], 
							// 						&C_p[c_ind + n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
							// 						m_r, n_r, k_c_t, 
							// 						&nnz_outer[out_ind + m_reg*k_c_t],
							// 						&k_inds[out_ind + m_reg*k_c_t], 
							// 						&loc_m[a_ind + m_reg*m_r*k_c_t]);

						}
					}
					// ob_offset += nnz_ob[m*p*Kb + k*p_used + core];
				}
			}
		}
	}
}






void schedule_KMN_sp(sp_pack_t* sp_pack, float* B_p, float* C, float** C_p, int M, int N, int K, int p, 
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
	float* A_p = sp_pack->A_sp_p;
	char* nnz_outer = sp_pack->nnz_outer;
	int* k_inds = sp_pack->k_inds;
	char* loc_m = sp_pack->loc_m;


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

					int out_ind = (m*p*m_c*K + k*m_cb*k_c + core*m_c_x*k_c_t) / m_r;
					int a_ind = m*p*m_c*K + k*m_cb*k_c + core*m_c_x*k_c_t;
					// int a_ind = ((m*p*m_c*Kb + k*m_cb) / m_r) + core*mr_per_core;
					int b_ind = n*K*n_c + k*k_c*n_c_t;
					int c_ind = n*M_padded*n_c + m*p*m_c*n_c_t + core*m_c_x*n_c_t;
					
					for(n_reg = 0; n_reg < (n_c_t / n_r); n_reg++) {

						for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {							

							kernel_map_sp[m_map][n_map](&A_p[a_ind + m_reg*m_r*k_c_t], 
													&B_p[b_ind + n_reg*k_c_t*n_r], 
													&C_p[core][n_reg*m_c_t*n_r + m_reg*m_r*n_r],
													m_r, n_r, k_c_t, 
													&nnz_outer[out_ind + m_reg*k_c_t],
													&k_inds[out_ind + m_reg*k_c_t], 
													&loc_m[a_ind + m_reg*m_r*k_c_t]);

							// cake_spgemm_ukernel(&A_p[a_ind + m_reg*m_r*k_c_t], 
							// 						&B_p[b_ind + n_reg*k_c_t*n_r], 
							// 						&C_p[c_ind + n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
							// 						m_r, n_r, k_c_t, 
							// 						&nnz_outer[out_ind + m_reg*k_c_t],
							// 						&k_inds[out_ind + m_reg*k_c_t], 
							// 						&loc_m[a_ind + m_reg*m_r*k_c_t]);

						}
					}
					// ob_offset += nnz_ob[m*p*Kb + k*p_used + core];
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






void schedule_MKN_sp(sp_pack_t* sp_pack, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x) {


	// copy over block dims to local vars to avoid readibility ussiues with x->
	int m_r = cake_cntx->mr, n_r = cake_cntx->nr;
	int m_map = cake_cntx->m_map, n_map = cake_cntx->n_map;

	int m_c = x->m_c, k_c = x->k_c, n_c = x->n_c;
	int m_c1 = x->m_c1, k_c1 = x->k_c1, n_c1 = x->n_c1;
	int k_c1_last_core = x->k_c1_last_core;
	int k_rem = x->k_rem;
	int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad, n_pad = x->n_pad;
	int Mb = x->Mb, Kb = x->Kb, Nb = x->Nb;
	int M_padded = x->M_padded;

	int m, k, n, m_start, m_end, m_inc, k_start, k_end, k_inc;
	int k_cb, n_c_t, m_c_t, p_used, core;

	float* A_p = sp_pack->A_sp_p;
	char* nnz_outer = sp_pack->nnz_outer;
	int* k_inds = sp_pack->k_inds;
	char* loc_m = sp_pack->loc_m;


	for(n = 0; n < Nb; n++) {

		if(n % 2) {
			k_start = Kb - 1;
			k_end = -1;
			k_inc = -1;
		} else {
			k_start = 0;
			k_end = Kb;
			k_inc = 1;
		}

		n_c_t = n_c;
		if((n == Nb - 1) && n_pad) {
			n_c_t = n_c1;
		}

		for(k = k_start; k != k_end; k += k_inc) {

			if(n % 2) {
				if(k % 2) {
					m_start = 0;
					m_end = Mb;
					m_inc = 1;
				} else {
					m_start = Mb - 1;
					m_end = -1;
					m_inc = -1;
				}
			} else {
				if(k % 2) {
					m_start = Mb - 1;
					m_end = -1;
					m_inc = -1;
				} else {
					m_start = 0;
					m_end = Mb;
					m_inc = 1;
				}
			}

			if((k == Kb - 1) && k_pad) {
				p_used = p_l;
				k_cb = k_rem; 
			} else {
				p_used = p;
				k_cb = p_used*k_c;
			}

			#pragma omp parallel for private(m, m_c_t, core)
			for(int m_ind = 0; m_ind < Mb; m_ind++) {
			// for(m = 0; m < Mb; m++) {

				m = m_start + m_ind*m_inc;

				m_c_t = m_c; 
				if((m == Mb - 1) && m_pad) {
					m_c_t = m_c1;
				}

				// #pragma omp parallel for private(core)
				for(core = 0; core < p_used; core++) {

					// These vars must be private to thread, 
					// otherwise out of bounds memory access possible
					int k_c_t, k_c_x, n_reg, m_reg;

					if((k == Kb - 1) && k_pad) {
						k_c_t = (core == (p_l - 1) ? k_c1_last_core : k_c1);
						k_c_x = k_c1;
					} else {
						k_c_t = k_c;
						k_c_x = k_c;
					}

					int out_ind = (k*M_padded*p*k_c + m*m_c*k_cb + core*m_c_t*k_c_x) / m_r;
					int a_ind = k*M_padded*p*k_c + m*m_c*k_cb + core*m_c_t*k_c_x;
					int b_ind = n*K*n_c + k*p*k_c*n_c_t + core*k_c_x*n_c_t;
					int c_ind = n*M_padded*n_c + m*m_c*n_c_t;

					for(n_reg = 0; n_reg < (n_c_t / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {	

							kernel_map_sp[m_map][n_map](&A_p[a_ind + m_reg*m_r*k_c_t], 
													&B_p[b_ind + n_reg*k_c_t*n_r], 
													&C_p[c_ind + n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
													m_r, n_r, k_c_t, 
													&nnz_outer[out_ind + m_reg*k_c_t],
													&k_inds[out_ind + m_reg*k_c_t], 
													&loc_m[a_ind + m_reg*m_r*k_c_t]);
						}
					}
				}
			}
		}
	}
}






void schedule_NKM_sp(sp_pack_t* sp_pack, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x) {

	// copy over block dims to local vars to avoid readibility ussiues with x->
	int m_r = cake_cntx->mr, n_r = cake_cntx->nr;
	int m_map = cake_cntx->m_map, n_map = cake_cntx->n_map;

	int m_c = x->m_c, k_c = x->k_c, n_c = x->n_c;
	int m_c1 = x->m_c1, k_c1 = x->k_c1, n_c1 = x->n_c1;
	int k_c1_last_core = x->k_c1_last_core;
	int k_rem = x->k_rem;
	int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad, n_pad = x->n_pad;
	int Mb = x->Mb, Kb = x->Kb, Nb = x->Nb;
	int N_padded = x->N_padded;

	int m, k, n, n_start, n_end, n_inc, k_start, k_end, k_inc;
	int k_cb, n_c_t, m_c_t, p_used, core;

	float* A_p = sp_pack->A_sp_p;
	char* nnz_outer = sp_pack->nnz_outer;
	int* k_inds = sp_pack->k_inds;
	char* loc_m = sp_pack->loc_m;


	for(m = 0; m < Mb; m++) {

		if(m % 2) {
			k_start = Kb - 1;
			k_end = -1;
			k_inc = -1;
		} else {
			k_start = 0;
			k_end = Kb;
			k_inc = 1;
		}

		m_c_t = p*m_c;
		if((m == Mb - 1) && m_pad) {
			m_c_t = m_c1;
		}

		for(k = k_start; k != k_end; k += k_inc) {

			if(m % 2) {
				if(k % 2) {
					n_start = 0;
					n_end = Nb;
					n_inc = 1;
				} else {
					n_start = Nb - 1;
					n_end = -1;
					n_inc = -1;
				}
			} else {
				if(k % 2) {
					n_start = Nb - 1;
					n_end = -1;
					n_inc = -1;
				} else {
					n_start = 0;
					n_end = Nb;
					n_inc = 1;
				}
			}

			if((k == Kb - 1) && k_pad) {
				p_used = p_l;
				k_cb = k_rem; 
			} else {
				p_used = p;
				k_cb = p_used*k_c;
			}

			#pragma omp parallel for private(n, n_c_t, core)
			for(int n_ind = 0; n_ind < Nb; n_ind++) {
			// for(n = 0; n < Nb; n++) {

				n = n_start + n_ind*n_inc;
				
				n_c_t = n_c; 
				if((n == Nb - 1) && n_pad) {
					n_c_t = n_c1;
				}

				// #pragma omp parallel for private(core)
				for(core = 0; core < p_used; core++) {

					// These vars must be private to thread, 
					// otherwise out of bounds memory access possible
					int k_c_t, k_c_x, n_reg, m_reg;

					if((k == Kb - 1) && k_pad) {
						k_c_t = (core == (p_l - 1) ? k_c1_last_core : k_c1);
						k_c_x = k_c1;
					} else {
						k_c_t = k_c;
						k_c_x = k_c;
					}

					int out_ind = (m*p*m_c*K + m_c_t*k*p*k_c + core*m_c_t*k_c_x) / m_r;
					int a_ind = m*p*m_c*K + m_c_t*k*p*k_c + core*m_c_t*k_c_x;
					int b_ind = k*p*k_c*N_padded + n*n_c*k_cb + core*k_c_x*n_c_t;
					int c_ind = m*p*m_c*N_padded + m_c_t*n*n_c;

					for(n_reg = 0; n_reg < (n_c_t / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {	

							kernel_map_sp[m_map][n_map](&A_p[a_ind + m_reg*m_r*k_c_t], 
													&B_p[b_ind + n_reg*k_c_t*n_r], 
													&C_p[c_ind + n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
													m_r, n_r, k_c_t, 
													&nnz_outer[out_ind + m_reg*k_c_t],
													&k_inds[out_ind + m_reg*k_c_t], 
													&loc_m[a_ind + m_reg*m_r*k_c_t]);
						}
					}
				}
			}
		}
	}
}


