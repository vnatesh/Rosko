#include "rosko.h"




// increase mr,nr from default 6x16 according to sparsity 
void rosko_mr_nr(cake_cntx_t* cake_cntx, float density) {

	int k_f = clamp_val(density * cake_cntx->mr, 0, 1);

	if(k_f == 1) {
	    cake_cntx->mr = 20;
	    cake_cntx->nr = 96;
	}

    cake_cntx->m_map = (cake_cntx->mr/MR_FACT) - (MR_MIN / MR_FACT);
    cake_cntx->n_map = (cake_cntx->nr/NR_FACT) - (NR_MIN / NR_FACT);
}







cache_dims_t* get_sparse_cache_dims(int M, int N, int K, int p, 
			cake_cntx_t* cake_cntx, enum sched sch, 
			char* argv[], float density, float type_size, int alg, int mcu, int kcu, int ncu) {


	int mc, mc_ret, nc_ret, a, mc_L2 = 0, mc_L3 = 0, kc_L1 = 0;
	int max_threads = cake_cntx->ncores; // 2-way hyperthreaded
	int mr = cake_cntx->mr;
	int nr = cake_cntx->nr;
	int b = 2*nr;

    cache_dims_t* blk_ret = (cache_dims_t*) malloc(sizeof(cache_dims_t));

	// set schedule to MEMA-derived optimal value or user-defined
	blk_ret->sch = (sch == NA ? 
					derive_schedule(M, N, K, p, mc_ret, cake_cntx) : 
					sch);


	// user-defined tile sizes
	if(mcu && ncu && kcu) {
		// printf("user-defined tiling\n");
		blk_ret->m_c = mcu;
		blk_ret->k_c = kcu;
		blk_ret->n_c = ncu;
		cake_cntx->alpha_n = 1.0;
	// sparsity-aware tiling when A matrix is sparse
	} else if(density > 0.0000001) {
		
		double k_f; // fraction of row vecs of B that must be loaded for mrxkcxnr outer product
		int kc_ret;
		float size_fact = 2.5; // total pack size expressed as factor of only A_sp (e.g., 2.5 times nnz vals)
		float alpha, a_q, b_q, c_q, d = density; 
		// double a_coeff = (d/mr) * ((int) ceil(d * mr)) ;
		double a_coeff = size_fact*d;
		cake_cntx->alpha_n = 1.0;
		alpha = cake_cntx->alpha_n;
		// 3*d*mr*kc + nr*k_f*kc + mr*nr <= L2 (roughly 3*4 = 12 bytes for each float nnz val due to metadata)
		// k_f = clamp_val(density * mr, 0, 1);
		// kc_L1 = (int) (((((double) cake_cntx->L1) / (type_size)) - 
		// 	(mr*nr)) / (3.0*density*mr + k_f*nr));


		// 3*d*mc*kc_L1 + nr*kc_L1 + mc*nr <= L2
		// mc_L2 = (int) (((((double) cake_cntx->L2) / (type_size)) - 
		// 	(nr*kc_L1)) / (3.0*density*kc_L1 + nr));
		// mc_L2 -= (mc_L2 % mr);


		// 3*d*p*mr*kc_L1 + nr*k_f*kc_L1 + alpha*p^2*mc^2 <= L3
		// k_f = clamp_val(p * density * mr, 0, 1);
		// mc_L3 = (int) sqrt(((((double) cake_cntx->L3) / (type_size)) - 
		// 	(3.0*density*p*mr*kc_L1 + nr*k_f*kc_L1)) / (p*p));
		// mc_L3 -= (mc_L3 % mr);


		// printf("sparsity-aware tiling\n");

		if(alg == 0) {

			// 
			mc_L2 = (int)  ((-b + sqrt(b*b + 4*a_coeff*(((double) cake_cntx->L2) / (type_size)))) / (2.0*a_coeff)) ;
			mc_L2 -= (mc_L2 % mr);

			mc_L3 = (int) sqrt((((double) cake_cntx->L3) / (type_size))  
			/ (p * (a_coeff + alpha + alpha*p)));
			mc_L3 -= (mc_L3 % mr);
		}


		if(alg == 1) {
			// 3*d*p*mc*kc + alpha*p*mc*kc + alpha*p^2*mc^2 <= L3
			mc_L3 = (int) sqrt((((double) cake_cntx->L3) / (type_size))  
			/ (p * (size_fact*d + alpha + alpha*p)));
			mc_L3 -= (mc_L3 % mr);

			// 3*d*mc*kc + kc*nr + mc*nr <= L2
			kc_ret = (int) (((((double) cake_cntx->L2) / (type_size)) - 
				(nr*mc_L3)) / (size_fact*d*mc_L3 + nr));
		}


		if(alg == 2) {
			// first find kc assuming all B rows are accessed
			// d*mr*kc + kc*nr + mr*nr <= L2
			float kc_tmp = ((((float) cake_cntx->L2) / (type_size*2)) - (mr*nr)) / (d*mr + nr);
			// kc_tmp / (d*mr)

			kc_ret = (d*mr < 1) ? (int) (kc_tmp / (d*mr)) : (int) kc_tmp;
			kc_ret = kc_ret > K ? K : kc_ret;
			
			// d*p*mc*kc_ret + nr*kc_ret + p^2*mc^2 <= L3 (A/B should be LRU on average, C stationary)
			a_q = p*p;
			b_q = size_fact*d*p*kc_ret;
			c_q = (((float) cake_cntx->L3) / (type_size)) - nr*kc_ret;
			mc_L3 = (int) ((-b_q + sqrt(b_q*b_q + 4*a_q*c_q)) / (2.0*a_q));
			mc_L3 -= (mc_L3 % mr);
		}


		if(alg == 3) {
			// tiling only at L3
			// 2.5*d*p*mc*kc + alpha*p*mc*kc + alpha*p^2*mc^2 <= L3 (A/B should be LRU on average, C stationary)
			float coeff = size_fact*d*p + alpha*p + alpha*p*p;
			kc_ret = (int) sqrt((((float) cake_cntx->L3) / (type_size*2.0)) / coeff);
			mc_L3 = kc_ret - (kc_ret % mr);
		}


		// If nc is going to be larger than N, set nc to N and re-compute mc
		if((p*mc_L3) > N) {

			nc_ret = N;
			nc_ret -= (nc_ret % nr);

			if(alg == 0) {
				// d*p*mc*kc + kc*nc_ret + p*mc*nc_ret <= L3 
				// a_q = size_fact*a_coeff*p;
				a_q = size_fact*d*p;
				b_q = alpha*nc_ret*(1 + p);
				c_q = (((float) cake_cntx->L3) / (type_size));
				mc_L3 = (int) ((-b_q + sqrt(b_q*b_q + 4*a_q*c_q)) / (2.0*a_q));
				mc_L3 -= (mc_L3 % mr);
			}

			if(alg == 2) {

				// d*p*mc*kc_ret + nr*kc_ret + p*mc*nc_ret <= L3 (A/B should be LRU on average, C stationary)
				a_q = size_fact*d*p*kc_ret;
				b_q = p*nc_ret;
				c_q = (((float) cake_cntx->L3) / (type_size)) - nr*kc_ret;
				mc_L3 = (int) (c_q / (a_q + b_q));
				mc_L3 -= (mc_L3 % mr);
			}

			if(alg == 3) {

				a_q = d*p;
				b_q = alpha*nc_ret*(1 + p);
				c_q = (((float) cake_cntx->L3) / (type_size));
				mc_L3 = (int) ((-b_q + sqrt(b_q*b_q + 4*a_q*c_q)) / (2.0*a_q));
				mc_L3 -= (mc_L3 % mr);
			}

			mc_ret = mc_L3;
			if(M < p*mr) {
				mc_ret = mr;
			} else if(M < p*mc_ret) {

				a = (M / p);
				if(a < mr) {
					mc_ret = mr;
				} else {
					a += (mr - (a % mr));
					mc_ret = a;
				}
			}

			nc_ret = nc_ret == 0 ? nr : nc_ret;

		} else {

			mc_ret = mc_L3;
			if(M < p*mr) {
				mc_ret = mr;
			} else if(M < p*mc_ret) {
				
				a = (M / p);
				if(a < mr) {
					mc_ret = mr;
				} else {
					a += (mr - (a % mr));
					mc_ret = a;
				}
			}

			// spMM is always K-first so using nc_ret from KMN
			nc_ret = (int) (p*mc_ret);
			nc_ret -= (nc_ret % nr);
			nc_ret = nc_ret == 0 ? nr : nc_ret;
		}



		if(alg == 0) {
			blk_ret->m_c = mc_L3 < M ? mc_ret : mr;
			blk_ret->k_c = mc_L2 < K ? mc_L2 : K;
		} else {
			blk_ret->m_c = mc_ret;
			blk_ret->k_c = kc_ret;
		}

		blk_ret->n_c = nc_ret;
	} 

	// printf("yo %d %d %d %d\n", blk_ret->m_c, blk_ret->k_c, blk_ret->n_c, alg);
	return blk_ret;
}



void init_sparse_block_dims(int M, int N, int K, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch, 
	char* argv[], float density, float type_size, int alg, int mcu, int kcu, int ncu) {

	int m_r = cake_cntx->mr;
	int n_r = cake_cntx->nr;
	cache_dims_t* cache_dims = get_sparse_cache_dims(M, N, K, p, 
									cake_cntx, sch, argv, density, type_size, alg, mcu, kcu, ncu);
    x->m_c = cache_dims->m_c;
	x->k_c = cache_dims->k_c;
    x->n_c = cache_dims->n_c;
    x->sch = cache_dims->sch;
    free(cache_dims);


	switch(x->sch) {

		case KMN: {

			x->k_pad = (K % x->k_c) ? 1 : 0; 
			x->n_pad = (N % x->n_c) ? 1 : 0; 
			x->m_pad = (M % (p*x->m_c)) ? 1 : 0; 

			x->mr_rem = (int) ceil( ((double) (M % (p*x->m_c))) / m_r) ;
			int mr_per_core = (int) ceil( ((double) x->mr_rem) / p );
			
			if(mr_per_core) 
				x->p_l = (int) ceil( ((double) x->mr_rem) / mr_per_core);
			else
				x->p_l = 0;

			x->nr_rem = (int) ceil( ((double) (N % x->n_c) / n_r)) ;
			x->n_c1 = x->nr_rem * n_r;

			x->m_c1 = mr_per_core * m_r;
			x->m_c1_last_core = (mr_per_core - (x->p_l*mr_per_core - x->mr_rem)) * m_r;
			x->k_c1 = K % x->k_c;

			//number of CB blocks in the M, N, and K dims
			x->Mb = (M / (p*x->m_c)) + x->m_pad;
			x->Nb = (N / x->n_c) + x->n_pad;
			x->Kb = (K / x->k_c) + x->k_pad;

			x->M_padded = (m_r*x->mr_rem + (M / (p*x->m_c))*p*x->m_c);
			x->N_padded = (N - (N % x->n_c)) + x->n_c1;

			break;
		}


		case MKN: {

			x->k_pad = (K % (p*x->k_c)) ? 1 : 0; 
			x->m_pad = (M % x->m_c) ? 1 : 0; 
			x->n_pad = (N % x->n_c) ? 1 : 0;

			x->k_rem = K % (p*x->k_c);
			x->k_c1 = (int) ceil( ((double) x->k_rem) / p);

			if(x->k_c1) 
				x->p_l = (int) ceil( ((double) x->k_rem) / x->k_c1);
			else
				x->p_l = 0;

			x->nr_rem = (int) ceil( ((double) (N % x->n_c) / n_r)) ;
			x->n_c1 = x->nr_rem * n_r;

			x->k_c1_last_core = x->k_rem - x->k_c1*(x->p_l-1);
			x->mr_rem = (int) ceil( ((double) (M % x->m_c)) / m_r);
			x->m_c1 = x->mr_rem * m_r;

			// number of CB blocks in the M, N, and K dims
			x->Mb = (M / x->m_c) + x->m_pad;
			x->Kb = (K / (p*x->k_c)) + x->k_pad;
			x->Nb = (N / x->n_c) + x->n_pad;

			x->M_padded = (M / x->m_c)*x->m_c + x->m_c1;
			x->N_padded = (N - (N % x->n_c)) + x->n_c1;


			break;
		}


		case NKM: {

			x->k_pad = (K % (p*x->k_c)) ? 1 : 0; 
			x->m_pad = (M % (p*x->m_c)) ? 1 : 0; 
			x->n_pad = (N % x->n_c) ? 1 : 0;

			x->k_rem = K % (p*x->k_c);
			x->k_c1 = (int) ceil( ((double) x->k_rem) / p);

			if(x->k_c1) 
				x->p_l = (int) ceil( ((double) x->k_rem) / x->k_c1);
			else
				x->p_l = 0;

			x->nr_rem = (int) ceil( ((double) (N % x->n_c) / n_r)) ;
			x->n_c1 = x->nr_rem * n_r;

			x->k_c1_last_core = x->k_rem - x->k_c1*(x->p_l-1);
			x->mr_rem = (int) ceil( ((double) (M % (p*x->m_c))) / m_r);
			x->m_c1 = x->mr_rem * m_r;

			// number of CB blocks in the M, N, and K dims
			x->Mb = (M / (p*x->m_c)) + x->m_pad;
			x->Kb = (K / (p*x->k_c)) + x->k_pad;
			x->Nb = (N / x->n_c) + x->n_pad;

			x->M_padded = (M / (p*x->m_c))*(p*x->m_c) + x->m_c1;
			x->N_padded = (N - (N % x->n_c)) + x->n_c1;

			break;
		}


		default: {
			printf("unknown schedule\n");
			exit(1);
		}
	}
}

