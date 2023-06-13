#include "rosko.h"





float cake_cpu_DRAM_accesses(int M1, int K1, int N1, int p, char* argv[], float type_size) {
	

	cake_cntx_t* cake_cntx = cake_query_cntx();
	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	init_block_dims(M1, N1, K1, p, x, cake_cntx, KMN, argv, type_size);

	int mr = cake_cntx->mr, nr = cake_cntx->nr, mc = x->m_c, Mb = x->Mb, Nb = x->Nb, Kb = x->Kb;
	float alpha = cake_cntx->alpha_n; 

 	printf("\nmc = %d, mc1 = %d, Mb = %d, Nb = %d, Kb = %d, kc = %d, nc = %d, mr = %d, nr = %d,  alpha = %f\n", 
 		x->m_c, x->m_c1, Mb, Nb, Kb, x->k_c, x->n_c, mr, nr, alpha);

	float M = (float) M1, K = (float) K1, N = (float) N1; 

	// float dram_acc = ((( ((float) (M*N*K)) / (alpha*p*mc) + ((float) (M*N*K)) / (p*mc)) + 4.0*(M*N) + 2.0*(M*K + K*N)) / 1e9)*4.0;
	float dram_acc = (((M*K*Nb + N*K*Mb) + 4.0*(M*N) + 2.0*(M*K + K*N)) / 1e9)*type_size;

 	free(x);
	free(cake_cntx);
	return dram_acc;
}


float rosko_cpu_DRAM_accesses(int M1, int K1, int N1, int p, float d, char* argv[], float type_size) {
	
	int alg;
	float dram_acc;

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
	init_sparse_block_dims(M1, N1, K1, p, x, cake_cntx, KMN, argv, d, type_size, alg);
	
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
	float dram_acc1 = (((2.5*d*M*K*Nb + N*K*Mb) + 4.0*M*N + 2.0*K*N) / 1e9)*4.0; // arm


	// On systems with high DRAM BW, 
	// We don't have to fit block in cache. We can use large blocks, allow cache misses and extra mem accesses
	// and we can do this if there's enough dram BW.  
	// IO Caused by read/write C, A/B reads during gemm, A metadata, and B/C packing
	float dram_acc2 = (((2.5*d*M*K*Nb + N*K*Mb + 2.0*M*N*Kb) + 2.0*M*N + 2.0*K*N) / 1e9)*4.0; // intel


	if(MR_MIN == 6 && NR_MIN == 16) {
		dram_acc = dram_acc2;
	} else {
		dram_acc = dram_acc1;
	}


	free(x);
	free(cake_cntx);
	return dram_acc;
}







int main( int argc, char** argv ) {

// exit(1);
	int M, K, N, p;
	float density, sp;
	enum sched sch = KMN;

	M = atoi(argv[1]);
	K = atoi(argv[2]);
	N = atoi(argv[3]);
	p = atoi(argv[4]);
	sp = atof(argv[5]);
	density = (100.0 - sp) / 100.0;

	float rosko = rosko_cpu_DRAM_accesses(M, K, N, p, density, NULL, 4);
	// float cake = cake_cpu_DRAM_accesses(M, K, N, p, NULL, 4);
    char fname[50];
    snprintf(fname, sizeof(fname), "results");
    FILE *fp;
    fp = fopen(fname, "a");
    fprintf(fp, "%s,rosko_dram,%d,%d,%d,%d,%f,%f\n",argv[6],M,K,N,p,sp,rosko);
    fclose(fp);

	return 0;
}




