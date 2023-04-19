#include "rosko.h"



void pack_A_sp(float* A, float* A_p, int M, int K, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) {

	switch(sch) {
		case KMN: {
			return pack_A_sp_k_first(A, A_p, M, K, p, sp_pack, x, cake_cntx);
		}
		case MKN: {
			return pack_A_sp_m_first(A, A_p, M, K, p, sp_pack, x, cake_cntx); 
		}
		case NKM: {
			return pack_A_sp_n_first(A, A_p, M, K, p, sp_pack, x, cake_cntx);
		}
		default: {
			printf("unknown schedule\n");
			exit(1);
		}	
	}
}



sp_pack_t* malloc_sp_pack(int M, int K, int nz, blk_dims_t* x, cake_cntx_t* cake_cntx) {

	sp_pack_t* sp_pack = (sp_pack_t*) malloc(sizeof(sp_pack_t));
	sp_pack->mat_sp_p = (float*) calloc(nz, sizeof(float)); // storing only nonzeros of A                                        
	sp_pack->loc = (char*) calloc(nz , sizeof(char)); // array for storing M dim C writeback location for each nnz in A
	           // each value ranges from 0 to mr-1
	sp_pack->nnz_outer = (char*) calloc(nz , sizeof(char)); // storing number of nonzeros 
	                                                 // in each outer prod col of A
	sp_pack->k_inds = (int*) calloc(nz , sizeof(int)); // storing kc_ind 
	                                                 // of each outer prod col of A

	// int M_padded = M + ((M % cake_cntx->mr) ? (cake_cntx->mr - (M % cake_cntx->mr)) : 0);
	sp_pack->nnz_tiles = (int*) calloc((x->M_padded / cake_cntx->mr)*x->Kb + 1 , sizeof(int)); 
	sp_pack->num_vec_tile = (int*) calloc((x->M_padded / cake_cntx->mr)*x->Kb + 1, sizeof(int)); 
	sp_pack->M = M;
	sp_pack->K = K;
	sp_pack->nnz = nz;

	return sp_pack;
}



void free_sp_pack(sp_pack_t* x) {
	free(x->loc);
	free(x->nnz_outer);
	free(x->k_inds);
	free(x->mat_sp_p);
	free(x->nnz_tiles);
	free(x->num_vec_tile);
}



// write matrix packed in rosko format to binary file
// M,K,nnz,nnz_cols,ntiles,loc_m,nnz_outer,k_inds,A_sp_p,nnz_tiles,num_vec_tile
void sp_pack_to_file(sp_pack_t* sp_pack, char* fname) {

	FILE *fptr = fopen(fname, "wb");

	int tmp[5];
	tmp[0] = sp_pack->M;
	tmp[1] = sp_pack->K;
	tmp[2] = sp_pack->nnz;
	tmp[3] = sp_pack->nnz_cols;
	tmp[4] = sp_pack->ntiles;

	fwrite(&tmp, sizeof(int), 5, fptr);
	fwrite(sp_pack->loc, sizeof(char), sp_pack->nnz, fptr);
	fwrite(sp_pack->nnz_outer, sizeof(char), sp_pack->nnz_cols, fptr);
	fwrite(sp_pack->k_inds, sizeof(int), sp_pack->nnz_cols, fptr);
	fwrite(sp_pack->mat_sp_p, sizeof(float), sp_pack->nnz, fptr);
	fwrite(sp_pack->nnz_tiles, sizeof(int), sp_pack->ntiles, fptr);
	fwrite(sp_pack->num_vec_tile, sizeof(int), sp_pack->ntiles, fptr);

	fclose(fptr);
}



sp_pack_t* file_to_sp_pack(int N, int p, cake_cntx_t* cake_cntx, enum sched sch, int alg, char* fname) {

	FILE *fptr = fopen(fname, "rb");

	int tmp[5];
	fread(&tmp, sizeof(int), 5, fptr);
	int M = tmp[0];
	int K = tmp[1];
	int nnz = tmp[2];
	int nnz_cols = tmp[3];
	int ntiles = tmp[4];

	float density = ((float) nnz) / ((float) (((float) M) * ((float) K)));

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	init_sparse_block_dims(M, N, K, p, x, cake_cntx, sch, NULL, density, 4, alg);
	sp_pack_t* sp_pack = malloc_sp_pack(M, K, nnz, x, cake_cntx);

	sp_pack->M = tmp[0];
	sp_pack->K = tmp[1];
	sp_pack->nnz = tmp[2];
	sp_pack->nnz_cols = tmp[3];
	sp_pack->ntiles = tmp[4];

	fread(sp_pack->loc, sizeof(char), sp_pack->nnz, fptr);
	fread(sp_pack->nnz_outer, sizeof(char), sp_pack->nnz_cols, fptr);
	fread(sp_pack->k_inds, sizeof(int), sp_pack->nnz_cols, fptr);
	fread(sp_pack->mat_sp_p, sizeof(float), sp_pack->nnz, fptr);
	fread(sp_pack->nnz_tiles, sizeof(int), sp_pack->ntiles, fptr);
	fread(sp_pack->num_vec_tile, sizeof(int), sp_pack->ntiles, fptr);
	fclose(fptr);
	free(x);
	
	return sp_pack;
}














void mat_to_file(float* A, int M, int K, char* fname) {
	
	FILE *fptr = fopen(fname, "wb");
	fwrite(A, sizeof(float), M*K, fptr);
	fclose(fptr);
}



float* file_to_mat(char* fname) {

	int M, K;

	FILE *fptr = fopen(fname, "rb");
	if (fptr == NULL) {
	   perror("fopen");
	   exit(EXIT_FAILURE);
	}

	int tmp[2];
	fread(&tmp, sizeof(int), 2, fptr);

	M = tmp[0];
	K = tmp[1];
	float* vals = (float*) malloc(M*K * sizeof(float));
	fread(vals, sizeof(float), M*K, fptr);
	printf("M = %d K\n", M, K);
   	fclose(fptr);
   	return vals;
}






// malloc_sp_pack2(x->M_padded, K, cake_cntx);

sp_pack_t* malloc_sp_pack2(int M, int K, cake_cntx_t* cake_cntx) {

	sp_pack_t* sp_pack = (sp_pack_t*) malloc(sizeof(sp_pack_t));
	sp_pack->mat_sp_p = (float*) calloc((M*K), sizeof(float)); // storing only nonzeros of A                                        
	sp_pack->loc = (char*) calloc((M*K), sizeof(char)); // array for storing M dim C writeback location for each nnz in A
	           // each value ranges from 0 to mr-1
	sp_pack->nnz_outer = (char*) calloc(((M*K) / cake_cntx->mr), sizeof(char)); // storing number of nonzeros 
	                                                 // in each outer prod col of A
	sp_pack->k_inds = (int*) calloc(((M*K) / cake_cntx->mr) , sizeof(int)); // storing kc_ind 
	                                                 // of each outer prod col of A
	sp_pack->M = M;
	sp_pack->K = K;
	sp_pack->mr = cake_cntx->mr;
	sp_pack->nr = cake_cntx->nr;

	return sp_pack;
}


// sp_pack_to_file2(sp_pack, cake_cntx, x->M_padded, K, fname);

// write uncompressed matrix packed in rosko format to binary file
// M,K,nnz,nnz_cols,ntiles,loc_m,nnz_outer,k_inds,A_sp_p,nnz_tiles,num_vec_tile
void sp_pack_to_file2(sp_pack_t* sp_pack, cake_cntx_t* cake_cntx, 
	int M, int K, char* fname) {

	FILE *fptr = fopen(fname, "wb");

	int tmp[4];
	tmp[0] = M;
	tmp[1] = K;
	tmp[2] = cake_cntx->mr;
	tmp[3] = cake_cntx->nr;

	fwrite(&tmp, sizeof(int), 4, fptr);
	fwrite(sp_pack->loc, sizeof(char), M*K, fptr);
	fwrite(sp_pack->nnz_outer, sizeof(char), ((M*K) / cake_cntx->mr), fptr);
	fwrite(sp_pack->k_inds, sizeof(int), ((M*K) / cake_cntx->mr), fptr);
	fwrite(sp_pack->mat_sp_p, sizeof(float), M*K, fptr);

	fclose(fptr);
}


void file_to_sp_pack2(sp_pack_t* sp_pack, char* fname) {

	FILE *fptr = fopen(fname, "rb");

	int tmp[4];
	fread(&tmp, sizeof(int), 4, fptr);
	sp_pack->M = tmp[0];
	sp_pack->K = tmp[1];
	sp_pack->mr = tmp[2];
	sp_pack->nr = tmp[3];

	fread(sp_pack->loc, sizeof(char),sp_pack->M*sp_pack->K, fptr);
	fread(sp_pack->nnz_outer, sizeof(char), sp_pack->M*sp_pack->K / sp_pack->mr, fptr);
	fread(sp_pack->k_inds, sizeof(int), sp_pack->M*sp_pack->K / sp_pack->mr, fptr);
	fread(sp_pack->mat_sp_p, sizeof(float), sp_pack->M*sp_pack->K, fptr);
	fclose(fptr);
}




void free_csr(csr_t* x) {
	free(x->rowptr);
	free(x->colind);
	free(x->vals);
	free(x);
}


double mat_to_csr_file(float* A, int M, int K, char* fname) {

	struct timespec start, end;
	long seconds, nanoseconds;
	double diff_t, times;

	clock_gettime(CLOCK_REALTIME, &start);

	float* vals = (float*) malloc(M * K * sizeof(float));
	int* colind = (int*) malloc(M * K * sizeof(int));
	int* rowptr = (int*) malloc((M+1) * sizeof(int));
	rowptr[0] = 0;

	FILE *fptr = fopen(fname, "wb");
	int nz = 0;

	for(int i = 0; i < M; i++) {
		for(int j = 0; j < K; j++) {
			// float tmp = A[i*K + j]; // assumes A stored row-major
			float tmp = A[i + j*M]; // assumes A stored col-major
			if(tmp != 0) {
				vals[nz] = tmp;
				colind[nz] = j;
				nz++;
			}
		}

		rowptr[i+1] = nz;
	}


	clock_gettime(CLOCK_REALTIME, &end);
	seconds = end.tv_sec - start.tv_sec;
	nanoseconds = end.tv_nsec - start.tv_nsec;
	diff_t = seconds + nanoseconds*1e-9;

	int tmp[3];
	tmp[0] = M;
	tmp[1] = K;
	tmp[2] = nz;

	fwrite(&tmp, sizeof(int), 3, fptr);
	fwrite(rowptr, sizeof(int), (M + 1), fptr);
	fwrite(colind, sizeof(int), nz, fptr);
	fwrite(vals, sizeof(float), nz, fptr);

	fclose(fptr);
	free(rowptr); free(vals); free(colind);
	return diff_t;
}



// read in CSR matrix from file
// M,K,nnz,rowptr,colind,vals
csr_t* file_to_csr(char* fname) {

	int M, K, nz;

	FILE *fptr = fopen(fname, "rb");
	if (fptr == NULL) {
	   perror("fopen");
	   exit(EXIT_FAILURE);
	}

	int tmp[3];
	fread(&tmp, sizeof(int), 3, fptr);

	M = tmp[0];
	K = tmp[1];
	nz = tmp[2];

	int* rowptr = (int*) malloc((M + 1) * sizeof(int));
	int* colind = (int*) malloc(nz * sizeof(int));
	float* vals = (float*) malloc(nz * sizeof(float));

	fread(rowptr, sizeof(int), (M + 1), fptr);
	fread(colind, sizeof(int), nz, fptr);
	fread(vals, sizeof(float), nz, fptr);

	// printf("M = %d K = %d nz = %d\n", M, K, nz);

   	fclose(fptr);

	csr_t* csr_ret = (csr_t*) malloc(sizeof(csr_t));
	csr_ret->rowptr = rowptr;
	csr_ret->colind = colind;
	csr_ret->vals = vals;
	csr_ret->M = M;
	csr_ret->K = K;

   	return csr_ret;
}



void csr_to_mat(float* A, int M, int K, int* rowptr, float* vals, int* colind) {

	int ks, ind = 0;

	for(int i = 0; i < M; i++) {
		ks = rowptr[i+1] - rowptr[i];
		for(int j = 0; j < ks; j++) {
			A[i*K + colind[ind]] = vals[ind];
			ind++;
		}
	}
}


void test_csr_convert(int M, int K, float sparsity) {

	char fname[50];
	snprintf(fname, sizeof(fname), "convert_test");
	float* A = (float*) malloc(M * K * sizeof( float ));
	float* A_check = (float*) malloc(M * K * sizeof(float));

    srand(time(NULL));
	rand_sparse(A, M, K, ((float) sparsity) / 100.0);

	int nz = mat_to_csr_file(A, M, K, fname);
	csr_t* csr = file_to_csr(fname);
	csr_to_mat(A_check, M, K, csr->rowptr, csr->vals, csr->colind);
	mat_equals(A, A_check, M, K);
	free(A); free(A_check);

	// for(int i = 0; i < M+1; i++) {
	// 	printf("%d ", csr->rowptr[i]);
	// }
	// printf("\n");

	// for(int i = 0; i < nz; i++) {
	// 	printf("%d ", csr->colind[i]);
	// }
	// printf("\n");


	// for(int i = 0; i < nz; i++) {
	// 	printf("%f ", csr->vals[i]);
	// }
	// printf("\n");
}


