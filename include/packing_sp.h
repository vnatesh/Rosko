#include "common_sp.h"


// sparse packing functions
void pack_A_sp_k_first(float* A, float* A_p, int M, int K, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_A_sp_m_first(float* A, float* A_p, int M, int K, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_A_sp_n_first(float* A, float* A_p, int M, int K, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_ob_A_sp(float* A, float* A_p, unsigned char* nnz_outer, int* k_inds, unsigned char* loc_m,
   int M, int K, int m1, int m2, int m_c, int k_c, int m_r, bool pad);
void pack_A_sp(float* A, float* A_p, int M, int K, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch);

void pack_B_sp_k_first(float* B, float* B_p, int K, int N, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_ob_B_sp(float* B, float* B_p, unsigned char* nnz_outer, int* k_inds, unsigned char* loc_n, int* num_vec_tile,
   int K, int N, int n1, int k_c, int n_c, int n_r, bool pad);
void pack_ob_A_crisko(float* A, float* A_p, unsigned char* nnz_outer, int* k_inds, unsigned char* loc_m, int* num_vec_tile,
   int M, int K, int m1, int m2, int m_c, int k_c, int m_r, bool pad);
void pack_A_sp_crisko(float* A, float* A_p, int M, int K, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx);



void pack_A_csr_to_sp_k_first(csr_t* csr, int M, int K, int nz, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx);
void csr_to_ob_A_sp(float* vals, int* colind_csr, int* rowptr_csr, int* nnz_tiles, int* num_vec_tile,
   unsigned char* nnz_outer, int* k_inds, unsigned char* loc_m, float* A_p, int M, int m1, int m2, int k1,
   int m_c, int k_c, int m_r, int nz_in, int col_tile_in, int* ret);



// helper funcs
sp_pack_t* malloc_sp_pack(int M, int K, int nz, blk_dims_t* x, cake_cntx_t* cake_cntx);
void free_sp_pack(sp_pack_t* x);
void sp_pack_to_file(sp_pack_t* sp_pack, char* fname);
// void file_to_sp_pack(sp_pack_t* sp_pack, char* fname);
sp_pack_t* file_to_sp_pack(int N, int p, cake_cntx_t* cake_cntx, enum sched sch, int alg, char* fname);

double mat_to_csr_file(float* A, int M, int K, char* fname);
void test_csr_convert(int M, int K, float sparsity);
csr_t* file_to_csr(char* fname);
void csr_to_mat(float* A, int M, int K, int* rowptr, float* vals, int* colind);
void free_csr(csr_t* x);


void file_to_sp_pack2(sp_pack_t* sp_pack, char* fname);
void sp_pack_to_file2(sp_pack_t* sp_pack,cake_cntx_t* cake_cntx, 
   int M, int K, char* fname);
sp_pack_t* malloc_sp_pack2(int M, int K, cake_cntx_t* cake_cntx);


float* file_to_mat(char* fname);
void mat_to_file(float* A, int M, int K, char* fname);

