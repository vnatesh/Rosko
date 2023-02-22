#include "kernel_wrapper_sp.h"
#include "util_sp.h"
#include "tiling_sp.h"
#include "packing_sp.h"



// sparse MM scheduling
double rosko_sgemm(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, float density = 0, char* argv[] = NULL, bool packedA = 0, 
	sp_pack_t* sp_pack = NULL, bool packedB = 0, 
	float alpha = 1, float beta = 0, enum sched sch = NA, int alg = 2);
void schedule_KMN_sp(sp_pack_t* sp_pack, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_MKN_sp(sp_pack_t* sp_pack, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_NKM_sp(sp_pack_t* sp_pack, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_sp(sp_pack_t* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x, enum sched sch);



double rosko_sgemm_compressed(char* fname, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, float density = 0, char* argv[] = NULL, sp_pack_t* sp_pack = NULL,
	bool packedA = 0, bool packedB = 0, float alpha = 1, float beta = 0, enum sched sch = NA,
	int alg = 2);
void schedule_KMN_sp_compressed(sp_pack_t* sp_pack, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);



double rosko_sgemm_online(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, float density, char* argv[] = NULL, bool packedA = 0, 
	sp_pack_t* sp_pack = NULL, bool packedB = 0, float alpha = 1, float beta = 0, 
	enum sched sch = NA, int alg = 2);

void schedule_KMN_sp_online(float* A, float* B, float* C, float** A_p, 
	char** loc_m, int** k_inds, char** nnz_outer, float* B_p, float** C_p, 
	int M, int N, int K, int p, cake_cntx_t* cake_cntx, blk_dims_t* x);


