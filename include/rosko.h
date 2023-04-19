#include "kernel_wrapper_sp.h"
#include "util_sp.h"
#include "tiling_sp.h"
#include "packing_sp.h"
#include "autotune.h"


// sparse MM scheduling

double rosko_sgemm(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, float density = 0, char* argv[] = NULL, bool packedA = 0, 
	sp_pack_t* sp_pack = NULL, bool packedB = 0, 
	float alpha = 1, float beta = 0, enum sched sch = NA, int alg = 2, int mcu = 0, int kcu = 0, int ncu = 0);
void schedule_KMN_sp(sp_pack_t* sp_pack, float* B_p, float* C, float** C_p, int M, int N, int K, int p, 
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
void schedule_KMN_sp_compressed(sp_pack_t* sp_pack, float* B_p, float* C, float** C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);



double rosko_sgemm_online(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, float density, char* argv[] = NULL, bool packedA = 0, 
	sp_pack_t* sp_pack = NULL, bool packedB = 0, float alpha = 1, float beta = 0, 
	enum sched sch = NA, int alg = 2, int mcu = 0, int kcu = 0, int ncu = 0);

void schedule_KMN_sp_online(float* A, float* B, float* C, float** A_p, 
	unsigned char** loc_m, int** k_inds, unsigned char** nnz_outer, float* B_p, float** C_p, 
	int M, int N, int K, int p, cake_cntx_t* cake_cntx, blk_dims_t* x);




double rosko_sgemm_online_B(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, float density, char* argv[] = NULL, bool packedA = 0, 
	sp_pack_t* sp_pack = NULL, bool packedB = 0, float alpha = 1, float beta = 0, 
	enum sched sch = NA, int alg = 2, int mcu = 0, int kcu = 0, int ncu = 0);

void schedule_KMN_sp_online_B(sp_pack_t* sp_pack, float* B, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);





double rosko_sgemm_online_BC(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, float density, char* argv[] = NULL, bool packedA = 0, 
	sp_pack_t* sp_pack = NULL, bool packedB = 0, float alpha = 1, float beta = 0, 
	enum sched sch = NA, int alg = 2, int mcu = 0, int kcu = 0, int ncu = 0);

void schedule_KMN_sp_online_BC(sp_pack_t* sp_pack, float* B, float* B_p, float* C, float** C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);






// // void crisko_sgemm_haswell_mxn(float* A, float* B, float* C, char* nnz_outer_A, char* nnz_outer_B, 
// // 								int* k_inds_A, int* k_inds_B, char* loc_m, char* loc_n,
// // 								int m, int n, int k, int k_A, int k_B);

// void schedule_KMN_sp_sp(sp_pack_t* sp_pack_A, sp_pack_t* sp_pack_B, 
// 						float* C, float** C_p, int M, int N, int K, int p, 
// 						cake_cntx_t* cake_cntx, blk_dims_t* x);

// double crisko_sgemm(float* A, float* B, float* C, int M, int N, int K, int p, 
// 	cake_cntx_t* cake_cntx, float density = 0, char* argv[] = NULL, bool packedA = 0, 
// 	sp_pack_t* sp_pack_A = NULL, bool packedB = 0, sp_pack_t* sp_pack_B = NULL,
// 	float alpha = 1, float beta = 0, enum sched sch = NA, int alg = 2, int mcu = 0, int kcu = 0, int ncu = 0);



