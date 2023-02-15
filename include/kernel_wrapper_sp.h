#include "common_sp.h"


inline void cake_spgemm_ukernel(float* A_p, float* B_p, float* C_p, 
	int m_r, int n_r, int k_c_t, char* nnz_outer, int* k_inds, char* loc_m) {

#ifdef USE_CAKE_HASWELL
	cake_sp_sgemm_haswell_6x16(A_p, B_p, C_p, m_r, n_r, k_c_t, nnz_outer, k_inds, loc_m);
#elif USE_CAKE_ARMV8
	cake_sp_sgemm_armv8_8x12(A_p, B_p, C_p, m_r, n_r, k_c_t, nnz_outer, k_inds, loc_m);
#endif

}



