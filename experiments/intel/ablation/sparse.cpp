#include "rosko.h"


// sparse kernel without density-based reordering
void rosko_sgemm_haswell_6x16(float* A, float* B, float* C, int m, int n, int k, 
							int* nnz_outer, int* k_inds,  int* loc_m) {

	int m_cnt1, m_cnt2, m_cnt3, m_cnt4;
	__m256 a, b1, b2;
	__m256 c[6*2];

	c[0]  = _mm256_loadu_ps(C);
	c[1]  = _mm256_loadu_ps(C + 8);
	c[2]  = _mm256_loadu_ps(C + 16);
	c[3]  = _mm256_loadu_ps(C + 24);
	c[4]  = _mm256_loadu_ps(C + 32);
	c[5]  = _mm256_loadu_ps(C + 40);
	c[6]  = _mm256_loadu_ps(C + 48);
	c[7]  = _mm256_loadu_ps(C + 56);
	c[8]  = _mm256_loadu_ps(C + 64);
	c[9]  = _mm256_loadu_ps(C + 72);
	c[10] = _mm256_loadu_ps(C + 80);
	c[11] = _mm256_loadu_ps(C + 88);

	int rem = k % 4;
	k -= rem;

	for(int kk = 0; kk < k; kk += 4) { 

		m_cnt1 = nnz_outer[kk];
		m_cnt2 = nnz_outer[kk+1];
		m_cnt3 = nnz_outer[kk+2];
		m_cnt4 = nnz_outer[kk+3];

		b1 = _mm256_load_ps(B);
		b2 = _mm256_load_ps(B + 8);

		for(int j = 0; j < m_cnt1; j++) {
			a = _mm256_broadcast_ss(A++);
			c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
			loc_m++;
		}

		B += n;



		b1 = _mm256_load_ps(B);
		b2 = _mm256_load_ps(B + 8);

		for(int j = 0; j < m_cnt2; j++) {
			a = _mm256_broadcast_ss(A++);
			c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
			loc_m++;
		}

		B += n;



		b1 = _mm256_load_ps(B);
		b2 = _mm256_load_ps(B + 8);

		for(int j = 0; j < m_cnt3; j++) {
			a = _mm256_broadcast_ss(A++);
			c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
			loc_m++;
		}

		B += n;



		b1 = _mm256_load_ps(B);
		b2 = _mm256_load_ps(B + 8);

		for(int j = 0; j < m_cnt4; j++) {
			a = _mm256_broadcast_ss(A++);
			c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
			loc_m++;
		}

		B += n;


	}

	for(int kk = 0; kk < rem; kk++) { 

		m_cnt1 = nnz_outer[kk];
		b1 = _mm256_load_ps(B);
		b2 = _mm256_load_ps(B + 8);

		for(int j = 0; j < m_cnt1; j++) {
			a = _mm256_broadcast_ss(A++);
			c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
			loc_m++;
		}

		B += n;
	}


	_mm256_storeu_ps(C, c[0]);
	_mm256_storeu_ps((C + 8), c[1]);
	_mm256_storeu_ps((C + 16), c[2]);
	_mm256_storeu_ps((C + 24), c[3]);
	_mm256_storeu_ps((C + 32), c[4]);
	_mm256_storeu_ps((C + 40), c[5]);
	_mm256_storeu_ps((C + 48), c[6]);
	_mm256_storeu_ps((C + 56), c[7]);
	_mm256_storeu_ps((C + 64), c[8]);
	_mm256_storeu_ps((C + 72), c[9]);
	_mm256_storeu_ps((C + 80), c[10]);
	_mm256_storeu_ps((C + 88), c[11]);
}



void rosko_sgemm_haswell_6x16(float* A, float* B, float* C, int m, int n, int k, 
							int* nnz_outer, int* loc_m) {

	__m256 a, b1, b2;
	__m256 c[6*2];

	c[0]  = _mm256_loadu_ps(C);
	c[1]  = _mm256_loadu_ps(C + 8);
	c[2]  = _mm256_loadu_ps(C + 16);
	c[3]  = _mm256_loadu_ps(C + 24);
	c[4]  = _mm256_loadu_ps(C + 32);
	c[5]  = _mm256_loadu_ps(C + 40);
	c[6]  = _mm256_loadu_ps(C + 48);
	c[7]  = _mm256_loadu_ps(C + 56);
	c[8]  = _mm256_loadu_ps(C + 64);
	c[9]  = _mm256_loadu_ps(C + 72);
	c[10] = _mm256_loadu_ps(C + 80);
	c[11] = _mm256_loadu_ps(C + 88);


	for(int kk = 0; kk < k; kk++) { 

		const int m_cnt = nnz_outer[kk];

		b1 = _mm256_load_ps(B);
		b2 = _mm256_load_ps(B + 8);

		for(int j = 0; j < m_cnt; j++) {
			a = _mm256_broadcast_ss(A++);
			c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
			loc_m++;
		}

		B += n;
	}

	_mm256_storeu_ps(C, c[0]);
	_mm256_storeu_ps((C + 8), c[1]);
	_mm256_storeu_ps((C + 16), c[2]);
	_mm256_storeu_ps((C + 24), c[3]);
	_mm256_storeu_ps((C + 32), c[4]);
	_mm256_storeu_ps((C + 40), c[5]);
	_mm256_storeu_ps((C + 48), c[6]);
	_mm256_storeu_ps((C + 56), c[7]);
	_mm256_storeu_ps((C + 64), c[8]);
	_mm256_storeu_ps((C + 72), c[9]);
	_mm256_storeu_ps((C + 80), c[10]);
	_mm256_storeu_ps((C + 88), c[11]);
}


