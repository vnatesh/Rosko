#include "rosko.h"



// packing without density-based reordering
void pack_ob_A_sp(float* A, float* A_p, int* nnz_outer, int* k_inds, int* loc_m, 
   int M, int K, int m1, int m2, int m_c, int k_c, int m_r, bool pad) {

   int nnz_col, ind_blk, outer_ind = 0, a_ind = 0;
   float a_tmp = 0;

   if(pad) {
      for(int m3 = 0; m3 < m_c; m3 += m_r) {

         ind_blk = 0;

         for(int i = 0; i < k_c; i++) {

            nnz_col = 0;

            for(int j = 0; j < m_r; j++) {

               if((m1 + m2 + m3 + j) >=  M) {
                  A_p[a_ind + ind_blk] = 0.0;
               } else {

                  a_tmp = A[m3*K + i + j*K];
                  if(a_tmp != 0) {
                     A_p[a_ind + ind_blk] = a_tmp;
                     loc_m[a_ind + ind_blk++] = j;
                     nnz_col++;
                  }
               }

            }

            nnz_outer[outer_ind++] = nnz_col;
         }

         a_ind += m_r*k_c;
      }     
   } 

   else {
      for(int m3 = 0; m3 < m_c; m3 += m_r) {

         ind_blk = 0;

         for(int i = 0; i < k_c; i++) {

            nnz_col = 0;

            for(int j = 0; j < m_r; j++) {

               a_tmp = A[m3*K + i + j*K];
               if(a_tmp != 0) {
                  A_p[a_ind + ind_blk] = a_tmp;
                  loc_m[a_ind + ind_blk++] = j;
                  nnz_col++;
               }
            }

            nnz_outer[outer_ind++] = nnz_col;
         }

         a_ind += m_r*k_c;
      }     
   }
}



void pack_ob_A_single_buf(float* A, float* A_p, int M, int K, int m1, int m2, int m_c, int k_c, int m_r, bool pad) {

   int ind_ob = 0;

   for(int m3 = 0; m3 < m_c; m3 += m_r) {
      for(int i = 0; i < k_c; i++) {
         for(int j = 0; j < m_r; j++) {

            if((m1 + m2 + m3 + j) >=  M) {
               A_p[ind_ob] = 0.0;
            } else {
               // printf("PAD IND %d\n", m3*K + i + j*K);

               A_p[ind_ob] = A[m3*K + i + j*K];
            }

            ind_ob++;
            // printf("ind_ob %d\n", ind_ob);
         }
      }
   }     

}





void pack_ob_B_single_buf(float* B, float* B_p, int K, int N, int n1,
            int k_c, int n_c, int n_r, bool pad_n) {

   int ind_ob = 0;

   if(pad_n) {
      for(int n2 = 0; n2 < n_c; n2 += n_r) {
         for(int i = 0; i < k_c; i++) {
            for(int j = 0; j < n_r; j++) {
               if((n1 + n2 + j) >=  N) {
                     B_p[ind_ob] = 0.0;
               } else {
                  // B_p[ind1 + local_ind + (k1/k_c)*k_c*n_c1] = B[n1 + k1*N + n2 + i*N + j];
                  B_p[ind_ob] = B[n2 + i*N + j];
               }
               ind_ob++;
            }
         }
      }
   }
   
   else {
      for(int n2 = 0; n2 < n_c; n2 += n_r) {
         for(int i = 0; i < k_c; i++) {
            for(int j = 0; j < n_r; j++) {
               B_p[ind_ob] = B[n2 + i*N + j];
               ind_ob++;
            }

            // _mm256_store_ps (&B_p[ind_ob], _mm256_load_ps(&B[n2 + i*N]));
            // _mm256_store_ps (&B_p[ind_ob + 8], _mm256_load_ps(&B[n2 + i*N + 8]));
            // ind_ob += n_r;
         }
      }
   }
}




void pack_ob_C_single_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
            int m_c, int n_c, int m_r, int n_r, bool pad_m, bool pad_n) {

   int ind_ob = 0;

   if(pad_m || pad_n) {

      for(int n2 = 0; n2 < n_c; n2 += n_r) {
         for(int m3 = 0; m3 < m_c; m3 += m_r) {
            for(int i = 0; i < m_r; i++) {
               for(int j = 0; j < n_r; j++) {
                  if((n1 + n2 + j) >= N  ||  (m1 + m2 + m3 + i) >=  M) {
                     C_p[ind_ob] = 0.0; // padding
                  } else {
                     C_p[ind_ob] = C[n2 + m3*N + i*N + j];
                  }
                  ind_ob++;
               }
            }
         }
      }

   } else {

      for(int n2 = 0; n2 < n_c; n2 += n_r) {
         for(int m3 = 0; m3 < m_c; m3 += m_r) {
            for(int i = 0; i < m_r; i++) {
               for(int j = 0; j < n_r; j++) {
                  C_p[ind_ob] = C[n2 + m3*N + i*N + j];
                  ind_ob++;
               }
            }
         }
      }
   }
}




// // initialize an operation block of matrix A
// initialize an operation block of matrix A
void pack_ob_A_multiple_buf(float* A, float* A_p, int M, int K, int m1, int k1, int m2, int m_c, int k_c, int m_r, bool pad) {

   int ind2 = 0;
   
   if(pad) {
      for(int m3 = 0; m3 < m_c; m3 += m_r) {
         for(int i = 0; i < k_c; i++) {
            for(int j = 0; j < m_r; j++) {

               if((m1 + m2 + m3 + j) >=  M) {
                  A_p[ind2] = 0.0;
               } else {
                  A_p[ind2] = A[m1*K + k1 + m2*K + m3*K + i + j*K];
               }

               ind2++;
            }
         }
      }     
   } 

   else {
      for(int m3 = 0; m3 < m_c; m3 += m_r) {
         for(int i = 0; i < k_c; i++) {
            for(int j = 0; j < m_r; j++) {
               A_p[ind2] = A[m1*K + k1 + m2*K + m3*K + i + j*K];
               ind2++;
            }
         }
      }
   }
}



void pack_ob_C_multiple_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
            int m_c, int n_c, int m_r, int n_r, bool pad) {

   int ind2 = 0;

   if(pad) {

      for(int n2 = 0; n2 < n_c; n2 += n_r) {
         for(int m3 = 0; m3 < m_c; m3 += m_r) {
            for(int i = 0; i < m_r; i++) {
               for(int j = 0; j < n_r; j++) {
                  if((n1 + n2 + j) >= N  ||  (m1 + m2 + m3 + i) >=  M) {
                     C_p[ind2] = 0.0; // padding
                  } else {
                     C_p[ind2] = C[n1 + m1*N + m2*N + n2 + m3*N + i*N + j];
                  }
                  ind2++;
               }
            }
         }
      }

   } else {

      for(int n2 = 0; n2 < n_c; n2 += n_r) {
         for(int m3 = 0; m3 < m_c; m3 += m_r) {
            for(int i = 0; i < m_r; i++) {
               for(int j = 0; j < n_r; j++) {
                  C_p[ind2] = C[n1 + m1*N + m2*N + n2 + m3*N + i*N + j];
                  ind2++;
               }
            }
         }
      }
   }
}



void unpack_ob_C_multiple_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
            int m_c, int n_c, int m_r, int n_r) {

   int ind2 = 0;

   for(int n2 = 0; n2 < n_c; n2 += n_r) {
      for(int m3 = 0; m3 < m_c; m3 += m_r) {
         for(int i = 0; i < m_r; i++) {
            for(int j = 0; j < n_r; j++) {
               if((n1 + n2 + j) < N  &&  (m1 + m2 + m3 + i) <  M) {
                  C[n1 + m1*N + m2*N + n2 + m3*N + i*N + j] = C_p[ind2];
               }
               ind2++;
            }
         }
      }
   }

}


// when writing reorderd rows back to C, 
// write to correct row row_inds[m1 + m2 + m3 + i]

void unpack_ob_C_single_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
            int m_c, int n_c, int m_r, int n_r) {

   int ind2 = 0;

   for(int n2 = 0; n2 < n_c; n2 += n_r) {
      for(int m3 = 0; m3 < m_c; m3 += m_r) {
         for(int i = 0; i < m_r; i++) {
            for(int j = 0; j < n_r; j++) {
               if((n1 + n2 + j) < N  &&  (m1 + m2 + m3 + i) <  M) {
                  // C[n1 + m1*N + m2*N + n2 + m3*N + i*N + j] = C_p[ind2];
                  C[n2 + m3*N + i*N + j] = C_p[ind2];
               }
               ind2++;
            }
         }
      }
   }
}
