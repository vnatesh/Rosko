#include "rosko.h"





void csr_to_ob_A_sp(float* vals, int* colind_csr, int* rowptr_csr, int* nnz_tiles, int* num_vec_tile,
   char* nnz_outer, int* k_inds, char* loc_m, float* A_p, int M, int m1, int m2, int k1,
   int m_c, int k_c, int m_r, int nz_in, int col_tile_in, int* ret) {
         

   int nnz_col, ind_tile, outer_ind = 0, a_ind = 0;
   float a_tmp = 0;

   int mr_bins = m_r + 1;
   int** cnt = (int**) malloc(mr_bins * sizeof(int*)); // 
   int* cnt_inds = (int*) malloc(mr_bins * sizeof(int)); // number of cols with 0-mr nnz vals

   for(int i = 0; i < mr_bins; i++) {
      cnt[i] = (int*) malloc(k_c * sizeof(int));
   }


   for(int m3 = 0; m3 < m_c; m3 += m_r) {

      // convert csr to temporary outer product tile
      float* A = (float*) calloc(k_c * m_r , sizeof(float));

      for(int i = 0; i < m_r; i++) {

         if(m1 + m2 + m3 + i < M) {

            int k_col = rowptr_csr[m3 + i + 1] - rowptr_csr[m3 + i];
            for(int j = 0; j < k_col; j++) {

               if((*colind_csr >= k1) && (*colind_csr < (k1 + k_c))) {
                  A[i*k_c + (*colind_csr - k1)] = *vals;
               } 

               colind_csr++;
               vals++;
            }
         }
      }

      // print_mat(A, m_r, k_c);

      // create k_inds, nnz_outer, loc_m from outer product tile
      ind_tile = 0;
      memset(cnt_inds, 0, mr_bins*sizeof(int));

      for(int i = 0; i < k_c; i++) {

         nnz_col = 0;

         for(int j = 0; j < m_r; j++) {

               if(A[i + j*k_c] != 0) {
                  nnz_col++;
               }
         }

         cnt[nnz_col][cnt_inds[nnz_col]++] = i;
      }


       for(int c = m_r; c > 0; c--) {

           if(!cnt_inds[c]) {
              continue;
           }

           for(int i = 0; i < cnt_inds[c]; i++) {

               for(int j = 0; j < m_r; j++) {
                 
                  a_tmp = A[cnt[c][i] + j*k_c];
                   if(a_tmp != 0) {
                     A_p[a_ind + ind_tile] = a_tmp;
                     loc_m[a_ind + ind_tile++] = j;
                   }
               }

              k_inds[outer_ind] = cnt[c][i];
              nnz_outer[outer_ind++] = c;
           }
       }

       // outer_ind += cnt_inds[0]; // skip ahead over cols with 0 nonzeros
       a_ind += ind_tile;
       *nnz_tiles = nz_in + a_ind;
       nnz_tiles++;

       *num_vec_tile = col_tile_in + outer_ind;
       num_vec_tile++;
       free(A);
   }

   for(int i = 0; i < mr_bins; i++) {
      free(cnt[i]);
   }

   free(cnt);
   free(cnt_inds);

   ret[0] = outer_ind;
   ret[1] = a_ind;
}



// packing without density-based reordering
void pack_ob_A_sp(float* A, float* A_p, char* nnz_outer, int* k_inds, char* loc_m,
   int M, int K, int m1, int m2, int m_c, int k_c, int m_r, bool pad) {

   int nnz_col, ind_blk, outer_ind = 0, a_ind = 0, empty = 0;
   float a_tmp = 0;

   if(pad) {
      for(int m3 = 0; m3 < m_c; m3 += m_r) {

          ind_blk = 0;
          empty = 0;

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

            if(nnz_col) {
              k_inds[outer_ind] = i;
              nnz_outer[outer_ind++] = nnz_col;
            } else {
              empty++;
            }           
         }

         outer_ind += empty; // skip ahead over cols with 0 nonzeros
         a_ind += m_r*k_c;
      }     
   } 

   else {
      for(int m3 = 0; m3 < m_c; m3 += m_r) {

         ind_blk = 0;
         empty = 0;
         
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

            if(nnz_col) {
              k_inds[outer_ind] = i;
              nnz_outer[outer_ind++] = nnz_col;
            } else {
              empty++;
            }           
         }

        outer_ind += empty; // skip ahead over cols with 0 nonzeros
        a_ind += m_r*k_c;
      }     
   }
}









// packing without density-based reordering
void pack_ob_A_crisko(float* A, float* A_p, char* nnz_outer, int* k_inds, char* loc_m, int* num_vec_tile,
   int M, int K, int m1, int m2, int m_c, int k_c, int m_r, bool pad) {

   int nnz_col, ind_blk, vec_cnt, outer_ind = 0, a_ind = 0, empty = 0;
   float a_tmp = 0;

   if(pad) {
      for(int m3 = 0; m3 < m_c; m3 += m_r) {

          ind_blk = 0;
          empty = 0;
          vec_cnt = 0;

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

            if(nnz_col) {
              k_inds[outer_ind] = i;
              nnz_outer[outer_ind++] = nnz_col;
              vec_cnt++;
            } else {
              empty++;
            }           
         }

         *num_vec_tile++ = vec_cnt;
         outer_ind += empty; // skip ahead over cols with 0 nonzeros
         a_ind += m_r*k_c;
      }     
   } 

   else {
      for(int m3 = 0; m3 < m_c; m3 += m_r) {

         ind_blk = 0;
         empty = 0;
         vec_cnt = 0;
         
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

            if(nnz_col) {
              vec_cnt++;
              k_inds[outer_ind] = i;
              nnz_outer[outer_ind++] = nnz_col;
            } else {
              empty++;
            }           
         }

        *num_vec_tile++ = vec_cnt;
        outer_ind += empty; // skip ahead over cols with 0 nonzeros
        a_ind += m_r*k_c;
      }     
   }
}


// packing B 
void pack_ob_B_sp(float* B, float* B_p, char* nnz_outer, int* k_inds, char* loc_n, int* num_vec_tile,
   int K, int N, int n1, int k_c, int n_c, int n_r, bool pad) {

   int nnz_row, ind_blk, vec_cnt, outer_ind = 0, b_ind = 0, empty = 0;
   float b_tmp = 0;

   if(pad) {
      for(int n3 = 0; n3 < n_c; n3 += n_r) {

         ind_blk = 0;
         empty = 0;
         vec_cnt = 0;

         for(int i = 0; i < k_c; i++) {

            nnz_row = 0;

            for(int j = 0; j < n_r; j++) {

               if((n1 + n3 + j) >=  N) {
                  B_p[b_ind + ind_blk] = 0.0;
               } else {

                  // a_tmp = A[m3*K + i + j*K];
                  b_tmp = B[n3 + i*N + j];
                  if(b_tmp != 0) {
                     B_p[b_ind + ind_blk] = b_tmp;
                     loc_n[b_ind + ind_blk++] = j;
                     nnz_row++;
                  }
               }
            }

            if(nnz_row) {
              k_inds[outer_ind] = i;
              nnz_outer[outer_ind++] = nnz_row;
              vec_cnt++;
            } else {
              empty++;
            }           
         }

        *num_vec_tile++ = vec_cnt;
        outer_ind += empty; // skip ahead over cols with 0 nonzeros
        b_ind += k_c*n_r;
        // num_vec_tile++;
      }     
   } 

   else {
      for(int n3 = 0; n3 < n_c; n3 += n_r) {

         ind_blk = 0;
         empty = 0;
         vec_cnt = 0;

         for(int i = 0; i < k_c; i++) {

            nnz_row = 0;

            for(int j = 0; j < n_r; j++) {

                  // a_tmp = A[m3*K + i + j*K];
               b_tmp = B[n3 + i*N + j];
               if(b_tmp != 0) {
                  B_p[b_ind + ind_blk] = b_tmp;
                  loc_n[b_ind + ind_blk++] = j;
                  nnz_row++;
               }
            }
            
            if(nnz_row) {
              k_inds[outer_ind] = i;
              nnz_outer[outer_ind++] = nnz_row;
              vec_cnt++;
            } else {
              empty++;
            }           
         }


        *num_vec_tile++ = vec_cnt;
        outer_ind += empty; // skip ahead over cols with 0 nonzeros
        b_ind += k_c*n_r;
      }     
   }
}








//// packing with density-based reordering

// void pack_ob_A_sp(float* A, float* A_p, char* nnz_outer, int* k_inds, char* loc_m, 
//    int M, int K, int m1, int m2, int m_c, int k_c, int m_r, bool pad) {

//    int nnz_col, ind_blk, outer_ind = 0, a_ind = 0;
//    float a_tmp = 0;

//    int mr_bins = m_r + 1;
//    int** cnt = (int**) malloc(mr_bins * sizeof(int*));
//    int* cnt_inds = (int*) malloc(mr_bins * sizeof(int));


//    for(int i = 0; i < mr_bins; i++) {
//       cnt[i] = (int*) malloc(k_c * sizeof(int));
//    }

//    if(pad) {

//       for(int m3 = 0; m3 < m_c; m3 += m_r) {

//          ind_blk = 0;
//          memset(cnt_inds, 0, mr_bins*sizeof(int));

//          for(int i = 0; i < k_c; i++) {

//             nnz_col = 0;

//             for(int j = 0; j < m_r; j++) {

//                if((m1 + m2 + m3 + j) < M) {

//                   if(A[m3*K + i + j*K] != 0) {
//                      nnz_col++;
//                   }
//                }
//             }

//             cnt[nnz_col][cnt_inds[nnz_col]++] = i;
//          }

//          // reorder columns in A in descending order of their densities
//          for(int c = m_r; c > 0; c--) {
       
//             if(!cnt_inds[c]) {
//                // ind_blk += 6;
//                continue;
//             }

//             for(int i = 0; i < cnt_inds[c]; i++) {

//                for(int j = 0; j < m_r; j++) {

//                   if((m1 + m2 + m3 + j) >=  M) {
//                      A_p[a_ind + ind_blk] = 0.0;
//                   } else {

//                      a_tmp = A[m3*K + cnt[c][i] + j*K];
//                      if(a_tmp != 0) {
//                         A_p[a_ind + ind_blk] = a_tmp;
//                         loc_m[a_ind + ind_blk++] = j;
//                      }
//                   }
//                }

//                k_inds[outer_ind] = cnt[c][i];
//                nnz_outer[outer_ind++] = c;
//             }

//          }

//          outer_ind += cnt_inds[0]; // skip ahead over cols with 0 nonzeros
//          a_ind += m_r*k_c;
//       }
//    } 

//    else {

//       for(int m3 = 0; m3 < m_c; m3 += m_r) {

//          ind_blk = 0;
//          memset(cnt_inds, 0, mr_bins*sizeof(int));

//          for(int i = 0; i < k_c; i++) {

//             nnz_col = 0;

//             for(int j = 0; j < m_r; j++) {

//                if(A[m3*K + i + j*K] != 0) {
//                   nnz_col++;
//                }
//             }

//             cnt[nnz_col][cnt_inds[nnz_col]++] = i;
//          }


//          for(int c = m_r; c > 0; c--) {

//             if(!cnt_inds[c]) {
//                // ind_blk += 6;
//                continue;
//             }

//             for(int i = 0; i < cnt_inds[c]; i++) {

//                for(int j = 0; j < m_r; j++) {

//                   // A_p[a_ind + ind_blk] = A[m3*K + cnt[c][i] + j*K];
//                   // loc_m[a_ind + ind_blk++] = j;
                  
//                   a_tmp = A[m3*K + cnt[c][i] + j*K];
//                   if(a_tmp != 0) {
//                      A_p[a_ind + ind_blk] = a_tmp;
//                      loc_m[a_ind + ind_blk++] = j;
//                   }
//                }

//                k_inds[outer_ind] = cnt[c][i];
//                nnz_outer[outer_ind++] = c;
//             }

//          }

//          outer_ind += cnt_inds[0]; // skip ahead over cols with 0 nonzeros
//          a_ind += m_r*k_c;
//       }
//    }

//    for(int i = 0; i < mr_bins; i++) {
//       free(cnt[i]);
//    }

//    free(cnt);
//    free(cnt_inds);
// }





// // packing without density-based reordering
// void pack_ob_A_sp(float* A, float* A_p, int* nnz_outer, int* k_inds, int* loc_m, 
//    int M, int K, int m1, int m2, int m_c, int k_c, int m_r, bool pad) {

//    int nnz_col, ind_blk, outer_ind = 0, a_ind = 0;
//    float a_tmp = 0;

//    if(pad) {
//       for(int m3 = 0; m3 < m_c; m3 += m_r) {

//          ind_blk = 0;

//          for(int i = 0; i < k_c; i++) {

//             nnz_col = 0;

//             for(int j = 0; j < m_r; j++) {

//                if((m1 + m2 + m3 + j) >=  M) {
//                   A_p[a_ind + ind_blk] = 0.0;
//                } else {

//                   a_tmp = A[m3*K + i + j*K];
//                   if(a_tmp != 0) {
//                      A_p[a_ind + ind_blk] = a_tmp;
//                      loc_m[a_ind + ind_blk++] = j;
//                      nnz_col++;
//                   }
//                }

//             }

//             nnz_outer[outer_ind++] = nnz_col;
//          }

//          a_ind += m_r*k_c;
//       }     
//    } 

//    else {
//       for(int m3 = 0; m3 < m_c; m3 += m_r) {

//          ind_blk = 0;

//          for(int i = 0; i < k_c; i++) {

//             nnz_col = 0;

//             for(int j = 0; j < m_r; j++) {

//                a_tmp = A[m3*K + i + j*K];
//                if(a_tmp != 0) {
//                   A_p[a_ind + ind_blk] = a_tmp;
//                   loc_m[a_ind + ind_blk++] = j;
//                   nnz_col++;
//                }
//             }

//             nnz_outer[outer_ind++] = nnz_col;
//          }

//          a_ind += m_r*k_c;
//       }     
//    }
// }



// void pack_ob_A_sp(float* A, float* A_p, int* nnz_outer_blk, int* nnz_outer, int* loc_m, 
//    int M, int K, int m1, int m2, int m_c, int k_c, int m_r, bool pad) {

//    int nnz_col, nnz_blk, nnz_ob = 0, ind_ob = 0, outer_ind = 0, outer_blk_ind = 0;
//    float a_tmp = 0;

//    if(pad) {
//       for(int m3 = 0; m3 < m_c; m3 += m_r) {

//          nnz_blk = 0;

//          for(int i = 0; i < k_c; i++) {

//             nnz_col = 0;

//             for(int j = 0; j < m_r; j++) {

//                if((m1 + m2 + m3 + j) >=  M) {
//                   A_p[ind_ob++] = 0.0;
//                } else {

//                   a_tmp = A[m3*K + i + j*K];
//                   if(a_tmp != 0) {
//                      A_p[ind_ob] = a_tmp;
//                      loc_m[ind_ob++] = j+1;
//                      nnz_col++;
//                      // nnz_ob++;
//                   }
//                }

//                // ind_ob++;
//             }

//             nnz_outer[outer_ind++] = nnz_col;
//             nnz_blk += nnz_col;
//          }

//          nnz_outer_blk[outer_blk_ind++] = nnz_blk;
//       }     
//    } 

//    else {
//       for(int m3 = 0; m3 < m_c; m3 += m_r) {

//          nnz_blk = 0;

//          for(int i = 0; i < k_c; i++) {

//             nnz_col = 0;

//             for(int j = 0; j < m_r; j++) {

//                a_tmp = A[m3*K + i + j*K];
//                if(a_tmp != 0) {
//                   A_p[ind_ob] = a_tmp;
//                   loc_m[ind_ob++] = j+1;
//                   nnz_col++;
//                   // nnz_ob++;
//                }

//                // ind_ob++;
//             }

//             nnz_outer[outer_ind++] = nnz_col;
//             nnz_blk += nnz_col;
//          }

//          nnz_outer_blk[outer_blk_ind++] = nnz_blk;
//       }     
//    }

//    // return nnz_ob;
// }
