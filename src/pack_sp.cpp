#include "rosko.h"



void pack_A_csr_to_sp_k_first(csr_t* csr, int M, int K, int nz, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx) {
   
   // copy over block dims to local vars to avoid readibility issues with x->
   int m_r = cake_cntx->mr;

   int m_c = x->m_c, k_c = x->k_c;
   int m_c1 = x->m_c1, k_c1 = x->k_c1;
   int m_c1_last_core = x->m_c1_last_core;
   int mr_rem = x->mr_rem;
   int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad;
   int Mb = x->Mb, Kb = x->Kb;

   sp_pack->nnz_tiles[0] = 0;
   sp_pack->num_vec_tile[0] = 0;

   int m, k, m_cb, k_c_t, p_used, core;
   int nz_curr = 0, val_ind = 0, nz_col_curr = 0, ntiles = 0;
   int* ret_tmp = (int*) malloc(2 * sizeof(int));


   for(m = 0; m < Mb; m++) {

      if((m == Mb - 1) && m_pad) {
         p_used = p_l;
         m_cb = m_r*mr_rem;
      } else {
         p_used = p;
         m_cb = p_used*m_c;
      }

      for(k = 0; k < Kb; k++) {
      
         k_c_t = k_c; 
         if((k == Kb - 1) && k_pad) {
            k_c_t = k_c1;
         }

         for(core = 0; core < p_used; core++) {

            int m_c_t, m_c_x;

            if((m == Mb - 1) && m_pad) {
               m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
               m_c_x = m_c1;
            } else {
               m_c_t = m_c;
               m_c_x = m_c;
            }

            val_ind = csr->rowptr[m*p*m_c + core*m_c_x];

            csr_to_ob_A_sp(
                  &csr->vals[val_ind], 
                  &csr->colind[val_ind], 
                  &csr->rowptr[m*p*m_c + core*m_c_x],
                  &sp_pack->nnz_tiles[ntiles + 1],
                  &sp_pack->num_vec_tile[ntiles + 1],
                  &sp_pack->nnz_outer[nz_col_curr], 
                  &sp_pack->k_inds[nz_col_curr], 
                  &sp_pack->loc[nz_curr],
                  &sp_pack->mat_sp_p[nz_curr], M, m*p*m_c, core*m_c_x,
                  k*k_c, m_c_t, k_c_t, m_r, nz_curr, nz_col_curr, ret_tmp
            );

            nz_col_curr += ret_tmp[0];
            nz_curr += ret_tmp[1];
            ntiles += m_c_t / m_r;
         }
      }
   }


   free(ret_tmp);
   sp_pack->nnz = nz;
   sp_pack->nnz_cols = nz_col_curr;
   sp_pack->ntiles = ntiles;
}



void pack_A_sp_k_first(float* A, float* A_p, int M, int K, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx) {
   
   // copy over block dims to local vars to avoid readibility issues with x->
   int m_r = cake_cntx->mr;

   int m_c = x->m_c, k_c = x->k_c;
   int m_c1 = x->m_c1, k_c1 = x->k_c1;
   int m_c1_last_core = x->m_c1_last_core;
   int mr_rem = x->mr_rem;
   int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad;
   int Mb = x->Mb, Kb = x->Kb;

   int m, k, A_offset = 0, A_p_offset = 0, vec_cnt = 0;
   int m_cb, k_c_t, p_used, core;

   unsigned char* nnz_outer = (unsigned char*) calloc(((x->M_padded*K) / m_r) , sizeof(unsigned char)); // storing number of nonzeros 
                                                                          // in each outer prod col of A

   int* k_inds = (int*) calloc(((x->M_padded*K) / m_r) , sizeof(int)); // storing kc_ind 
                                                                          // of each outer prod col of A

   unsigned char* loc_m = (unsigned char*) calloc(x->M_padded*K , sizeof(unsigned char)); // array for storing M dim C writeback location for each nnz in A
                                    // each value ranges from 0 to mr-1


   for(m = 0; m < Mb; m++) {

      if((m == Mb - 1) && m_pad) {
         p_used = p_l;
         m_cb = m_r*mr_rem;
      } else {
         p_used = p;
         m_cb = p_used*m_c;
      }

      for(k = 0; k < Kb; k++) {
         
         k_c_t = k_c; 
         if((k == Kb - 1) && k_pad) {
            k_c_t = k_c1;
         }

         A_offset = m*p*m_c*K + k*k_c;

         #pragma omp parallel for private(core)
         for(core = 0; core < p_used; core++) {

            int m_c_t, m_c_x;
            bool pad;

            if((m == Mb - 1) && m_pad) {
               m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
               m_c_x = m_c1;
               pad = (core == (p_l - 1) ? 1 : 0);
            } else {
               m_c_t = m_c;
               m_c_x = m_c;
               pad = 0;
            }

            pack_ob_A_sp(&A[A_offset + core*m_c_x*K], &A_p[A_p_offset + core*m_c_x*k_c_t], 
               &nnz_outer[(A_p_offset + core*m_c_x*k_c_t) / m_r], 
               &k_inds[(A_p_offset + core*m_c_x*k_c_t) / m_r], 
               &loc_m[A_p_offset + core*m_c_x*k_c_t],
               M, K, m*p*m_c, core*m_c_x, m_c_t, k_c_t, m_r, pad);
         }

         A_p_offset += m_cb*k_c_t;
      }
   }

   sp_pack->mat_sp_p = A_p;
   sp_pack->loc = loc_m;
   sp_pack->nnz_outer = nnz_outer;
   sp_pack->k_inds = k_inds;

}











void pack_A_sp_crisko(float* A, float* A_p, int M, int K, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx) {
   
   // copy over block dims to local vars to avoid readibility issues with x->
   int m_r = cake_cntx->mr;

   int m_c = x->m_c, k_c = x->k_c;
   int m_c1 = x->m_c1, k_c1 = x->k_c1;
   int m_c1_last_core = x->m_c1_last_core;
   int mr_rem = x->mr_rem;
   int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad;
   int Mb = x->Mb, Kb = x->Kb;

   int m, k, A_offset = 0, A_p_offset = 0, vec_cnt = 0;
   int m_cb, k_c_t, p_used, core;

   unsigned char* nnz_outer = (unsigned char*) calloc(((x->M_padded*K) / m_r) , sizeof(unsigned char)); // storing number of nonzeros 
                                                                          // in each outer prod col of A

   int* k_inds = (int*) calloc(((x->M_padded*K) / m_r) , sizeof(int)); // storing kc_ind 
                                                                          // of each outer prod col of A

   unsigned char* loc_m = (unsigned char*) calloc(x->M_padded*K , sizeof(unsigned char)); // array for storing M dim C writeback location for each nnz in A
                                    // each value ranges from 0 to mr-1

   int* num_vec_tile = (int*) calloc((x->M_padded / m_r)*Kb + 1, sizeof(int)); // num of cols with nnz vals in each tile


   for(m = 0; m < Mb; m++) {

      if((m == Mb - 1) && m_pad) {
         p_used = p_l;
         m_cb = m_r*mr_rem;
      } else {
         p_used = p;
         m_cb = p_used*m_c;
      }

      for(k = 0; k < Kb; k++) {
         
         k_c_t = k_c; 
         if((k == Kb - 1) && k_pad) {
            k_c_t = k_c1;
         }

         A_offset = m*p*m_c*K + k*k_c;

         #pragma omp parallel for private(core)
         for(core = 0; core < p_used; core++) {

            int m_c_t, m_c_x;
            bool pad;

            if((m == Mb - 1) && m_pad) {
               m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
               m_c_x = m_c1;
               pad = (core == (p_l - 1) ? 1 : 0);
            } else {
               m_c_t = m_c;
               m_c_x = m_c;
               pad = 0;
            }

            pack_ob_A_crisko(&A[A_offset + core*m_c_x*K], &A_p[A_p_offset + core*m_c_x*k_c_t], 
               &nnz_outer[(A_p_offset + core*m_c_x*k_c_t) / m_r], 
               &k_inds[(A_p_offset + core*m_c_x*k_c_t) / m_r], 
               &loc_m[A_p_offset + core*m_c_x*k_c_t],
               &num_vec_tile[vec_cnt + (core*m_c_x / m_r)],
               M, K, m*p*m_c, core*m_c_x, m_c_t, k_c_t, m_r, pad);
         }

         vec_cnt += m_cb / m_r;
         A_p_offset += m_cb*k_c_t;
      }
   }

   sp_pack->mat_sp_p = A_p;
   sp_pack->loc = loc_m;
   sp_pack->nnz_outer = nnz_outer;
   sp_pack->k_inds = k_inds;
   sp_pack->num_vec_tile = num_vec_tile;
}



void pack_B_sp_k_first(float* B, float* B_p, int K, int N, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx) {
   
      // copy over block dims to local vars to avoid readibility ussiues with x->
   int n_r = cake_cntx->nr;

   int k_c = x->k_c, n_c = x->n_c;
   int k_c1 = x->k_c1, n_c1 = x->n_c1;
   int k_c1_last_core = x->k_c1_last_core;
   int k_rem = x->k_rem;
   int p_l = x->p_l, k_pad = x->k_pad, n_pad = x->n_pad;
   int Kb = x->Kb, Nb = x->Nb;

   int n, n1, k, B_offset = 0, B_p_offset = 0, vec_cnt = 0;
   int k_c_t, n_c_t, p_used, core;
   bool pad;

   unsigned char* nnz_outer = (unsigned char*) calloc(((x->N_padded*K) / n_r) , sizeof(unsigned char)); // storing number of nonzeros 
                                                                          // in each outer prod row of B

   int* k_inds = (int*) calloc(((x->N_padded*K) / n_r) , sizeof(int)); // storing kc_ind 
                                                                          // of each outer prod row of B

   unsigned char* loc_n = (unsigned char*) calloc(x->N_padded*K , sizeof(unsigned char)); // array for storing M dim C writeback location for each nnz in B


   int* num_vec_tile = (int*) calloc((x->N_padded / n_r)*Kb + 1, sizeof(int)); // num of rows with nnz vals in each tile


   for(n = 0; n < Nb; n++) {

      if((n == Nb - 1) && n_pad) {
         n_c_t = n_c1;
         pad = 1;
         // n1 = (N - (N % n_c));
      } else {
         n_c_t = n_c;
         pad = 0;
         // n1 = n*n_c;
      }

      #pragma omp parallel for private(k, k_c_t)
      for(k = 0; k < Kb; k++) {

         k_c_t = k_c; 
         if((k == Kb - 1) && k_pad) {
            k_c_t = k_c1;
         }

         pack_ob_B_sp(&B[k*k_c*N + n*n_c], &B_p[B_p_offset + k*k_c*n_c_t], 
            &nnz_outer[(B_p_offset +  k*k_c*n_c_t) / n_r], 
            &k_inds[(B_p_offset +  k*k_c*n_c_t) / n_r], 
            &loc_n[B_p_offset +  k*k_c*n_c_t], 
            &num_vec_tile[vec_cnt + k*n_c_t/n_r],
            K, N, n*n_c, k_c_t, n_c_t, n_r, pad);
      }

      vec_cnt += Kb*n_c_t / n_r;
      B_p_offset += K*n_c_t;
   }


   sp_pack->mat_sp_p = B_p;
   sp_pack->loc = loc_n;
   sp_pack->nnz_outer = nnz_outer;
   sp_pack->k_inds = k_inds;
   sp_pack->num_vec_tile = num_vec_tile;
}






void pack_A_sp_m_first(float* A, float* A_p, int M, int K, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx) {
   
      // copy over block dims to local vars to avoid readibility ussiues with x->
   int m_r = cake_cntx->mr;

   int m_c = x->m_c, k_c = x->k_c;
   int m_c1 = x->m_c1, k_c1 = x->k_c1;
   int k_c1_last_core = x->k_c1_last_core;
   int k_rem = x->k_rem;
   int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad;
   int Mb = x->Mb, Kb = x->Kb;


   int m, k, A_offset = 0, A_p_offset = 0;
   int k_cb, m_c_t, p_used, core;

   unsigned char* nnz_outer = (unsigned char*) calloc(((x->M_padded*K) / m_r) , sizeof(int)); // storing number of nonzeros 
                                                                          // in each outer prod col of A

   int* k_inds = (int*) calloc(((x->M_padded*K) / m_r) , sizeof(int)); // storing kc_ind 
                                                                          // of each outer prod col of A

   unsigned char* loc_m = (unsigned char*) calloc(x->M_padded*K , sizeof(int)); // array for storing M dim C writeback location for each nnz in A
                                    // each value ranges from 0 to mr-1

   for(k = 0; k < Kb; k++) {

      if((k == Kb - 1) && k_pad) {
         p_used = p_l;
         k_cb = k_rem; 
      } else {
         p_used = p;
         k_cb = p_used*k_c;
      }

      for(m = 0; m < Mb; m++) {
         
         m_c_t = m_c; 
         if((m == Mb - 1) && m_pad) {
            m_c_t = m_c1;
         }

         A_offset = m*m_c*K + k*p*k_c;

         #pragma omp parallel for private(core)
         for(core = 0; core < p_used; core++) {

            int k_c_t, k_c_x;
            bool pad;

            if((k == Kb - 1) && k_pad) {
               k_c_t = (core == (p_l - 1) ? k_c1_last_core : k_c1);
               k_c_x = k_c1;
               pad = (core == (p_l - 1) ? 1 : 0);
            } else {
               k_c_t = k_c;
               k_c_x = k_c;
               pad = 0;
            }

            pack_ob_A_sp(&A[A_offset + core*k_c_x], &A_p[A_p_offset + core*k_c_x*m_c_t], 
               &nnz_outer[(A_p_offset + core*k_c_x*m_c_t) / m_r], 
               &k_inds[(A_p_offset + core*k_c_x*m_c_t) / m_r], 
               &loc_m[A_p_offset + core*k_c_x*m_c_t], 
               M, K, m*m_c, 0, m_c_t, k_c_t, m_r, pad);

            // pack_ob_A_single_buf(&A[A_offset + core*k_c_x], &A_p[A_p_offset + core*k_c_x*m_c_t], 
               // M, K, m*m_c, 0, m_c_t, k_c_t, m_r, pad);
         }


         A_p_offset += k_cb*m_c_t;
      }
   }

   sp_pack->mat_sp_p = A_p;
   sp_pack->loc = loc_m;
   sp_pack->nnz_outer = nnz_outer;
   sp_pack->k_inds = k_inds;

}






void pack_A_sp_n_first(float* A, float* A_p, int M, int K, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx) {
   
      // copy over block dims to local vars to avoid readibility ussiues with x->
   int m_r = cake_cntx->mr;
   int k_c = x->k_c;
   int m_c = x->m_c;
   int m_c1 = x->m_c1, k_c1 = x->k_c1;
   int k_c1_last_core = x->k_c1_last_core;
   int k_rem = x->k_rem;
   int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad;
   int Mb = x->Mb, Kb = x->Kb;


   int m, k, A_offset = 0, A_p_offset = 0;
   int k_cb, m_c_t, p_used, core;


   unsigned char* nnz_outer = (unsigned char*) calloc(((x->M_padded*K) / m_r) , sizeof(int)); // storing number of nonzeros 
                                                                          // in each outer prod col of A

   int* k_inds = (int*) calloc(((x->M_padded*K) / m_r) , sizeof(int)); // storing kc_ind 
                                                                          // of each outer prod col of A

   unsigned char* loc_m = (unsigned char*) calloc(x->M_padded*K , sizeof(int)); // array for storing M dim C writeback location for each nnz in A
                                    // each value ranges from 0 to mr-1


   for(m = 0; m < Mb; m++) {
      
      m_c_t = p*m_c; 
      if((m == Mb - 1) && m_pad) {
         m_c_t = m_c1;
      }

      for(k = 0; k < Kb; k++) {

         if((k == Kb - 1) && k_pad) {
            p_used = p_l;
            k_cb = k_rem; 
         } else {
            p_used = p;
            k_cb = p_used*k_c;
         }

         A_offset = m*p*m_c*K + k*p*k_c;

         #pragma omp parallel for private(core)
         for(core = 0; core < p_used; core++) {

            int k_c_t, k_c_x;
            bool pad;

            if((k == Kb - 1) && k_pad) {
               k_c_t = (core == (p_l - 1) ? k_c1_last_core : k_c1);
               k_c_x = k_c1;
               pad = (core == (p_l - 1) ? 1 : 0);
            } else {
               k_c_t = k_c;
               k_c_x = k_c;
               pad = 0;
            }

            pack_ob_A_sp(&A[A_offset + core*k_c_x], &A_p[A_p_offset + core*k_c_x*m_c_t], 
               &nnz_outer[(A_p_offset + core*k_c_x*m_c_t) / m_r], 
               &k_inds[(A_p_offset + core*k_c_x*m_c_t) / m_r], 
               &loc_m[A_p_offset + core*k_c_x*m_c_t], 
               M, K, m*p*m_c, 0, m_c_t, k_c_t, m_r, pad);
            // pack_ob_A_single_buf(&A[A_offset + core*k_c_x], &A_p[A_p_offset + core*k_c_x*m_c_t], 
            //    M, K, m*p*m_c, 0, m_c_t, k_c_t, m_r, pad);
         }


         A_p_offset += k_cb*m_c_t;
      }
   }

   sp_pack->mat_sp_p = A_p;
   sp_pack->loc = loc_m;
   sp_pack->nnz_outer = nnz_outer;
   sp_pack->k_inds = k_inds;
}




