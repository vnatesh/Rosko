#pragma once

#include "cake.h"
#include "rosko_kernels.h"

// #define DEBUG 0
// #define ARR_PRINT 0
// #define CHECK_PRINT 0




// sparse matrix handling

typedef struct sp_pack_t {
   char* loc_m; // nnz vals (8-bit) // M dim C writeback location for each nnz value in A (3-5 bits)
   char* nnz_outer; // at most nnz vals (8-bit) // number of nnz in every outer prod col vec (with len m_r) of A;
   int* k_inds; // at most nnz vals // density-based reorder indices of A cols within a mrxkcxnr tile
   float* A_sp_p; // nnz vals (32-bit) // sparse packed A (only storing nonzeros)
   int* nnz_tiles; // (M*K)/(mr*kc) vals // cum-sum number of nnz vals in each mr x kc tile of A
   int* num_col_tile; // (M*K)/(mr*kc) vals // cum-sum number of outer-product cols in each mr x kc tile that have 1 or more nnz vals 
   int nnz; // total number of nonzero elements
   int nnz_cols; // number of outer product column (m_r long) that have 1 or more nonzeros
   int ntiles; // total number of mr x kc tiles in sparse matrix
   int M;
   int K;
   int mr;
   int nr;
} sp_pack_t;



// CSR data structure
typedef struct csr_t {
  float* vals; // nnz vals (32-bit)
  int* rowptr; // M+1 vals (32-bit)
  int* colind; // nnz vals (32-bit)
  int M;
  int K;
} csr_t;



