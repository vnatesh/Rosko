/*******************************************************************************
* Copyright 2020 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
*
*  Content:
*       This example demonstrates use of oneAPI Math Kernel Library (oneMKL)
*       DPCPP API oneapi::mkl::sparse::gemm to perform general sparse matrix-matrix
*       multiplication on a SYCL device (Host, CPU, GPU).
*
*       c = alpha * op(A) * b + beta * c
*
*       where op() is defined by one of
*oneapi::mkl::transpose::{nontrans,trans,conjtrans}
*
*
*       The supported floating point data types for gemm matrix data are:
*           float
*           double
*
*
*******************************************************************************/

// stl includes
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <vector>

#include "mkl.h"
#include "oneapi/mkl.hpp"
#include <CL/sycl.hpp>

// local includes
#include "../common/common_for_examples.hpp"
#include "common_for_sparse_examples.hpp"

#include <sys/time.h> 
#include <time.h> 

// #include "mmio.h" 








#define MM_MAX_LINE_LENGTH 1025
#define MatrixMarketBanner "%%MatrixMarket"
#define MM_MAX_TOKEN_LENGTH 64

typedef char MM_typecode[4];

char *mm_typecode_to_str(MM_typecode matcode);

int mm_read_banner(FILE *f, MM_typecode *matcode);
int mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz);
int mm_read_mtx_array_size(FILE *f, int *M, int *N);

int mm_write_banner(FILE *f, MM_typecode matcode);
int mm_write_mtx_crd_size(FILE *f, int M, int N, int nz);
int mm_write_mtx_array_size(FILE *f, int M, int N);


/********************* MM_typecode query fucntions ***************************/

#define mm_is_matrix(typecode)  ((typecode)[0]=='M')

#define mm_is_sparse(typecode)  ((typecode)[1]=='C')
#define mm_is_coordinate(typecode)((typecode)[1]=='C')
#define mm_is_dense(typecode)   ((typecode)[1]=='A')
#define mm_is_array(typecode)   ((typecode)[1]=='A')

#define mm_is_complex(typecode) ((typecode)[2]=='C')
#define mm_is_real(typecode)        ((typecode)[2]=='R')
#define mm_is_pattern(typecode) ((typecode)[2]=='P')
#define mm_is_integer(typecode) ((typecode)[2]=='I')

#define mm_is_symmetric(typecode)((typecode)[3]=='S')
#define mm_is_general(typecode) ((typecode)[3]=='G')
#define mm_is_skew(typecode)    ((typecode)[3]=='K')
#define mm_is_hermitian(typecode)((typecode)[3]=='H')

int mm_is_valid(MM_typecode matcode);       /* too complex for a macro */


/********************* MM_typecode modify fucntions ***************************/

#define mm_set_matrix(typecode) ((*typecode)[0]='M')
#define mm_set_coordinate(typecode) ((*typecode)[1]='C')
#define mm_set_array(typecode)  ((*typecode)[1]='A')
#define mm_set_dense(typecode)  mm_set_array(typecode)
#define mm_set_sparse(typecode) mm_set_coordinate(typecode)

#define mm_set_complex(typecode)((*typecode)[2]='C')
#define mm_set_real(typecode)   ((*typecode)[2]='R')
#define mm_set_pattern(typecode)((*typecode)[2]='P')
#define mm_set_integer(typecode)((*typecode)[2]='I')


#define mm_set_symmetric(typecode)((*typecode)[3]='S')
#define mm_set_general(typecode)((*typecode)[3]='G')
#define mm_set_skew(typecode)   ((*typecode)[3]='K')
#define mm_set_hermitian(typecode)((*typecode)[3]='H')

#define mm_clear_typecode(typecode) ((*typecode)[0]=(*typecode)[1]= \
                                    (*typecode)[2]=' ',(*typecode)[3]='G')

#define mm_initialize_typecode(typecode) mm_clear_typecode(typecode)


/********************* Matrix Market error codes ***************************/


#define MM_COULD_NOT_READ_FILE  11
#define MM_PREMATURE_EOF        12
#define MM_NOT_MTX              13
#define MM_NO_HEADER            14
#define MM_UNSUPPORTED_TYPE     15
#define MM_LINE_TOO_LONG        16
#define MM_COULD_NOT_WRITE_FILE 17


/******************** Matrix Market internal definitions ********************

   MM_matrix_typecode: 4-character sequence

                    ojbect      sparse/     data        storage 
                                dense       type        scheme

   string position:  [0]        [1]         [2]         [3]

   Matrix typecode:  M(atrix)  C(oord)      R(eal)      G(eneral)
                                A(array)    C(omplex)   H(ermitian)
                                            P(attern)   S(ymmetric)
                                            I(nteger)   K(kew)

 ***********************************************************************/

#define MM_MTX_STR      "matrix"
#define MM_ARRAY_STR    "array"
#define MM_DENSE_STR    "array"
#define MM_COORDINATE_STR "coordinate" 
#define MM_SPARSE_STR   "coordinate"
#define MM_COMPLEX_STR  "complex"
#define MM_REAL_STR     "real"
#define MM_INT_STR      "integer"
#define MM_GENERAL_STR  "general"
#define MM_SYMM_STR     "symmetric"
#define MM_HERM_STR     "hermitian"
#define MM_SKEW_STR     "skew-symmetric"
#define MM_PATTERN_STR  "pattern"


/*  high level routines */

int mm_write_mtx_crd(char fname[], int M, int N, int nz, int I[], int J[],
         double val[], MM_typecode matcode);
int mm_read_mtx_crd_data(FILE *f, int M, int N, int nz, int I[], int J[],
        double val[], MM_typecode matcode);
int mm_read_mtx_crd_entry(FILE *f, int *I, int *J, double *real, double *img,
            MM_typecode matcode);

int mm_read_unsymmetric_sparse(const char *fname, int *M_, int *N_, int *nz_,
                double **val_, int **I_, int **J_);












/* 
*   Matrix Market I/O library for ANSI C
*
*   See http://math.nist.gov/MatrixMarket for details.
*
*
*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#include "mmio.h"

int mm_read_unsymmetric_sparse(const char *fname, int *M_, int *N_, int *nz_,
                double **val_, int **I_, int **J_)
{
    FILE *f;
    MM_typecode matcode;
    int M, N, nz;
    int i;
    double *val;
    int *I, *J;
 
    if ((f = fopen(fname, "r")) == NULL)
            return -1;
 
 
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("mm_read_unsymetric: Could not process Matrix Market banner ");
        printf(" in file [%s]\n", fname);
        return -1;
    }
 
 
 
    if ( !(mm_is_real(matcode) && mm_is_matrix(matcode) &&
            mm_is_sparse(matcode)))
    {
        fprintf(stderr, "Sorry, this application does not support ");
        fprintf(stderr, "Market Market type: [%s]\n",
                mm_typecode_to_str(matcode));
        return -1;
    }
 
    /* find out size of sparse matrix: M, N, nz .... */
 
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) !=0)
    {
        fprintf(stderr, "read_unsymmetric_sparse(): could not parse matrix size.\n");
        return -1;
    }
 
    *M_ = M;
    *N_ = N;
    *nz_ = nz;
 
    /* reseve memory for matrices */
 
    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));
 
    *val_ = val;
    *I_ = I;
    *J_ = J;
 
    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
 
    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }
    fclose(f);
 
    return 0;
}

int mm_is_valid(MM_typecode matcode)
{
    if (!mm_is_matrix(matcode)) return 0;
    if (mm_is_dense(matcode) && mm_is_pattern(matcode)) return 0;
    if (mm_is_real(matcode) && mm_is_hermitian(matcode)) return 0;
    if (mm_is_pattern(matcode) && (mm_is_hermitian(matcode) || 
                mm_is_skew(matcode))) return 0;
    return 1;
}

int mm_read_banner(FILE *f, MM_typecode *matcode)
{
    char line[MM_MAX_LINE_LENGTH];
    char banner[MM_MAX_TOKEN_LENGTH];
    char mtx[MM_MAX_TOKEN_LENGTH]; 
    char crd[MM_MAX_TOKEN_LENGTH];
    char data_type[MM_MAX_TOKEN_LENGTH];
    char storage_scheme[MM_MAX_TOKEN_LENGTH];
    char *p;


    mm_clear_typecode(matcode);  

    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL) 
        return MM_PREMATURE_EOF;

    if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type, 
        storage_scheme) != 5)
        return MM_PREMATURE_EOF;

    for (p=mtx; *p!='\0'; *p=tolower(*p),p++);  /* convert to lower case */
    for (p=crd; *p!='\0'; *p=tolower(*p),p++);  
    for (p=data_type; *p!='\0'; *p=tolower(*p),p++);
    for (p=storage_scheme; *p!='\0'; *p=tolower(*p),p++);

    /* check for banner */
    if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
        return MM_NO_HEADER;

    /* first field should be "mtx" */
    if (strcmp(mtx, MM_MTX_STR) != 0)
        return  MM_UNSUPPORTED_TYPE;
    mm_set_matrix(matcode);


    /* second field describes whether this is a sparse matrix (in coordinate
            storgae) or a dense array */


    if (strcmp(crd, MM_SPARSE_STR) == 0)
        mm_set_sparse(matcode);
    else
    if (strcmp(crd, MM_DENSE_STR) == 0)
            mm_set_dense(matcode);
    else
        return MM_UNSUPPORTED_TYPE;
    

    /* third field */

    if (strcmp(data_type, MM_REAL_STR) == 0)
        mm_set_real(matcode);
    else
    if (strcmp(data_type, MM_COMPLEX_STR) == 0)
        mm_set_complex(matcode);
    else
    if (strcmp(data_type, MM_PATTERN_STR) == 0)
        mm_set_pattern(matcode);
    else
    if (strcmp(data_type, MM_INT_STR) == 0)
        mm_set_integer(matcode);
    else
        return MM_UNSUPPORTED_TYPE;
    

    /* fourth field */

    if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
        mm_set_general(matcode);
    else
    if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
        mm_set_symmetric(matcode);
    else
    if (strcmp(storage_scheme, MM_HERM_STR) == 0)
        mm_set_hermitian(matcode);
    else
    if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
        mm_set_skew(matcode);
    else
        return MM_UNSUPPORTED_TYPE;
        

    return 0;
}

int mm_write_mtx_crd_size(FILE *f, int M, int N, int nz)
{
    if (fprintf(f, "%d %d %d\n", M, N, nz) != 3)
        return MM_COULD_NOT_WRITE_FILE;
    else 
        return 0;
}

int mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz )
{
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;

    /* set return null parameter values, in case we exit with errors */
    *M = *N = *nz = 0;

    /* now continue scanning until you reach the end-of-comments */
    do 
    {
        if (fgets(line,MM_MAX_LINE_LENGTH,f) == NULL) 
            return MM_PREMATURE_EOF;
    }while (line[0] == '%');

    /* line[] is either blank or has M,N, nz */
    if (sscanf(line, "%d %d %d", M, N, nz) == 3)
        return 0;
        
    else
    do
    { 
        num_items_read = fscanf(f, "%d %d %d", M, N, nz); 
        if (num_items_read == EOF) return MM_PREMATURE_EOF;
    }
    while (num_items_read != 3);

    return 0;
}


int mm_read_mtx_array_size(FILE *f, int *M, int *N)
{
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;
    /* set return null parameter values, in case we exit with errors */
    *M = *N = 0;
    
    /* now continue scanning until you reach the end-of-comments */
    do 
    {
        if (fgets(line,MM_MAX_LINE_LENGTH,f) == NULL) 
            return MM_PREMATURE_EOF;
    }while (line[0] == '%');

    /* line[] is either blank or has M,N, nz */
    if (sscanf(line, "%d %d", M, N) == 2)
        return 0;
        
    else /* we have a blank line */
    do
    { 
        num_items_read = fscanf(f, "%d %d", M, N); 
        if (num_items_read == EOF) return MM_PREMATURE_EOF;
    }
    while (num_items_read != 2);

    return 0;
}

int mm_write_mtx_array_size(FILE *f, int M, int N)
{
    if (fprintf(f, "%d %d\n", M, N) != 2)
        return MM_COULD_NOT_WRITE_FILE;
    else 
        return 0;
}



/*-------------------------------------------------------------------------*/

/******************************************************************/
/* use when I[], J[], and val[]J, and val[] are already allocated */
/******************************************************************/

int mm_read_mtx_crd_data(FILE *f, int M, int N, int nz, int I[], int J[],
        double val[], MM_typecode matcode)
{
    int i;
    if (mm_is_complex(matcode))
    {
        for (i=0; i<nz; i++)
            if (fscanf(f, "%d %d %lg %lg", &I[i], &J[i], &val[2*i], &val[2*i+1])
                != 4) return MM_PREMATURE_EOF;
    }
    else if (mm_is_real(matcode))
    {
        for (i=0; i<nz; i++)
        {
            if (fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i])
                != 3) return MM_PREMATURE_EOF;

        }
    }

    else if (mm_is_pattern(matcode))
    {
        for (i=0; i<nz; i++)
            if (fscanf(f, "%d %d", &I[i], &J[i])
                != 2) return MM_PREMATURE_EOF;
    }
    else
        return MM_UNSUPPORTED_TYPE;

    return 0;
        
}

int mm_read_mtx_crd_entry(FILE *f, int *I, int *J,
        double *real, double *imag, MM_typecode matcode)
{
    if (mm_is_complex(matcode))
    {
            if (fscanf(f, "%d %d %lg %lg", I, J, real, imag)
                != 4) return MM_PREMATURE_EOF;
    }
    else if (mm_is_real(matcode))
    {
            if (fscanf(f, "%d %d %lg\n", I, J, real)
                != 3) return MM_PREMATURE_EOF;

    }

    else if (mm_is_pattern(matcode))
    {
            if (fscanf(f, "%d %d", I, J) != 2) return MM_PREMATURE_EOF;
    }
    else
        return MM_UNSUPPORTED_TYPE;

    return 0;
        
}


/************************************************************************
    mm_read_mtx_crd()  fills M, N, nz, array of values, and return
                        type code, e.g. 'MCRS'

                        if matrix is complex, values[] is of size 2*nz,
                            (nz pairs of real/imaginary values)
************************************************************************/

int mm_read_mtx_crd(char *fname, int *M, int *N, int *nz, int **I, int **J, 
        double **val, MM_typecode *matcode)
{
    int ret_code;
    FILE *f;

    if (strcmp(fname, "stdin") == 0) f=stdin;
    else
    if ((f = fopen(fname, "r")) == NULL)
        return MM_COULD_NOT_READ_FILE;


    if ((ret_code = mm_read_banner(f, matcode)) != 0)
        return ret_code;

    if (!(mm_is_valid(*matcode) && mm_is_sparse(*matcode) && 
            mm_is_matrix(*matcode)))
        return MM_UNSUPPORTED_TYPE;

    if ((ret_code = mm_read_mtx_crd_size(f, M, N, nz)) != 0)
        return ret_code;


    *I = (int *)  malloc(*nz * sizeof(int));
    *J = (int *)  malloc(*nz * sizeof(int));
    *val = NULL;

    if (mm_is_complex(*matcode))
    {
        *val = (double *) malloc(*nz * 2 * sizeof(double));
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val, 
                *matcode);
        if (ret_code != 0) return ret_code;
    }
    else if (mm_is_real(*matcode))
    {
        *val = (double *) malloc(*nz * sizeof(double));
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val, 
                *matcode);
        if (ret_code != 0) return ret_code;
    }

    else if (mm_is_pattern(*matcode))
    {
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val, 
                *matcode);
        if (ret_code != 0) return ret_code;
    }

    if (f != stdin) fclose(f);
    return 0;
}

int mm_write_banner(FILE *f, MM_typecode matcode)
{
    char *str = mm_typecode_to_str(matcode);
    int ret_code;

    ret_code = fprintf(f, "%s %s\n", MatrixMarketBanner, str);
    free(str);
    if (ret_code !=2 )
        return MM_COULD_NOT_WRITE_FILE;
    else
        return 0;
}

int mm_write_mtx_crd(char fname[], int M, int N, int nz, int I[], int J[],
        double val[], MM_typecode matcode)
{
    FILE *f;
    int i;

    if (strcmp(fname, "stdout") == 0) 
        f = stdout;
    else
    if ((f = fopen(fname, "w")) == NULL)
        return MM_COULD_NOT_WRITE_FILE;
    
    /* print banner followed by typecode */
    fprintf(f, "%s ", MatrixMarketBanner);
    fprintf(f, "%s\n", mm_typecode_to_str(matcode));

    /* print matrix sizes and nonzeros */
    fprintf(f, "%d %d %d\n", M, N, nz);

    /* print values */
    if (mm_is_pattern(matcode))
        for (i=0; i<nz; i++)
            fprintf(f, "%d %d\n", I[i], J[i]);
    else
    if (mm_is_real(matcode))
        for (i=0; i<nz; i++)
            fprintf(f, "%d %d %20.16g\n", I[i], J[i], val[i]);
    else
    if (mm_is_complex(matcode))
        for (i=0; i<nz; i++)
            fprintf(f, "%d %d %20.16g %20.16g\n", I[i], J[i], val[2*i], 
                        val[2*i+1]);
    else
    {
        if (f != stdout) fclose(f);
        return MM_UNSUPPORTED_TYPE;
    }

    if (f !=stdout) fclose(f);

    return 0;
}
  

/**
*  Create a new copy of a string s.  mm_strdup() is a common routine, but
*  not part of ANSI C, so it is included here.  Used by mm_typecode_to_str().
*
*/
char *mm_strdup(const char *s)
{
    int len = strlen(s);
    char *s2 = (char *) malloc((len+1)*sizeof(char));
    return strcpy(s2, s);
}

char  *mm_typecode_to_str(MM_typecode matcode)
{
    char buffer[MM_MAX_LINE_LENGTH];
    char *types[4];
    char *mm_strdup(const char *);
    int error =0;

    /* check for MTX type */
    if (mm_is_matrix(matcode)) 
        types[0] = MM_MTX_STR;
    else
        error=1;

    /* check for CRD or ARR matrix */
    if (mm_is_sparse(matcode))
        types[1] = MM_SPARSE_STR;
    else
    if (mm_is_dense(matcode))
        types[1] = MM_DENSE_STR;
    else
        return NULL;

    /* check for element data type */
    if (mm_is_real(matcode))
        types[2] = MM_REAL_STR;
    else
    if (mm_is_complex(matcode))
        types[2] = MM_COMPLEX_STR;
    else
    if (mm_is_pattern(matcode))
        types[2] = MM_PATTERN_STR;
    else
    if (mm_is_integer(matcode))
        types[2] = MM_INT_STR;
    else
        return NULL;


    /* check for symmetry type */
    if (mm_is_general(matcode))
        types[3] = MM_GENERAL_STR;
    else
    if (mm_is_symmetric(matcode))
        types[3] = MM_SYMM_STR;
    else 
    if (mm_is_hermitian(matcode))
        types[3] = MM_HERM_STR;
    else 
    if (mm_is_skew(matcode))
        types[3] = MM_SKEW_STR;
    else
        return NULL;

    sprintf(buffer,"%s %s %s %s", types[0], types[1], types[2], types[3]);
    return mm_strdup(buffer);

}









float rand_gen();

float rand_gen() {
   // return a uniformly distributed random value
   return ( (float)(rand()) + 1. )/( (float)(((float) RAND_MAX)) + 1. );
}


//
// Main example for Sparse Matrix-Dense Matrix Multiply consisting of
// initialization of A matrix, x and y vectors as well as
// scalars alpha and beta.  Then the product
//
// c = alpha * op(A) * b + beta * c
//
// is performed and finally the results are post processed.
//
template <typename fp, typename intType>
int run_sparse_matrix_dense_matrix_multiply_example(const cl::sycl::device &dev, int argc, char* argv[])
{

    // printf("start sparseMM\n");
    // Initialize data for Sparse Matrix-Vector Multiply
    oneapi::mkl::transpose transpose_val = oneapi::mkl::transpose::nontrans;

    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int i, *I, *J;
    double *val;

  int M, K, N, p, nz; 
  int ntrials = atoi(argv[3]), write_result = atoi(argv[4]), cnt = 0;
  struct timespec start, end;
  double diff_t, times;
  long seconds, nanoseconds;
  float sp;


  N = 2048; // fix N dimension for now (batch size = 8, seq len = 256)
  p = 10; // 10 cores on intel 

  int id = atoi(argv[2]);

  // read in sparse matrix A from google DNN benchmark
  FILE *fptr, *fp1;
  char *line = NULL;
  size_t len = 0;
  ssize_t nread;

  char fname[50];
  snprintf(fname, sizeof(fname), "rand.mtx");
  fp1 = fopen(fname, "a");

  fptr = fopen(argv[1], "r");
  if (fptr == NULL) {
     perror("fopen");
     exit(EXIT_FAILURE);
  }

  nread = getline(&line, &len, fptr);
  M = atoi(strtok(line," "));
  K = atoi(strtok(NULL, " "));
  nz = atoi(strtok(NULL, " "));

  printf("M = %d K = %d nz = %d  N = %d, cores = %d\n", M, K, nz, N, p);

  fprintf(fp1, "%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(fp1, "%d %d %d\n", M, K, nz);

  nread = getline(&line, &len, fptr);
  nread = getline(&line, &len, fptr);

  int ii = 0, jj = 0, prev = 0;
  char* tok;
  tok = strtok(line," \n");

  while (tok != NULL) {

    jj = atoi(tok);

    if(jj < prev) {
      ii++;
    }

    prev = jj;
    fprintf(fp1, "%d %d %f\n", ii, jj, rand_gen());
    tok = strtok(NULL, " \n");
  }

  free(line);
  fclose(fptr);
  fclose(fp1);













    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
        exit(1);
    }
    else    
    { 
        if ((f = fopen("rand.mtx", "r")) == NULL) 
            exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &K, &nz)) !=0)
        exit(1);

    intType nrows = M;
    intType ncols = K;
    std::int64_t columns = N;



    std::int64_t ldb     = columns;
    std::int64_t ldc     = columns;
    /* reseve memory for matrices */
    // I = (int *) malloc(nz * sizeof(int));
    // J = (int *) malloc(nz * sizeof(int));
    // val = (double *) malloc(nz * sizeof(double));
    std::vector<intType, mkl_allocator<intType, 64>> ia;
    std::vector<intType, mkl_allocator<intType, 64>> ja;
    std::vector<fp, mkl_allocator<fp, 64>> a;

    int i_tmp,j_tmp; float a_tmp;




    std::vector<float> val_unsorted(nz, 0.0);
    std::vector<int> row_idx_unsorted(nz, 0);
    std::vector<int> col_idx_unsorted(nz, 0);
    std::vector<int> row_idx_ctr(nrows, 0);
    std::vector<int> permutation(nz);


    for (int i = 0; i < nz; i++) {
        fscanf(f, "%d %d %f\n", &i_tmp, &j_tmp, &a_tmp); // row, col, val
        i_tmp--;  /* adjust from 1-based to 0-based */
        j_tmp--;
        row_idx_unsorted[i] = i_tmp;
        col_idx_unsorted[i] = j_tmp;
        val_unsorted[i] = a_tmp;

        ++row_idx_ctr[row_idx_unsorted[i]];
        permutation[i] = i;
    }

    if (f !=stdin) fclose(f);

    std::sort(permutation.begin(), permutation.end(),
        [&](int a, int b)
        {
            if (row_idx_unsorted[a] < row_idx_unsorted[b]) {
                return true;
            } else if (row_idx_unsorted[a] == row_idx_unsorted[b]) {
                return col_idx_unsorted[a] < col_idx_unsorted[b];
            }
            return false;
        }
    );





    double density_val = ((double) nz) / (M*K);
    printf("nz = %d, density = %f\n", nz, density_val);
    printf("M = %d, K = %d, N = %d\n", M,K,N);

    // if(density_val >= 0.999) {
    //     exit(1);
    // }



    ia.push_back(0); // starting index of row0.

    /// Generate the row pointer array.
    for (int i = 0; i < nrows; ++i) {
        ia.push_back(0); // dummy append
        ia[i + 1] = ia[i] + row_idx_ctr[i];
        row_idx_ctr[i] = 0;
    }

    /// Generate col index and value array.
    for (int i = 0; i < nz; ++i) {
        a.push_back(val_unsorted[permutation[i]]);
        ja.push_back(col_idx_unsorted[permutation[i]]);
    }


    // printf("ia %d ja %d a %d\n", ia.size(), ja.size(), a.size());

    // for(int i = 0; i < 100; i++) {
    //     printf("%d ", ja[i] );
    // }





    // for(int x = 0; x < 100; x++) {
    //     printf("%f ", a[x]);
    // }

    // Matrices b and c
    std::vector<fp, mkl_allocator<fp, 64>> b;
    std::vector<fp, mkl_allocator<fp, 64>> c;
    std::vector<fp, mkl_allocator<fp, 64>> d;

    intType nrows_b = ncols;
    intType nrows_c = nrows;

    // dense matrix B
    rand_matrix<std::vector<fp, mkl_allocator<fp, 64>>>(b, oneapi::mkl::transpose::nontrans,
                                                        columns, nrows_b, ldb);
    b.resize(nrows_b * ldb);
    c.resize(nrows_c * ldc);
    d.resize(nrows_c * ldc);

    // Init matrices c and d
    for (int i = 0; i < c.size(); i++) {
        c[i] = set_fp_value(fp(0.0), fp(0.0));
        d[i] = set_fp_value(fp(0.0), fp(0.0));
    }

    // Set scalar fp values
    fp alpha = set_fp_value(fp(1.0), fp(0.0));
    fp beta  = set_fp_value(fp(0.0), fp(0.0));

    // Catch asynchronous exceptions
    auto exception_handler = [](cl::sycl::exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (cl::sycl::exception const &e) {
                std::cout << "Caught asynchronous SYCL "
                             "exception during sparse::gemm:\n"
                          << e.what() << std::endl;
            }
        }
    };




    printf("start sparseMM\n");

    //
    // Execute Matrix Multiply
    //

    // create execution queue and buffers of matrix data
    cl::sycl::queue main_queue(dev, exception_handler);

    cl::sycl::buffer<intType, 1> ia_buffer(ia.data(), (nrows + 1));
    cl::sycl::buffer<intType, 1> ja_buffer(ja.data(), (ia[nrows]));
    cl::sycl::buffer<fp, 1> a_buffer(a.data(), (ia[nrows]));
    cl::sycl::buffer<fp, 1> b_buffer(b.data(), b.size());
    cl::sycl::buffer<fp, 1> c_buffer(c.data(), c.size());

    // create and initialize handle for a Sparse Matrix in CSR format
    oneapi::mkl::sparse::matrix_handle_t handle;

    try {


        float ressss;
        float tttmp[18];
        int flushsz=10000000;
        diff_t = 0;

        for(int i = 0; i < ntrials; i++) {


            oneapi::mkl::sparse::init_matrix_handle(&handle);

            oneapi::mkl::sparse::set_csr_data(handle, nrows, ncols, oneapi::mkl::index_base::zero,
                                              ia_buffer, ja_buffer, a_buffer);

        
            float *dirty = (float *)malloc(flushsz * sizeof(float));
            #pragma omp parallel for
            for (int dirt = 0; dirt < flushsz; dirt++){
                dirty[dirt] += dirt%100;
                tttmp[dirt%18] += dirty[dirt];
            }

            for(int ii =0; ii<18;ii++){
                ressss+= tttmp[ii];
            }


            clock_gettime(CLOCK_REALTIME, &start);

        // add oneapi::mkl::sparse::gemm to execution queue
            oneapi::mkl::sparse::gemm(main_queue, transpose_val, alpha, handle, b_buffer, columns, ldb,
                                  beta, c_buffer, ldc);

            clock_gettime(CLOCK_REALTIME, &end);
            seconds = end.tv_sec - start.tv_sec;
            nanoseconds = end.tv_nsec - start.tv_nsec;
            diff_t += seconds + nanoseconds*1e-9;

            oneapi::mkl::sparse::release_matrix_handle(&handle);

            free(dirty);

        }

        printf("sparse gemm time: %f \n", diff_t / ntrials); 

      if(write_result) {
          char fname1[50];
          snprintf(fname1, sizeof(fname1), "result_dlmc");
          FILE *fp2;
          fp2 = fopen(fname1, "a");
          fprintf(fp2, "mkl_sparse,%d,%d,%d,%d,%d,%f\n",M,K,N,nz,id, diff_t / ntrials);
          fclose(fp2);
      }

remove("rand.mtx") ;

    }

       
    catch (cl::sycl::exception const &e) {
        std::cout << "\t\tCaught synchronous SYCL exception:\n" << e.what() << std::endl;
        oneapi::mkl::sparse::release_matrix_handle(&handle);
        return 1;
    }
    catch (std::exception const &e) {
        std::cout << "\t\tCaught std exception:\n" << e.what() << std::endl;
        oneapi::mkl::sparse::release_matrix_handle(&handle);
        return 1;
    }



    //
    // Post Processing
    //

    std::cout << "\n\t\tsparse::gemm parameters:\n";
    std::cout << "\t\t\ttranspose_val = "
              << (transpose_val == oneapi::mkl::transpose::nontrans ?
                          "nontrans" :
                          (transpose_val == oneapi::mkl::transpose::trans ? "trans" : "conjtrans"))
              << std::endl;
    std::cout << "\t\t\tnrows = " << nrows << std::endl;
    std::cout << "\t\t\tncols = " << ncols << std::endl;
    std::cout << "\t\t\tcolumns = " << columns << std::endl;
    std::cout << "\t\t\tldb = " << ldb << ", ldc = " << ldc << std::endl;
    std::cout << "\t\t\talpha = " << alpha << ", beta = " << beta << std::endl;
    std::cout << "\t\t\tdensity = " << density_val << std::endl;

    // auto res = c_buffer.template get_access<cl::sycl::access::mode::read>();
    // for (intType row = 0; row < nrows; row++) {
    //     for (intType col = 0; col < columns; col++) {
    //         intType index = row * ldc + col;

    //         if (beta == (fp)0) {
    //             d[index] = set_fp_value(fp(0.0), fp(0.0));
    //         }

    //         fp tmp = set_fp_value(fp(0.0), fp(0.0));
    //         for (intType i = ia[row]; i < ia[row + 1]; i++) {
    //             tmp += a[i] * b[col + ja[i] * ldb];
    //         }
    //         d[index] = alpha * tmp + beta * d[index];
    //     }
    // }

    // intType avg_flps_per_val = 2 * ((ia[nrows] / nrows) + 1);

    // fp diff  = set_fp_value(fp(0.0), fp(0.0));
    // fp diff2 = set_fp_value(fp(0.0), fp(0.0));
    // for (intType i = 0; i < d.size(); i++) {
    //     if (!check_result(res[i], d[i], avg_flps_per_val, i))
    //         return 1;
    //     diff += (d[i] - res[i]) * (d[i] - res[i]);
    //     diff2 += d[i] * d[i];
    // }

    // std::cout << "\n\t\t sparse::gemm residual:\n"
    //           << "\t\t\t" << diff / diff2 << "\n\tFinished" << std::endl;

    return 0;
}

//
// Description of example setup, apis used and supported floating point type
// precisions
//
void print_example_banner()
{

    std::cout << "" << std::endl;
    std::cout << "###############################################################"
                 "#########"
              << std::endl;
    std::cout << "# Sparse Matrix-Dense Matrix Multiply Example: " << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# c = alpha * op(A) * b + beta * c" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# where A is a sparse matrix in CSR format, b and c are "
                 "dense matrices"
              << std::endl;
    std::cout << "# and alpha, beta are floating point type precision scalars." << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using apis:" << std::endl;
    std::cout << "#   sparse::gemm" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Supported floating point type precisions:" << std::endl;
    std::cout << "#   float" << std::endl;
    std::cout << "#   double" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "###############################################################"
                 "#########"
              << std::endl;
    std::cout << std::endl;
}

//
// Main entry point for example.
//
// Dispatches to appropriate device types as set at build time with flag:
// -DSYCL_DEVICES_host -- only runs host implementation
// -DSYCL_DEVICES_cpu -- only runs SYCL CPU implementation
// -DSYCL_DEVICES_gpu -- only runs SYCL GPU implementation
// -DSYCL_DEVICES_all (default) -- runs on all: host, cpu and gpu devices
//
//  For each device selected and each supported data type, MatrixMultiplyExample
//  runs is with all supported data types
//





int main(int argc, char* argv[]) {


    // std::cout <<  density << "\n";
    // exit(1);
    std::list<my_sycl_device_types> list_of_devices;
    set_list_of_devices(list_of_devices);

    int status = 0;
    for (auto it = list_of_devices.begin(); it != list_of_devices.end(); ++it) {

        cl::sycl::device my_dev;
        bool my_dev_is_found = false;
        get_sycl_device(my_dev, my_dev_is_found, *it);

        if (my_dev_is_found) {
            std::cout << "Running tests on " << sycl_device_names[*it] << ".\n";

            std::cout << "\tRunning with single precision real data type:" << std::endl;
            status = run_sparse_matrix_dense_matrix_multiply_example<float, std::int32_t>(my_dev, argc, argv);
            if (status != 0)
                return status;

            // if (my_dev.get_info<cl::sycl::info::device::double_fp_config>().size() != 0) {
            //     std::cout << "\tRunning with double precision real data type:" << std::endl;
            //     status = run_sparse_matrix_dense_matrix_multiply_example<double, std::int32_t>(
            //             my_dev);
            //     if (status != 0)
            //         return status;
            // }
        }
        else {
#ifdef FAIL_ON_MISSING_DEVICES
            std::cout << "No " << sycl_device_names[*it]
                      << " devices found; Fail on missing devices "
                         "is enabled.\n";
            return 1;
#else
            std::cout << "No " << sycl_device_names[*it] << " devices found; skipping "
                      << sycl_device_names[*it] << " tests.\n";
#endif
        }
    }

    return status;
}


