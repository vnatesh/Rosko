#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> 
#include <time.h> 
#include "mkl.h"

// Compile MKL test file using the intel advisor below:
// https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html

// gcc -fopenmp -m64 -I${MKLROOT}/include mkl_sgemm_test.c 
// -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core
//  -lmkl_gnu_thread -lpthread -lm -ldl -o mkl_sgemm_test

void rand_init(float* mat, int r, int c);


int main(int argc, char* argv[])  {
    // struct timeval start, end;
    struct timespec start, end;
    double diff_t;
    // printf("max threads %d\n\n", mkl_get_max_threads());
    if(argc < 2) {
        printf("Enter number of threads and dim size\n");
        exit(1);
    }

     //int p = atoi(argv[1]);
    // mkl_set_num_threads(atoi(argv[1]));
    int p = atoi(argv[4]);
    mkl_set_num_threads(p);

    float *A, *B, *C;
    int m, n, k, i, j, write_result, ntrials;
    float alpha, beta;

    m = atoi(argv[1]), k = atoi(argv[2]), n = atoi(argv[3]);
    write_result = atoi(argv[5]);
    ntrials = atoi(argv[6]);

    //m = atoi(argv[2]);
    //k = m;
    //n = m;

    // m = 3000, k = 3000, n = 3000;  
    //  m = 25921, k = 25921, n = 25921;      
    // m = 23040, k = 23040, n = 23040;
    alpha = 1.0; beta = 0.0;

    A = (float *)mkl_malloc( m*k*sizeof( float ), 64 );
    B = (float *)mkl_malloc( k*n*sizeof( float ), 64 );
    C = (float *)mkl_malloc( m*n*sizeof( float ), 64 );

    printf("M = %d, K = %d, N = %d\n", m, k, n);

    if (A == NULL || B == NULL || C == NULL) {
      printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
      return 1;
    }

    // gettimeofday (&start, NULL);
    srand(time(NULL));
    rand_init(A, m, k);
    rand_init(B, k, n);


    // for (i = 0; i < (m*k); i++) {
    //     // A[i] = (double)(i+1);
    //     A[i] = (double)(i);
    // }

    // for (i = 0; i < (k*n); i++) {
    //     // B[i] = (double)(-i-1);
    //     B[i] = (double)(i);
    // }

    // for (i = 0; i < (m*n); i++) {
    //     C[i] = 0.0;
    // }

    // gettimeofday (&end, NULL);
    // diff_t = (((end.tv_sec - start.tv_sec)*1000000L
    // +end.tv_usec) - start.tv_usec) / (1000000.0);
    // printf("init time: %f \n", diff_t); 



    // gettimeofday (&start, NULL);
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //             m, n, k, alpha, A, k, B, n, beta, C, n);
    // gettimeofday (&end, NULL);
    // diff_t = (((end.tv_sec - start.tv_sec)*1000000L
    // +end.tv_usec) - start.tv_usec) / (1000000.0);
    // printf("GEMM time: %f \n", diff_t); 

    clock_gettime(CLOCK_REALTIME, &start);

    for(int i = 0; i < ntrials; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C, n);
    }

    clock_gettime(CLOCK_REALTIME, &end);

    long seconds = end.tv_sec - start.tv_sec;
    long nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
    printf("GEMM time: %f \n", diff_t / ntrials); 

    if(write_result) {
        char fname[50];
        snprintf(fname, sizeof(fname), "result_ablate_intel");
        FILE *fp;
        fp = fopen(fname, "a");
        fprintf(fp, "mkl,%d,%d,%d,%d,%f\n",m,k,n,1,diff_t / ntrials);
        fclose(fp);
    }

    // printf ("\n Deallocating memory \n\n");
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    // printf (" Example completed. \n\n");
    return 0;
}



void rand_init(float* mat, int r, int c) {
    // int MAX = 65536;
    for(int i = 0; i < r*c; i++) {
        // mat[i] = (double) i;
        // mat[i] = 1.0;
        // mat[i] =  (double) (i%MAX);
        mat[i] =  (float) rand() / RAND_MAX*2.0 - 1.0;
    }   
}

