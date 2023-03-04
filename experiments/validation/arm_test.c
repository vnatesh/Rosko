#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> 
#include <time.h> 
#include <string.h>

#include <omp.h>
#include "armpl.h"


// Compile ARMPL test 

// gcc test.c -I/opt/arm/armpl_20.3_gcc-7.1/include -L{ARMPL_DIR} -lm 

// gcc -m64 -I${MKLROOT}/include mkl_sgemm_test.c 
// -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core
//  -lmkl_gnu_thread -lpthread -lm -ldl -o mkl_sgemm_test

void rand_init(float* mat, int r, int c);


int main(int argc, char* argv[])  {
    struct timespec start, end;
    double diff_t;
    float *A, *B, *C;
    int m, n, k, i, j, p, nz, ntrials, dram;
    float alpha, beta, sp;


    // printf("max threads %d\n\n", mkl_get_max_threads());
    if(argc < 2) {
        printf("Enter number of threads\n");
        exit(1);
    }

    m = atoi(argv[1]);
    k = atoi(argv[2]);
    n = atoi(argv[3]);
    p = atoi(argv[4]);
    sp = atof(argv[5]);
    ntrials = atoi(argv[10]);
    dram = atoi(argv[11]);

    omp_set_num_threads(p);

    alpha = 1.0; beta = 0.0;

    A = (float *) malloc(m * k * sizeof(float));
    B = (float *) malloc(k * n * sizeof(float));
    C = (float *) malloc(m * n * sizeof(float));

    printf("M = %d, K = %d, N = %d\n", m, k, n);

    if (A == NULL || B == NULL || C == NULL) {
      printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      free(A);
      free(B);
      free(C);
      return 1;
    }

    // rand_init(A, m, k);
    // rand_init(B, k, n);


    if(dram) {

        for(int i = 0; i < ntrials; i++) {

            clock_gettime(CLOCK_REALTIME, &start);

            // diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, A, k, B, n, beta, C, n);

            clock_gettime(CLOCK_REALTIME, &end);
            long seconds = end.tv_sec - start.tv_sec;
            long nanoseconds = end.tv_nsec - start.tv_nsec;
            diff_t += seconds + nanoseconds*1e-9;
        }

    } else {

        float ressss;
        float tttmp[18];
        int flushsz=200000;
        diff_t = 0.0;
        
        for(int i = 0; i < ntrials; i++) {


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

            // diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, A, k, B, n, beta, C, n);

            clock_gettime(CLOCK_REALTIME, &end);
            long seconds = end.tv_sec - start.tv_sec;
            long nanoseconds = end.tv_nsec - start.tv_nsec;
            diff_t += seconds + nanoseconds*1e-9;
            free(dirty);
        }
    }


    if(!dram) {
        char fname[50];
        snprintf(fname, sizeof(fname), "results");
        FILE *fp;
        fp = fopen(fname, "a");
        fprintf(fp, "%s,armpl,%d,%d,%d,%d,%f,%f\n",argv[9],m,k,n,p,sp,diff_t / ntrials);
        fclose(fp);
    }

    free(A);
    free(B);
    free(C);

    printf (" Example completed. \n\n");
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


