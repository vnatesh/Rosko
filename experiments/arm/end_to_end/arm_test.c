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
    int m, n, k, j, p, nz = 0;
    float alpha, beta, tmp;



    p = 4; // 4 cores on rasbpi 4

    omp_set_num_threads(p);

    FILE* fp = fopen(argv[1],"r");
    if(!fp) {
        printf("Error: Could not open file\n");
        exit(1);
    }

    char line[100];
    int i = 0;
    fgets(line, 100, fp);
    strtok(line, " ");
    n = atoi(strtok(NULL, " "));
    strtok(NULL, " ");
    k = atoi(strtok(NULL, " "));
    strtok(NULL, " ");
    m = atoi(strtok(NULL, " "));

    printf("%d %d %d \n",m,k,n );

    fgets(line, 100, fp);
    fgets(line, 100, fp);
    fgets(line, 100, fp);


     A = (float*) malloc(m * k * sizeof( float ));

    char* pEnd;
    fgets(line, 100, fp);
    tmp = strtof(line, &pEnd);

    while(fgets(line, 100, fp)) {

        if(tmp != 0) {
            nz++;
        }

        A[i] = tmp;
        i++;

        tmp = strtof(line, NULL);
        
    }

    if(tmp != 0) {
        nz++;
    }

    A[i] = tmp;

    fclose(fp);

    printf("M = %d K = %d N = %d nz = %d\n", m,k,n, nz);




    alpha = 1.0; beta = 0.0;


    B = (float *) malloc(k * n * sizeof(float));
    C = (float *) malloc(m * n * sizeof(float));


    if (A == NULL || B == NULL || C == NULL) {
      printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      free(A);
      free(B);
      free(C);
      return 1;
    }

    rand_init(B, k, n);

    clock_gettime(CLOCK_REALTIME, &start);

    int iters = 10;

    for(int i = 0; i < iters; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C, n);
    }


    clock_gettime(CLOCK_REALTIME, &end);
    long seconds = end.tv_sec - start.tv_sec;
    long nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
    printf("sgemm time: %f \n", diff_t); 


    char fname[50];
    snprintf(fname, sizeof(fname), "result_end_to_end");
    FILE *fp1;
    fp1 = fopen(fname, "a");
    fprintf(fp1, "armpl,%d,%d,%d,%d,%f\n",m,k,n,nz, diff_t / iters);
    fclose(fp1);

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
