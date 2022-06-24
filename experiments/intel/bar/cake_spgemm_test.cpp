#include "cake.h"
#include "../../common/mmio.h"



int main( int argc, char** argv ) {
     // run_tests();

    int M, K, N, p = atoi(argv[2]), write_result = atoi(argv[3]);
    struct timespec start, end;
    double diff_t;
    long seconds, nanoseconds;

    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int nz;   
    int i_tmp, j_tmp; 
    float a_tmp;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
        exit(1);
    }
    else    
    { 
        if ((f = fopen(argv[1], "r")) == NULL) 
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

    bool issymm = mm_is_symmetric(matcode) ? 1 : 0; 
    int nnz = issymm ? nz*2 : nz;

    N = M;
    double density_val = ((double) nnz) / (M*K);
    printf("nnz = %d, density = %f\n", nnz, density_val);
    printf("M = %d, K = %d, N = %d, cores = %d\n", M,K,N,p);

    float* A = (float*) malloc(M * K * sizeof( float ));
    float* B = (float*) malloc(K * N * sizeof( float ));
    float* C = (float*) calloc(M * N , sizeof( float ));


    for (int i = 0; i < nz; i++) {
        fscanf(f, "%d %d %f\n", &i_tmp, &j_tmp, &a_tmp); // row, col,
        i_tmp--;  /* adjust from 1-based to 0-based */
        j_tmp--;

        // if matrix is symmetric, set value for other triangle
        if(issymm) {
            A[j_tmp*K + i_tmp] = a_tmp;
        }

        A[i_tmp*K + j_tmp] = a_tmp;
    }

    if (f !=stdin) fclose(f);

    // initialize B
    srand(time(NULL));
    rand_init(B, K, N);

    cake_cntx_t* cake_cntx = cake_query_cntx();
    
    // clock_gettime(CLOCK_REALTIME, &start);
    
    // cake_sgemm(A, B, C, M, N, K, p, cake_cntx);

 //    clock_gettime(CLOCK_REALTIME, &end);
 //    seconds = end.tv_sec - start.tv_sec;
 //    nanoseconds = end.tv_nsec - start.tv_nsec;
 //    diff_t = seconds + nanoseconds*1e-9;
    // printf("dense sgemm time: %f \n", diff_t); 
    

    // cake_sp_sgemm(A, B, C, M, N, K, p, cake_cntx);
    int ntrials = atoi(argv[4]);
    double ans = 0;

    clock_gettime(CLOCK_REALTIME, &start);

    for(int i = 0; i < ntrials; i++) {
        ans += cake_sp_sgemm(A, B, C, M, N, K, p, cake_cntx, density_val, argv);
    }

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
    printf("sp_sgemm time: %f \n", diff_t / ntrials); 


    if(write_result) {
        char fname[50];
        snprintf(fname, sizeof(fname), "result_sp");
        FILE *fp;
        fp = fopen(fname, "a");
        fprintf(fp, "rosko,%s,%d,%f\n",argv[1],p,ans / ntrials);
        fclose(fp);
    }


    free(A);
    free(B);
    free(C);

    return 0;
}




