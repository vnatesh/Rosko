#include "rosko.h"
#include "../../common/mmio.h"



int main( int argc, char** argv ) {
     // run_tests();

    int M, K, N, alg;
    int p = atoi(argv[2]), write_result = atoi(argv[3]);
    int ntrials = atoi(argv[4]), dram = atoi(argv[5]);
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
    double density_val = ((double) nnz) / (((double) M) * ((double) K));
    float* A = (float*) malloc(M * K * sizeof( float ));
    float* B = (float*) malloc(K * N * sizeof( float ));
    float* C = (float*) calloc(M * N , sizeof( float ));


    for (int i = 0; i < nz; i++) {
        fscanf(f, "%d %d %f\n", &i_tmp, &j_tmp, &a_tmp); // row, col, val
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
    if(density_val > 0.05) {
        update_mr_nr(cake_cntx, 30, 128);
        alg = 0;
    } else {
        update_mr_nr(cake_cntx, 6, 16);
        alg = 2;
    }


    printf("M = %d K = %d nz = %d  N = %d, cores = %d, file = %s, alg = %d, mr = %d, nr = %d, density = %f\n",
     M, K, nnz, N, p, argv[1], alg, cake_cntx->mr, cake_cntx->nr, density_val);



    if(dram) {

        diff_t = 0.0;
        for(int i = 0; i < ntrials; i++) {
            // diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density);
            diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density_val, NULL, 0, NULL, 0, 1, 0, KMN, alg);
        }

    } else {

        float ressss;
        float tttmp[18];
        int flushsz=10000000;
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

            // diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density);
            diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density_val, NULL, 0, NULL, 0, 1, 0, KMN, alg);
            free(dirty);
        }
    }

    printf("sp_sgemm time: %f \n", diff_t / ntrials); 


    if(write_result) {
        char fname[50];
        snprintf(fname, sizeof(fname), "result_sp");
        FILE *fp;
        fp = fopen(fname, "a");
        fprintf(fp, "rosko,%s,%d,%d,%d,%d,%f,%f\n",argv[1],M,K,N,p,(1-density_val)*100,diff_t / ntrials);
        fclose(fp);
    }


    free(A);
    free(B);
    free(C);

    return 0;
}




