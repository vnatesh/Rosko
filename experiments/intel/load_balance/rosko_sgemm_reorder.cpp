#include "rosko.h"
#include "../../common/mmio.h"

#include <map>
#include <vector>
#include <algorithm>


float* row_reordering(float* A, int M, int K, int N);



float* row_reordering(float* A, int M, int K, int N) {

    // assign rows to bins based on their nnz
    std::map<int, std::vector<int>, std::greater<int>> nnz_bins;
    float* A_reord = (float*) calloc(M * K, sizeof( float ));
    int rows_rem = M, bin_ind = 0, a_ind = 0, nnz = 0;


    for(int i = 0; i < M; i++) {

        nnz = 0;

        for(int j = 0; j < K; j++) {
            if(A[i*K + j]) {
                nnz++;
            }
        }

        nnz_bins[nnz].push_back(i);
        nnz = 0;
    }


    // round-robin assignment of rows from each nnz bin to new A mat to
    // remove clusters of high/low density rows and distribute them evenly
    // throughout the matrix for load balancing

    std::map<int, std::vector<int>, std::greater<int>>::reverse_iterator it;


    while(rows_rem) {

        for(std::pair<int, std::vector<int>> e : nnz_bins) {
            
            if(bin_ind < e.second.size()) {

                // write row 
                for(int j = 0; j < K; j++) {
                    A_reord[a_ind] = A[e.second[bin_ind]*K + j];
                    a_ind++;
                }

                rows_rem--;
            }
        }

        bin_ind++;



        for (it = nnz_bins.rbegin(); it != nnz_bins.rend(); it++) {    
            
            if(bin_ind < it->second.size()) {

                // write row 
                for(int j = 0; j < K; j++) {
                    A_reord[a_ind] = A[it->second[bin_ind]*K + j];
                    a_ind++;
                }

                rows_rem--;
            }
        }

        bin_ind++;
    }



    // int split_id;
    // int row_acc = 0;

    // for(std::pair<int, std::vector<int>> e : nnz_bins) {
        
    //     if(row_acc >= M/2) {
    //         split_id = e.first;
    //         break;
    //     }
        
    //     row_acc += e.second.size();
    // }

    // printf("split = %d\n", split_id);

    // for(std::pair<int, std::vector<int>> e : nnz_bins) {
    //     printf("nnz = %d, nrows = %d\n",e.first, e.second.size());
    // }
    
    // print_mat(A, M, K);

    // print_mat(A_reord, M, K);

    // for(std::pair<int, std::vector<int>> e : nnz_bins) {
        
    //     for(bin_ind = 0; bin_ind < e.second.size(); bin_ind++) {

    //         for(int j = 0; j < K; j++) {
    //             A_reord[a_ind] = A[e.second[bin_ind]*K + j];
    //             a_ind++;
    //         }

    //         rows_rem--;
    //     }
    // }


    return A_reord;
}


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
    
    // clock_gettime(CLOCK_REALTIME, &start);
    
    // cake_sgemm(A, B, C, M, N, K, p, cake_cntx);

 //    clock_gettime(CLOCK_REALTIME, &end);
 //    seconds = end.tv_sec - start.tv_sec;
 //    nanoseconds = end.tv_nsec - start.tv_nsec;
 //    diff_t = seconds + nanoseconds*1e-9;
    // printf("dense sgemm time: %f \n", diff_t); 
    

    // rosko_sgemm(A, B, C, M, N, K, p, cake_cntx);
    int ntrials = atoi(argv[4]);
    double ans = 0;


    float* A_reord = row_reordering(A, M, K, N);

    clock_gettime(CLOCK_REALTIME, &start);

    for(int i = 0; i < ntrials; i++) {
        ans += rosko_sgemm(A_reord, B, C, M, N, K, p, cake_cntx, density_val);
    }

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
    printf("row-reorder sp_sgemm time: %f \n", diff_t / ntrials); 

    if(write_result) {
        char fname[50];
        snprintf(fname, sizeof(fname), "load_balance");
        FILE *fp;
        fp = fopen(fname, "a");
        fprintf(fp, "rosko row reorder,%s,%d,%d,%d,%d,%f,%f\n",argv[1],M,K,N,p,(1-density_val)*100, ans / ntrials);
        fclose(fp);
    }


    double ans1 = 0;

    clock_gettime(CLOCK_REALTIME, &start);

    for(int i = 0; i < ntrials; i++) {
        ans1 += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density_val);
    }

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
    printf("sp_sgemm time: %f \n", diff_t / ntrials); 

    if(write_result) {
        char fname[50];
        snprintf(fname, sizeof(fname), "load_balance");
        FILE *fp;
        fp = fopen(fname, "a");
        fprintf(fp, "rosko,%s,%d,%d,%d,%d,%f,%f\n",argv[1],M,K,N,p,(1-density_val)*100, ans1 / ntrials);
        fclose(fp);
    }


    free(A);
    free(B);
    free(C);
    free(A_reord);
    
    return 0;
}



