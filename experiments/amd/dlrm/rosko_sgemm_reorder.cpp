#include "cake.h"



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



// randomized sparse Normal(0,1) matrix with sparsity % of values determined by sigma (std dev)
void rand_sparse_gaussian1(float* mat, int r, int c, float mu, float sigma) {
	int nnz = 0;
	for(int i = 0; i < r*c; i++) {
		float x = normalRandom()*sigma+mu;
		if(fabs(x) <= 4.26) { // 2 sigmas i.e. 95% sparse
			mat[i] = 0;
		} else {
			mat[i] =  x;
			nnz++;
		}
	}	
	printf("nnz = %d\n", nnz);
}


int main( int argc, char** argv ) {
	 // run_tests();

    if(argc < 3) {
        printf("Enter M, K, and N\n");
        exit(1);
    }

	int M, K, N, p, write_result;
	struct timespec start, end;
	long nanoseconds, seconds;
	double diff_t;

	M = atoi(argv[1]);
	K = atoi(argv[2]);
	N = atoi(argv[3]);
	p = atoi(argv[4]);
	float sp = atof(argv[5]);
	write_result = atoi(argv[6]);

	printf("M = %d, K = %d, N = %d, cores = %d, sparsity = %f\n", M,K,N,p, ((float) sp) / 100.0);

	float* A = (float*) malloc(M * K * sizeof( float ));
	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));
	rand_sparse(A, M, K, ((float) sp) / 100.0);
	// rand_sparse_gaussian1(A, M, K, 0, 4.26);
	// rand_init(A, M, K);
	// print_array(A, M*K);
	// exit(1);
	rand_init(B, K, N);


    float* A_reord = row_reordering(A, M, K, N);



	cake_cntx_t* cake_cntx = cake_query_cntx();
	// update_mr_nr(cake_cntx, 30, 16);
	int iters = atoi(argv[7]);
	// if(M < 1792) {
	// 	iters = 20;
	// }
	double ret = 0;

	float ressss;
	float tttmp[18];
	int flushsz=100000000;

	for(int i = 0; i < iters; i++) {

        float *dirty = (float *)malloc(flushsz * sizeof(float));
        #pragma omp parallel for
        for (int dirt = 0; dirt < flushsz; dirt++){
            dirty[dirt] += dirt%100;
            tttmp[dirt%18] += dirty[dirt];
        }

        for(int ii =0; ii<18;ii++){
            ressss+= tttmp[ii];
        }


		ret += cake_sp_sgemm(A_reord, B, C, M, N, K, p, cake_cntx, ((float) sp) / 100.0, NULL);


        free(dirty);

	}

	printf("sp_sgemm time: %f \n", ret/iters); 

	if(write_result) {
	    char fname[50];
	    snprintf(fname, sizeof(fname), "results");
	    FILE *fp;
	    fp = fopen(fname, "a");
	    fprintf(fp, "reorder,%d,%d,%d,%d,%f,%f\n",M,K,N,p,sp, ret/iters);
	    fclose(fp);
	}

	// cake_sgemm_checker(A, B, C, N, M, K);
	
	free(A);
	free(B);
	free(C);
    free(A_reord);

	return 0;
}




