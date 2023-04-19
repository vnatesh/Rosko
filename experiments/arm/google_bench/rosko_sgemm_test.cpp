#include "rosko.h"
#include <string.h>

float rand_gen();

float rand_gen() {
   // return a uniformly distributed random value
   return ( (float)(rand()) + 1. )/( (float)(((float) RAND_MAX)) + 1. );
}


int main( int argc, char** argv ) {
	 // run_tests();

	int M, K, N, p, nz, alg, ntrials = atoi(argv[3]), write_result = atoi(argv[4]);
	struct timespec start, end;
	double diff_t;

	N = 2048; // fix N dimension for now (batch size = 8, seq len = 256)
	p = 4; // 4 cores on rasbpi 4

	int id = atoi(argv[2]), dram = atoi(argv[5]);

	// read in sparse matrix A from google DNN benchmark
	FILE *fptr;
	char *line = NULL;
	size_t len = 0;
	ssize_t nread;

	fptr = fopen(argv[1], "r");
	if (fptr == NULL) {
	   perror("fopen");
	   exit(EXIT_FAILURE);
	}

	nread = getline(&line, &len, fptr);
	M = atoi(strtok(line," "));
	K = atoi(strtok(NULL, " "));
	nz = atoi(strtok(NULL, " "));
	float density = ((float) nz) / ((float) (M*K));
	float* A = (float*) malloc(M * K * sizeof( float ));


	nread = getline(&line, &len, fptr);
	nread = getline(&line, &len, fptr);

	int i = 0, j = 0, prev = 0;
	char* tok;
	tok = strtok(line," \n");

	while (tok != NULL) {

		j = atoi(tok);

		if(j < prev) {
			i++;
		}

		prev = j;
		A[i*K + j] = rand_gen();
		tok = strtok(NULL, " \n");
	}

   	free(line);
   	fclose(fptr);


	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));
	 // rand_sparse(A, M, K, 0.80);
	//rand_sparse_gaussian(A, M, K, 0, 1);
	//rand_init(A, M, K);
	// print_array(A, M*K);
	// exit(1);
	rand_init(B, K, N);

	cake_cntx_t* cake_cntx = cake_query_cntx();
//	cake_sgemm(A, B, C, M, N, K, p, cake_cntx);

	// double ret = rosko_sgemm(A, B, C, M, N, K, p, cake_cntx);	

	if(density > 0.05) {
		update_mr_nr(cake_cntx, 20, 72);
		alg = 0;
	} else {
		update_mr_nr(cake_cntx, 20, 72);
		alg = 2;
	}

	// update_mr_nr(cake_cntx, 6, 16);
	// alg = 2;
	printf("M = %d K = %d nz = %d  N = %d, cores = %d, file = %s, alg = %d, mr = %d, nr = %d\n",
	 M, K, nz, N, p, argv[1], alg, cake_cntx->mr, cake_cntx->nr);



	if(dram) {

	    diff_t = 0.0;
	    for(int i = 0; i < ntrials; i++) {
			// diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density);
			diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, alg);
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

			// diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density);
			diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density, NULL, 0, NULL, 0, 1, 0, KMN, alg);
	        free(dirty);
	    }
	}

	printf("rosko,%d,%d,%d,%d,%d,%f\n",M,K,N,nz,id, diff_t / ntrials);

	if(write_result) {
	    char fname[50];
	    snprintf(fname, sizeof(fname), "result_dlmc");
	    FILE *fp;
	    fp = fopen(fname, "a");
	    fprintf(fp, "rosko,%d,%d,%d,%d,%d,%f\n",M,K,N,nz,id, diff_t / ntrials);
	    fclose(fp);
	}
	// cake_sgemm_checker(A, B, C, N, M, K);
	
	free(A);
	free(B);
	free(C);

	return 0;
}
