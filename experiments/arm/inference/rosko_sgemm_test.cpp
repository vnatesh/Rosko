#include "rosko.h"
#include <string.h>



int main(int argc, char* argv[]) {

	int M, K, N, p, nz = 0;
	float tmp;
	struct timespec start, end;
	double diff_t;

	p = 4; // 4 cores on rasbpi 4

	FILE* fp = fopen(argv[1],"r");
	if(!fp) {
		printf("Error: Could not open file\n");
		exit(1);
	}

	char line[100];
	int i = 0;
	fgets(line, 100, fp);
	strtok(line, " ");
	N = atoi(strtok(NULL, " "));
	strtok(NULL, " ");
	K = atoi(strtok(NULL, " "));
	strtok(NULL, " ");
	M = atoi(strtok(NULL, " "));

	printf("%d %d %d \n",M,K,N );

	fgets(line, 100, fp);
	fgets(line, 100, fp);
	fgets(line, 100, fp);


	float* A = (float*) malloc(M * K * sizeof( float ));

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

	printf("M = %d K = %d N = %d nz = %d\n", M, K, N, nz);




	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize B
    srand(time(NULL));
	rand_init(B, K, N);

	cake_cntx_t* cake_cntx = cake_query_cntx();
	
	double ret1 = 0.0, ret2 = 0.0;
	int iters = 10;
	float density = ((float) nz) / ((float) (M*K));

	for(int i = 0; i < iters; i++) {
		ret1 += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density);
		ret2 += cake_sgemm(A, B, C, M, N, K, p, cake_cntx);
	}

    char fname[50];
    snprintf(fname, sizeof(fname), "result_end_to_end");
    FILE *fp1;
    fp1 = fopen(fname, "a");
    fprintf(fp1, "rosko,%d,%d,%d,%d,%f\n",M,K,N,nz,ret1 / iters);
    fprintf(fp1, "cake,%d,%d,%d,%d,%f\n",M,K,N,nz,ret2 / iters);
    fclose(fp1);

	// cake_sgemm_checker(A, B, C, N, M, K);
	
	free(A);
	free(B);
	free(C);

	return 0; 
}

