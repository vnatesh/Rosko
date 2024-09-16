#include "rosko.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>


int main( int argc, char** argv ) {

	int M, K, N, p, sp, nz, mr, nr, ntrials, store;
	struct timespec start, end;
	long seconds, nanoseconds;
	float density;
	struct stat buffer;
	enum sched sch = KMN;
	csr_t* csr;

	M = atoi(argv[1]);
	K = atoi(argv[2]);
	N = 10000;
	p = atoi(argv[3]);
	sp = atoi(argv[4]);
	store = atoi(argv[7]);

	printf("M = %d, K = %d, N = %d, cores = %d, sparsity = %f\n", M,K,N,p, ((float) sp) / 100.0);

	float* A = (float*) malloc(M * K * sizeof( float ));
    srand(time(NULL));
	nz = rand_sparse(A, M, K, ((float) sp) / 100.0);
	density = ((float) nz) / ((float) (((float) M) * ((float) K)));
	cake_cntx_t* cake_cntx = cake_query_cntx();
	update_mr_nr(cake_cntx, 30, 128);


	// measure MKL-CSR packing DRAM bw
	// JONAS too many arguments
	// double csr_time = mat_to_csr_file(A, M, K, argv[5], store);
	double csr_time = mat_to_csr_file(A, M, K, argv[5]);
	stat(argv[5], &buffer);
	int csr_bytes = buffer.st_size;
	csr = file_to_csr(argv[5]);
	printf("csr pack time: %f \n", csr_time); 
	printf("csr bytes: %d \n", csr_bytes); 




	// measure Rosko packing DRAM bw
	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	init_sparse_block_dims(M, N, K, p, x, cake_cntx, sch, NULL, density, 4, 0);
	size_t A_sz = cake_sgemm_packed_A_size(M, K, p, x, cake_cntx, sch) / sizeof(float);
    float* A_p = (float*) calloc(A_sz, sizeof(float));
	sp_pack_t* sp_pack = (sp_pack_t*) malloc(sizeof(sp_pack_t));

	clock_gettime(CLOCK_REALTIME, &start);
	// JONAS too many arguments
	// pack_A_sp_k_first(A, A_p, M, K, p, sp_pack, x, cake_cntx, store);
	pack_A_sp_k_first(A, A_p, M, K, p, sp_pack, x, cake_cntx);
	clock_gettime(CLOCK_REALTIME, &end);
	seconds = end.tv_sec - start.tv_sec;
	nanoseconds = end.tv_nsec - start.tv_nsec;
	double rosko_time = seconds + nanoseconds*1e-9;
	printf("rosko pack time: %f \n", rosko_time); 

	sp_pack_t* sp_pack1 = malloc_sp_pack(M, K, nz, x, cake_cntx);
	pack_A_csr_to_sp_k_first(csr, M, K, nz, p, sp_pack1, x, cake_cntx);
	sp_pack_to_file(sp_pack1, argv[6]);
	stat(argv[6], &buffer);
	int rosko_bytes = buffer.st_size;
	printf("rosko bytes: %d \n", rosko_bytes); 



	// (1 read of M*K matrix A) + (writing nonzeros and indexing arrays)
	float csr_bw = ((float) (csr_bytes + M*K*4)) / csr_time / 1e9;
	float rosko_bw = ((float) (rosko_bytes + M*K*4)) / rosko_time / 1e9;
	printf("rosko bw = %f, csr bw = %f\n", rosko_bw, csr_bw);


    char fname[50];
    snprintf(fname, sizeof(fname), "result_pack");
    FILE *fp;
    fp = fopen(fname, "a");
    fprintf(fp, "rosko bw,%d,%d,%d,%d,%d,%f\n",store,M,K,N,sp,rosko_bw);
    fprintf(fp, "mkl bw,%d,%d,%d,%d,%d,%f\n",store,M,K,N,sp,csr_bw);
    fprintf(fp, "rosko time,%d,%d,%d,%d,%d,%f\n",store,M,K,N,sp,rosko_time);
    fprintf(fp, "mkl time,%d,%d,%d,%d,%d,%f\n",store,M,K,N,sp,csr_time);

    fclose(fp);
	


	free_sp_pack(sp_pack1);
	free_csr(csr);
	free(A);
	free(A_p);
	free(sp_pack);
	free(x);
	return 0;
}




