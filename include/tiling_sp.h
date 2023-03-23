#include "common_sp.h"


// default type size = 4 bytes for float
cache_dims_t* get_sparse_cache_dims(int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, enum sched sch, char* argv[], 
	float density = 0, float type_size = 4, int alg = 2, 
	int mcu = 0, int kcu = 0, int ncu = 0);

void init_sparse_block_dims(int M, int N, int K, int p, blk_dims_t* x, 
	cake_cntx_t* cake_cntx, enum sched sch, char* argv[], 
	float density = 0, float type_size = 4, int alg = 2,
	int mcu = 0, int kcu = 0, int ncu = 0);





