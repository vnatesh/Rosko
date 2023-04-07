// On Linux and MacOS, you can compile and run this program like so:
// export LD_LIBRARY_PATH=/home/vnatesh/SparseLNR/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

//   g++ -std=c++11 -fopenmp -pthread -O3 -DNDEBUG -DTACO -I include -Lbuild/lib taco_spmm.cpp -o spmm -ltaco
//   LD_LIBRARY_PATH=../../build/lib ./tac_spmm
#include <random>
#include "taco.h"
#include <omp.h>
#include <pthread.h>

using namespace taco;
int main(int argc, char* argv[]) {

  int M, K, N, p, nnz, ntrials, cnt = 0;
  struct timespec start, end;
  double diff_t;
  long seconds, nanoseconds;
  float sp;

  M = atoi(argv[1]);
  K = atoi(argv[2]);
  N = atoi(argv[3]);
  p = atoi(argv[4]);
  ntrials = atoi(argv[5]);
  sp = atof(argv[6]);
  nnz = (int) ((1.0 - (((float) sp) / 100.0))*M*K);

  // omp_set_num_threads(p);
  taco_set_num_threads(p);
  int ww = taco_get_num_threads();
  printf("yoooo %d\n", ww);

  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  Format csr({Dense,Sparse});
  Format   dm({Dense,Dense});           // (Row-major) dense matrix

  // Load a sparse matrix from file (stored in the Matrix Market format) and 
  // store it as a compressed sparse row matrix. Matrices correspond to order-2 
  // tensors in taco. The matrix in this example can be downloaded from:
  // https://www.cise.ufl.edu/research/sparse/MM/Boeing/pwtk.tar.gz
  
  char fname[50];
  snprintf(fname, sizeof(fname), "rand.mtx");
  FILE *fp1;
  fp1 = fopen(fname, "a");


  fprintf(fp1, "%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(fp1, "%d %d %d\n", M, K, nnz);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      int x = rand();
      if((x > ((float) RAND_MAX)*(sp / 100.0)) && (cnt != nnz)) {
        fprintf(fp1, "%d %d %f\n",i,j,unif(gen));
        cnt++;
      }
    }
  }
  // fprintf(fp1, "%d %d %f\n",i,j,0);
  // fprintf(fp1, "%d %d %f\n",i,j,0);

  fclose(fp1);




  Tensor<double> A = read("rand.mtx", csr);
  A.pack();

  printf("m = %d, k = %d\n",A.getDimension(0), A.getDimension(1) );
  // exit(1);

  // Generate a random dense matrix and store it in the dense matrix format. 
  Tensor<double> B({K, N}, dm);
  for (int i = 0; i < B.getDimension(0); ++i) {
    for (int j = 0; j < B.getDimension(1); ++j) {
      B.insert({i,j}, unif(gen));
    }
  }
  B.pack();


  printf("k = %d, n = %d\n",B.getDimension(0), B.getDimension(1) );

  // Declare the output matrix to be a dense matrix with the same dimensions as 
  // input matrix B, to be also stored as a doubly compressed sparse row matrix.
  Tensor<double> C({M, N}, dm);
  for (int i = 0; i < C.getDimension(0); ++i) {
    for (int j = 0; j < C.getDimension(1); ++j) {
      C.insert({i,j}, 0.0);
    }
  }
  C.pack();


  // Define the SpMM computation using index notation.
  // IndexVar i, j, k;
  // C(i,j) = (A(i,k) * B(k,j));
  // IndexStmt stmt = C.getAssignment().concretize();
  // stmt = stmt.parallelize(i, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);

  int CHUNK_SIZE = 8;
  int TILE_SIZE = 8;
  IndexVar i("i"), j("j"), k("k");
  C(i, k) = A(i, j) * B(j, k);
  IndexVar i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1");
  IndexVar jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  IndexStmt stmt = C.getAssignment().concretize();
  stmt = stmt.split(i, i0, i1, CHUNK_SIZE)
              .pos(j, jpos, A(i,j))
                .split(jpos, jpos0, jpos1, TILE_SIZE)
                  .reorder({i0, i1, jpos0, k, jpos1})
                    .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
                      .parallelize(k, ParallelUnit::CPUVector, OutputRaceStrategy::IgnoreRaces);



  C.compile(stmt);
  C.assemble();



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

      clock_gettime(CLOCK_REALTIME, &start);

      C.compute();
      
      clock_gettime(CLOCK_REALTIME, &end);
      seconds = end.tv_sec - start.tv_sec;
      nanoseconds = end.tv_nsec - start.tv_nsec;
      diff_t += seconds + nanoseconds*1e-9;

      free(dirty);
  }


  char fname1[50];
  snprintf(fname1, sizeof(fname1), "results");
  FILE *fp2;
  fp2 = fopen(fname1, "a");
  fprintf(fp2, "taco,%d,%d,%d,%d,%f,%f\n",M,K,N,p,sp, diff_t / ntrials);
  fclose(fp2);



  printf("done compute\n");
  remove("rand.mtx"); 

  // Write the output of the computation to file (stored in the FROSTT format).
  // write("C.tns", C);
}













// // On Linux and MacOS, you can compile and run this program like so:
// //   g++ -std=c++11 -O3 -DNDEBUG -DTACO -I ../../include -L../../build/lib spmv.cpp -o spmv -ltaco
// //   LD_LIBRARY_PATH=../../build/lib ./spmv
// #include <random>
// #include "taco.h"
// using namespace taco;
// int main(int argc, char* argv[]) {
//   std::default_random_engine gen(0);
//   std::uniform_real_distribution<double> unif(0.0, 1.0);
//   // Predeclare the storage formats that the inputs and output will be stored as.
//   // To define a format, you must specify whether each dimension is dense or sparse 
//   // and (optionally) the order in which dimensions should be stored. The formats 
//   // declared below correspond to compressed sparse row (csr) and dense vector (dv). 
//   Format csr({Dense,Sparse});
//   Format  dv({Dense});

//   // Load a sparse matrix from file (stored in the Matrix Market format) and 
//   // store it as a compressed sparse row matrix. Matrices correspond to order-2 
//   // tensors in taco. The matrix in this example can be downloaded from:
//   // https://www.cise.ufl.edu/research/sparse/MM/Boeing/pwtk.tar.gz
//   Tensor<double> A = read("pwtk.mtx", csr);

//   // Generate a random dense vector and store it in the dense vector format. 
//   // Vectors correspond to order-1 tensors in taco.
//   Tensor<double> x({A.getDimension(1)}, dv);
//   for (int i = 0; i < x.getDimension(0); ++i) {
//     x.insert({i}, unif(gen));
//   }
//   x.pack();

//   // Generate another random dense vetor and store it in the dense vector format..
//   Tensor<double> z({A.getDimension(0)}, dv);
//   for (int i = 0; i < z.getDimension(0); ++i) {
//     z.insert({i}, unif(gen));
//   }
//   z.pack();

//   // Declare and initializing the scaling factors in the SpMV computation. 
//   // Scalars correspond to order-0 tensors in taco.
//   Tensor<double> alpha(42.0);
//   Tensor<double> beta(33.0);

//   // Declare the output matrix to be a sparse matrix with the same dimensions as 
//   // input matrix B, to be also stored as a doubly compressed sparse row matrix.
//   Tensor<double> y({A.getDimension(0)}, dv);
//   // Define the SpMV computation using index notation.
//   IndexVar i, j;
//   y(i) = alpha() * (A(i,j) * x(j)) + beta() * z(i);
//   // At this point, we have defined how entries in the output vector should be 
//   // computed from entries in the input matrice and vectorsbut have not actually 
//   // performed the computation yet. To do so, we must first tell taco to generate 
//   // code that can be executed to compute the SpMV operation.
//   y.compile();
//   // We can now call the functions taco generated to assemble the indices of the 
//   // output vector and then actually compute the SpMV.
//   y.assemble();
//   y.compute();
//   // Write the output of the computation to file (stored in the FROSTT format).
//   write("y.tns", y);
// }
