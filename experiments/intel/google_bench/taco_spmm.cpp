// On Linux and MacOS, you can compile and run this program like so:
// export LD_LIBRARY_PATH=/home/vnatesh/SparseLNR/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

//   g++ -std=c++11 -fopenmp -pthread -O3 -DNDEBUG -DTACO -I include -Lbuild/lib spmm.cpp -o spmm -ltaco
//   LD_LIBRARY_PATH=../../build/lib ./spmm
#include <random>
#include "taco.h"
#include <omp.h>
#include <pthread.h>

using namespace taco;
int main(int argc, char* argv[]) {

  int M, K, N, p, nz; 
  int ntrials = atoi(argv[3]), write_result = atoi(argv[4]), cnt = 0;
  struct timespec start, end;
  double diff_t;
  long seconds, nanoseconds;
  float sp;


  N = 2048; // fix N dimension for now (batch size = 8, seq len = 256)
  p = 10; // 10 cores on intel 

  int id = atoi(argv[2]);

  // omp_set_num_threads(p);
  taco_set_num_threads(p);
  int ww = taco_get_num_threads();

  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  Format csr({Dense,Sparse});
  Format   dm({Dense,Dense});           // (Row-major) dense matrix

  // Load a sparse matrix from file (stored in the Matrix Market format) and 
  // store it as a compressed sparse row matrix. Matrices correspond to order-2 
  // tensors in taco. The matrix in this example can be downloaded from:
  // https://www.cise.ufl.edu/research/sparse/MM/Boeing/pwtk.tar.gz
  

  // read in sparse matrix A from google DNN benchmark
  FILE *fptr, *fp1;
  char *line = NULL;
  size_t len = 0;
  ssize_t nread;

  char fname[50];
  snprintf(fname, sizeof(fname), "rand.mtx");
  fp1 = fopen(fname, "a");

  fptr = fopen(argv[1], "r");
  if (fptr == NULL) {
     perror("fopen");
     exit(EXIT_FAILURE);
  }

  nread = getline(&line, &len, fptr);
  M = atoi(strtok(line," "));
  K = atoi(strtok(NULL, " "));
  nz = atoi(strtok(NULL, " "));

  printf("M = %d K = %d nz = %d  N = %d, cores = %d\n", M, K, nz, N, p);

  fprintf(fp1, "%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(fp1, "%d %d %d\n", M, K, nz);

  nread = getline(&line, &len, fptr);
  nread = getline(&line, &len, fptr);

  int ii = 0, jj = 0, prev = 0;
  char* tok;
  tok = strtok(line," \n");

  while (tok != NULL) {

    jj = atoi(tok);

    if(jj < prev) {
      ii++;
    }

    prev = jj;
    fprintf(fp1, "%d %d %f\n",ii,jj,unif(gen));
    tok = strtok(NULL, " \n");
  }

  free(line);
  fclose(fptr);
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
  // see http://tensor-compiler.org/files/senanayake-meng-thesis-taco-scheduling.pdf
  
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



  if(write_result) {
      char fname1[50];
      snprintf(fname1, sizeof(fname1), "result_dlmc");
      FILE *fp2;
      fp2 = fopen(fname1, "a");
      fprintf(fp2, "taco,%d,%d,%d,%d,%d,%f\n",M,K,N,nz,id, diff_t / ntrials);
      fclose(fp2);
  }


  printf("done compute\n");
  remove("rand.mtx"); 

  // Write the output of the computation to file (stored in the FROSTT format).
  // write("C.tns", C);
}

