	/*
 * Copyright (c) 2018-2019 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "utils/Utils.h"
#include <omp.h>
#include <cstdlib>

using namespace arm_compute;
using namespace utils;

class NESGEMMExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        NPYLoader npy0;
        NPYLoader npy1;
        NPYLoader npy2;
        alpha = 1.0f;
        beta  = 0.0f;

		N = 2048; // fix N dimension for now (batch size = 8, seq len = 256)
		p = 4; // 4 cores on rasbpi 4

		// read in sparse matrix A from google DNN benchmark
		FILE *fptr;
		char *line = NULL;
		size_t len = 0;
		ssize_t nread;

        id = atoi(argv[2]);
        ntrials = atoi(argv[3]);
        write_result = atoi(argv[4]);
        dram = atoi(argv[5]);

		fptr = fopen(argv[1], "r");
		if (fptr == NULL) {
		   perror("fopen");
		   exit(EXIT_FAILURE);
		}

		nread = getline(&line, &len, fptr);
		M = atoi(strtok(line," "));
		K = atoi(strtok(NULL, " "));
		nz = atoi(strtok(NULL, " "));
	   	free(line);
	   	fclose(fptr);

        printf("M = %ld, K = %ld, N = %ld, p = %d\n", M,K,N,p);
        
        src0.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::F32));
        src1.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::F32));
        src2.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::F32));


        init_sgemm_output(dst, src0, src1, DataType::F32);

        // Configure function
        sgemm.configure(&src0, &src1, nullptr, &dst, alpha, beta);

        // Allocate all the images
        src0.allocator()->allocate();
        src1.allocator()->allocate();
        dst.allocator()->allocate();

        src2.allocator()->allocate();

        fill_random_tensor(src0, -1.f, 1.f);
        fill_random_tensor(src1, -1.f, 1.f);
        fill_random_tensor(src2, -1.f, 1.f);
    
        // // Dummy run for CLTuner
        // struct timespec start, end;
        // double diff_t;

        // clock_gettime(CLOCK_REALTIME, &start);

        // sgemm.run();

        // clock_gettime(CLOCK_REALTIME, &end);
        // long seconds = end.tv_sec - start.tv_sec;
        // long nanoseconds = end.tv_nsec - start.tv_nsec;
        // diff_t = seconds + nanoseconds*1e-9;
        // printf("sgemm 1 time: %f \n", diff_t); 


        return true;
    }

    void do_run() override
    {
        // Execute the function
        struct timespec start, end;
        double diff_t;


        // use p cores for experiment
        NEScheduler::get().set_num_threads(p);


        if(dram) {

            diff_t = 0.0;
            for(int i = 0; i < ntrials; i++) {

                clock_gettime(CLOCK_REALTIME, &start);

                // diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density);
                sgemm.run();

                clock_gettime(CLOCK_REALTIME, &end);
                long seconds = end.tv_sec - start.tv_sec;
                long nanoseconds = end.tv_nsec - start.tv_nsec;
                diff_t += seconds + nanoseconds*1e-9;
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

                clock_gettime(CLOCK_REALTIME, &start);

                // diff_t += rosko_sgemm(A, B, C, M, N, K, p, cake_cntx, density);
                sgemm.run();

                clock_gettime(CLOCK_REALTIME, &end);
                long seconds = end.tv_sec - start.tv_sec;
                long nanoseconds = end.tv_nsec - start.tv_nsec;
                diff_t += seconds + nanoseconds*1e-9;

                free(dirty);
            }
        }

        printf("sgemm time: %f \n", diff_t / ntrials); 

        if(write_result) {
            char fname[50];
            snprintf(fname, sizeof(fname), "result_dlmc");
            FILE *fp;
            fp = fopen(fname, "a");
            fprintf(fp, "armcl,%d,%d,%d,%d,%d,%f\n",M,K,N,nz,id,diff_t / ntrials);
            fclose(fp);
        }

    }
    void do_teardown() override
    {
        if(!output_filename.empty()) /* Save to .npy file */
        {
            save_to_npy(dst, output_filename, is_fortran);
        }
    }

private:
    Tensor      src0{}, src1{}, src2{}, dst{};
    NEGEMM      sgemm{};
    float       alpha{}, beta{};
    size_t      M;
    size_t      N;
    size_t      K;
    int         p;
    int 		nz;
    int id;
    int write_result;
    int dram;
    int ntrials;
    bool        is_fortran{};
    std::string output_filename{};
};

/** Main program for sgemm test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Matrix A, [optional] Matrix B, [optional] Matrix C, [optional] alpha, [optional] beta )
 */
int main(int argc, char **argv)
{
    return utils::run_example<NESGEMMExample>(argc, argv);
}







