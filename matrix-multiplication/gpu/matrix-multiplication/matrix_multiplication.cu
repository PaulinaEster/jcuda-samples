/* 
 * --------------------------------------------------------------------------------------------------------------
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 * As a special exception, you may use this file as part of a free software
 * library without restriction.  Specifically, if other files instantiate
 * templates or use macros or inline functions from this file, or you compile
 * this file and link it with other files to produce an executable, this
 * file does not by itself cause the resulting executable to be covered by
 * the GNU General Public License.  This exception does not however
 * invalidate any other reasons why the executable file might be covered by
 * the GNU General Public License.
 *
 * --------------------------------------------------------------------------------------------------------------
 * Authors: 
 *      Dalvan Griebler <dalvangriebler@gmail.com>
 *      Gabriell Araujo <hexenoften@gmail.com>
 *         
 * Copyright: GNU General Public License
 * Description: This is a simple matrix multiplication algorithm. 
 * File Name: matrix_multiplication_v1.c
 * Version: 2025/08/06
 * Compile: clear && make clean && make matrix_multiplication WORKLOAD=A DEBUG=ON TIMER=ON
 * Run: ./matrix_multiplication.A.exe
 * Workloads: A, B, C, D, E, F, G, H
 * Flags: DEBUG, TIMER
 * --------------------------------------------------------------------------------------------------------------
 * This version uses arrays with two dimensions
 * --------------------------------------------------------------------------------------------------------------
 */
#include <cuda_runtime.h> 

#include "../../include/matrix-multiplication/matrix_multiplication.h"

void linearizacao(int** matrix1, int** matrix2, int** matrix3, int* matrix1L, int* matrix2L, int* matrix3L){
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){ 
			matrix1L[i + j * N] = matrix1[i][j];
			matrix2L[i + j * N] = matrix2[i][j];
			matrix3L[i + j * N] = matrix3[i][j]; 
		}
	}	
}

void deslinearizacao(int** resultm, int* result){
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			resultm[i][j] = result[i + j * N]; 
		}
	}	
}

__global__ void matrix_multiplication(int* matrix1, int* matrix2, int* matrix3){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid <=(N * N)) {
		int j = (tid / N) % N;
		int k = tid % N;
        for(int i = 0; i < N; i++){
            matrix3[j + k * N] += matrix1[j + i * N] * matrix2[i + k * N];
        }
    }
}

int main(int argc, char const *argv[]){	
	timer_start(TIMER_TOTAL);
	
	// memory allocation of the global memory
	matrix1 = (int**)malloc(sizeof(int*) * N);
	matrix2 = (int**)malloc(sizeof(int*) * N);
	matrix3 = (int**)malloc(sizeof(int*) * N); 
	for(int i=0; i < N; i++){	    
		matrix1[i] = (int*)malloc(sizeof(int) * N);
		matrix2[i] = (int*)malloc(sizeof(int) * N);
		matrix3[i] = (int*)malloc(sizeof(int) * N);
	}

	// initial values
	initialization(matrix1, matrix2, matrix3);

	int *matrix1_d, *matrix2_d, *matrix3_d, *matrix1_l, *matrix2_l, *matrix3_l;

	matrix1_l = (int*)malloc(sizeof(int*) * N * N);
	matrix2_l = (int*)malloc(sizeof(int*) * N * N);
	matrix3_l = (int*)malloc(sizeof(int*) * N * N);

    cudaMalloc((void**) &matrix1_d, sizeof(int*) * N * N);
    cudaMalloc((void**) &matrix2_d, sizeof(int*) * N * N);
    cudaMalloc((void**) &matrix3_d, sizeof(int*) * N * N);

	if(timer_flag){timer_start(TIMER_LINEARIZATION);}
	linearizacao(matrix1, matrix2, matrix3, matrix1_l, matrix2_l, matrix3_l);
	if(timer_flag){timer_stop(TIMER_LINEARIZATION);}

	if(timer_flag){timer_start(TIMER_MEMORY_TRANSFERS);}
	cudaMemcpy(matrix1_d, matrix1_l, N * N * sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix2_d, matrix2_l, N * N * sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix3_d, matrix3_l, N * N * sizeof(int*), cudaMemcpyHostToDevice); 
	if(timer_flag){timer_stop(TIMER_MEMORY_TRANSFERS);}


	// matrix multiplication
	if(timer_flag){timer_start(TIMER_COMPUTATION);}
	matrix_multiplication<<<BLOCKS, THREADS>>>(matrix1_d, matrix2_d, matrix3_d);
	if(timer_flag){timer_stop(TIMER_COMPUTATION);}


	if(timer_flag){timer_start(TIMER_MEMORY_TRANSFERS);}
    cudaMemcpy(matrix3_l, matrix3_d, N * N * sizeof(int*), cudaMemcpyDeviceToHost); 
	if(timer_flag){timer_stop(TIMER_MEMORY_TRANSFERS);}
	
	if(timer_flag){timer_start(TIMER_DESLINEARIZATION);}
	deslinearizacao(matrix3, matrix3_l);
	if(timer_flag){timer_stop(TIMER_DESLINEARIZATION);}
	timer_stop(TIMER_TOTAL);

	// checksum routine
	verification(matrix3);

	// print results
	debug_results(matrix3);	

	// freeing memory and stuff
	release_resources(matrix1, matrix2, matrix3);
	cudaFree(matrix1_d);
	cudaFree(matrix2_d);
	cudaFree(matrix3_d);
	execution_report((char*)"Matrix Multiplication", (char*)WORKLOAD, timer_read(TIMER_TOTAL), passed_verification);
	return 0;
}
