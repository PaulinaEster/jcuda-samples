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
// #include <cuda_runtime.h> 

extern "C"
__global__ void matrix_multiplication(int* matrix1, int* matrix2, int* matrix3, int N){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid <=(N * N)) {
		int j = (tid / N) % N; // altura largura
		int k = tid % N;
        for(int i = 0; i < N; i++){
            matrix3[j + k * N] += matrix1[j + i * N] * matrix2[i + k * N];
        }
    }
}
