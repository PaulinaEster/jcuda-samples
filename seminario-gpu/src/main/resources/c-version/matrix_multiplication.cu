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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#define CPU_INFO_PATH "/proc/cpuinfo"
#define NO_CPU_INFO "No info"
#define PROFILING_SLOTS 64

double start[PROFILING_SLOTS];
double elapsed[PROFILING_SLOTS];
int debug_flag;
int timer_flag;
char timer_string[2048];
char checksum_string[2048];
char cpu_name[256];


#define WORKLOAD_A
#if defined(WORKLOAD_A)
#define WORKLOAD "A"
#define WORKLOAD_CHECKSUM_VALUE_1 763840LL
#define WORKLOAD_CHECKSUM_VALUE_2 21700LL
#define WORKLOAD_CHECKSUM_VALUE_3 30256LL
#define WORKLOAD_CHECKSUM_VALUE_4 26846LL
#define N 32
#elif defined(WORKLOAD_B)
#define WORKLOAD "B"
#define WORKLOAD_CHECKSUM_VALUE_1 6308736LL 
#define WORKLOAD_CHECKSUM_VALUE_2 89460LL
#define WORKLOAD_CHECKSUM_VALUE_3 165984LL
#define WORKLOAD_CHECKSUM_VALUE_4 111006LL
#define N 64
#elif defined(WORKLOAD_C)
#define WORKLOAD "C"
#define WORKLOAD_CHECKSUM_VALUE_1 51271424LL  
#define WORKLOAD_CHECKSUM_VALUE_2 363220LL
#define WORKLOAD_CHECKSUM_VALUE_3 1016000LL 
#define WORKLOAD_CHECKSUM_VALUE_4 451358LL
#define N 128
#elif defined(WORKLOAD_D)
#define WORKLOAD "D"
#define WORKLOAD_CHECKSUM_VALUE_1 413396480LL
#define WORKLOAD_CHECKSUM_VALUE_2 1463700LL
#define WORKLOAD_CHECKSUM_VALUE_3 6865280LL
#define WORKLOAD_CHECKSUM_VALUE_4 1820190LL
#define N 256
#elif defined(WORKLOAD_E)
#define WORKLOAD "E"
#define WORKLOAD_CHECKSUM_VALUE_1 3320110080LL
#define WORKLOAD_CHECKSUM_VALUE_2 5876500LL
#define WORKLOAD_CHECKSUM_VALUE_3 49840896LL 
#define WORKLOAD_CHECKSUM_VALUE_4 7310366LL
#define N 512
#elif defined(WORKLOAD_F)
#define WORKLOAD "F"
#define WORKLOAD_CHECKSUM_VALUE_1 26612709376LL
#define WORKLOAD_CHECKSUM_VALUE_2 23549460LL
#define WORKLOAD_CHECKSUM_VALUE_3 378340864LL 
#define WORKLOAD_CHECKSUM_VALUE_4 29300766LL
#define N 1024
#elif defined(WORKLOAD_G)
#define WORKLOAD "G"
#define WORKLOAD_CHECKSUM_VALUE_1 213109141504LL   
#define WORKLOAD_CHECKSUM_VALUE_2 94284820LL
#define WORKLOAD_CHECKSUM_VALUE_3 2945059840LL   
#define WORKLOAD_CHECKSUM_VALUE_4 117321758LL
#define N 2048
#elif defined(WORKLOAD_H)
#define WORKLOAD "H"
#define WORKLOAD_CHECKSUM_VALUE_1 1705703301120LL
#define WORKLOAD_CHECKSUM_VALUE_2 377313300LL
#define WORKLOAD_CHECKSUM_VALUE_3 23233566720LL   
#define WORKLOAD_CHECKSUM_VALUE_4 469524510LL
#define N 4096
#else
#define WORKLOAD_A
#define WORKLOAD "A"
#define WORKLOAD_A_REFERENCE_VALUE_1 763840LL
#define WORKLOAD_A_REFERENCE_VALUE_2 21700LL
#define WORKLOAD_A_REFERENCE_VALUE_3 30256LL
#define WORKLOAD_A_REFERENCE_VALUE_4 26846LL
#define N 32
#endif

#define TIMER_TOTAL 0
#define TIMER_MEMORY_TRANSFERS 1
#define TIMER_LINEARIZATION 2
#define TIMER_COMPUTATION 3
#define THREADS 512
#define BLOCKS (((N*N)/THREADS) + ((N*N)%THREADS))

// global variables
int** matrix1;
int** matrix2;
int** matrix3;
int passed_verification;

void timer_write_time(double* t){
	static int sec = -1;
	struct timeval tv;
	gettimeofday(&tv, 0);
	if (sec < 0) sec = tv.tv_sec;
	*t = (tv.tv_sec - sec) + 1.0e-6*tv.tv_usec;
}

double timer_elapsed_time(){
	double t;
	timer_write_time(&t);
	return(t);
}

void timer_clear(int n){
	elapsed[n] = 0.0;
}

void timer_start(int n){
	start[n] = timer_elapsed_time();
}

void timer_stop(int n){
	double t, now;
	now = timer_elapsed_time();
	t = now - start[n];
	elapsed[n] += t;
}

double timer_read(int n){
	return(elapsed[n]);
}

void activate_debug_flag(){
	/* the debug flag is disabled by default */
	debug_flag = 0;

	/* activating the debug flag through a macro */
#if defined(DEBUG)
	debug_flag = 1;	
#endif
}

void activate_timer_flag(){
	/* the timer flag is disabled by default */
	timer_flag = 0;

	/* activating the timer flag through a macro */
#if defined(TIMER)
	timer_flag = 1;	
#endif
}

void get_cpu_model(){
	FILE* file = fopen((char*)CPU_INFO_PATH, "r");
	char* error = (char*)NO_CPU_INFO;
	char* line = NULL;
	char* cpu_model;
	size_t n = 0;	
	if(file == NULL){
		strcpy(cpu_name, error);
		return;
	}		
	while(getline(&line, &n, file) > 0){
		if(strstr(line, "model name")){
			cpu_model=line;
			while(*cpu_model != ':'){
				cpu_model++;
			} cpu_model++;
			while(*cpu_model == ' '){
				cpu_model++;
			}
			strtok(cpu_model, "\n");
			fclose(file);
			strcpy(cpu_name, cpu_model);
			return;
		}
	}
	fclose(file);
	strcpy(cpu_name, error);
}

void execution_report(char* application_name, char* workload, double execution_time, int passed_verification){
	printf("----------------------------------------------------------------------------\n");
	printf(" %s:\n", application_name);
	printf("\n");
	printf(" Workload                  =     %s\n", workload);	
	printf(" Execution time in seconds =     %f\n", execution_time);
	if(passed_verification == 1){
		printf(" Correctness verification  =     SUCCESSFUL\n");
	}
	else{
		printf(" Correctness verification  =     UNSUCCESSFUL\n");
	}
	printf("----------------------------------------------------------------------------\n");
	printf(" Hardware:\n");
	printf("\n");
	printf(" CPU                       =     %s\n", cpu_name);
	printf("----------------------------------------------------------------------------\n");
	printf(" Flags:\n");
	printf("\n");
	if(debug_flag){printf(" Debug flag enabled\n");}else{printf(" Debug flag disabled\n");}
	if(timer_flag){printf(" Timer flag enabled\n");}else{printf(" Timer flag disabled\n");}
	printf("----------------------------------------------------------------------------\n");	
	printf(" Correctness:\n");
	printf("\n");
	printf("%s\n", checksum_string);
	printf("----------------------------------------------------------------------------\n");
	if(timer_flag){
		printf(" Timers:\n");
		printf("\n");
		printf("%s\n", timer_string);
		printf("----------------------------------------------------------------------------\n");
	}
}

void setup_common(){
	activate_debug_flag();
	activate_timer_flag();

	/* clear timers */
	for(int i=0; i<PROFILING_SLOTS; i++){
		timer_clear(i);
	}	

	get_cpu_model();
}

void initialization(int** matrix1, int** matrix2, int** matrix3){
	// setup common stuff
	setup_common();	

	// initial values of the global arrays
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			matrix1[i][j] = 4;
			matrix2[i][j] = 5;
			matrix3[i][j] = 0;

			if(i == j){
				matrix1[i][j] = i;
				matrix2[i][j] = j;
			}
		}
	}	
}

void verification(int** matrix){
	long long int value1 = 0;
	long long int value2 = 0;
	long long int value3 = 0;
	long long int value4 = 0;

	int i = 0;
	int j = 0;

	// total
	i = 0;
	j = 0;
	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			value1 += (long long int) matrix[i][j];
		}
	}

	// column 0
	i = 0;
	j = 0;
	for(i=0; i<N; i++){
		value2 += (long long int) matrix[i][0];
	}	

	// diagonal
	i = 0;
	j = 0;
	for(i=0; i<N; i++){
		value3 += (long long int) matrix[i][i];
	}

	// line N-1
	i = 0;
	j = 0;
	for(j=0; j<N; j++){
		value4 += (long long int) matrix[N-1][j];
	}

	if( (WORKLOAD_CHECKSUM_VALUE_1==value1) &&
			(WORKLOAD_CHECKSUM_VALUE_2==value2) &&
			(WORKLOAD_CHECKSUM_VALUE_3==value3) &&
			(WORKLOAD_CHECKSUM_VALUE_4==value4) ){
		passed_verification = 1;
	}
	else{
		passed_verification = 0;
	}

	char checksum_string_aux[256];	
	sprintf(checksum_string_aux, "%25s\t%20s\t%20s\n", "Reference", "Correct", "Found");
	strcpy(checksum_string, checksum_string_aux);
	sprintf(checksum_string_aux, "%25s\t%20lld\t%20lld\n", "checksum_1", WORKLOAD_CHECKSUM_VALUE_1, value1);
	strcat(checksum_string, checksum_string_aux);
	sprintf(checksum_string_aux, "%25s\t%20lld\t%20lld\n", "checksum_2", WORKLOAD_CHECKSUM_VALUE_2, value2);
	strcat(checksum_string, checksum_string_aux);
	sprintf(checksum_string_aux, "%25s\t%20lld\t%20lld\n", "checksum_3", WORKLOAD_CHECKSUM_VALUE_3, value3);
	strcat(checksum_string, checksum_string_aux);
	sprintf(checksum_string_aux, "%25s\t%20lld\t%20lld", "checksum_4", WORKLOAD_CHECKSUM_VALUE_4, value4);
	strcat(checksum_string, checksum_string_aux);

	char timer_string_aux[256];	
	sprintf(timer_string_aux, "%25s\t%20s\t%20s\n", "Timer", "Time (s)", "Percentage");
	strcpy(timer_string, timer_string_aux);
	sprintf(timer_string_aux, "%25s\t%20f\t%19.2f%%\n", "memory_transfers", timer_read(TIMER_MEMORY_TRANSFERS), (timer_read(TIMER_MEMORY_TRANSFERS)*100/timer_read(TIMER_TOTAL)));
	strcat(timer_string, timer_string_aux);
	sprintf(timer_string_aux, "%25s\t%20f\t%19.2f%%\n", "linearization", timer_read(TIMER_LINEARIZATION), (timer_read(TIMER_LINEARIZATION)*100/timer_read(TIMER_TOTAL)));
	strcat(timer_string, timer_string_aux);
	sprintf(timer_string_aux, "%25s\t%20f\t%19.2f%%", "matrix_multiplication", timer_read(TIMER_COMPUTATION), (timer_read(TIMER_COMPUTATION)*100/timer_read(TIMER_TOTAL)));
	strcat(timer_string, timer_string_aux);
}

void debug_results(int** matrix){
	if(debug_flag){
		FILE* file;
		file = fopen("matrix_multiplication.debug.dat", "w");
		fprintf(file, "%s\n\n", checksum_string);
		for(int i=0; i<N; i++){				
			for(int j=0; j<N; j++){
				fprintf(file, "%d ", matrix[i][j]);
			}
			fprintf(file, "\n");
		}
		fclose(file);	
	}			
}

void release_resources(int** matrix1, int** matrix2, int** matrix3){
	for (int i=0; i<N; i++){
		free(matrix1[i]);
		free(matrix2[i]);
		free(matrix3[i]);
	}
	free(matrix1);
	free(matrix2);
	free(matrix3);
}

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
		int j = (tid / N) % N; // altura largura
		int k = tid % N;
        for(int i = 0; i < N; i++){
            matrix3[j + k * N] += matrix1[j + i * N] * matrix2[i + k * N];
        }
    }
}




int main(int argc, char const *argv[]){
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

	linearizacao(matrix1, matrix2, matrix3, matrix1_l, matrix2_l, matrix3_l);

	cudaMemcpy(matrix1_d, matrix1_l, N * N * sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix2_d, matrix2_l, N * N * sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix3_d, matrix3_l, N * N * sizeof(int*), cudaMemcpyHostToDevice); 

	timer_start(TIMER_TOTAL);

	// matrix multiplication
	if(timer_flag){timer_start(TIMER_COMPUTATION);}
	matrix_multiplication<<<BLOCKS, THREADS>>>(matrix1_d, matrix2_d, matrix3_d);
	if(timer_flag){timer_stop(TIMER_COMPUTATION);}

	timer_stop(TIMER_TOTAL);

    cudaMemcpy(matrix3_l, matrix3_d, N * N * sizeof(int*), cudaMemcpyDeviceToHost); 
	
	deslinearizacao(matrix3, matrix3_l);
	// checksum routine
	verification(matrix3);

	// print results
	debug_results(matrix3);	

	// freeing memory and stuff
	release_resources(matrix1, matrix2, matrix3);

	execution_report((char*)"Matrix Multiplication", (char*)WORKLOAD, timer_read(TIMER_TOTAL), passed_verification);

	return 0;
}
