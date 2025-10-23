package jcuda.matrixmultiplication;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import java.io.IOException;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.utils.JCudaSamplesUtils;
import jcuda.*;
import jcuda.driver.CUdevice_attribute;

import utils.MatrixMultiplicationUtils;
import utils.Config;
import utils.WorkloadTimer;
import utils.TimerType;
public class JCudaMatrixMultiplication
{
    /**
     * Entry point of this sample
     *
     * @param args Not used
     * @throws IOException If an IO error occurs
     */
    public static void main(String args[]) throws IOException
    {
        Config.setupCommon();

        int N = Config.getWorkload().getN();
        int[][] matrix1 = new int[N][N];
        int[][] matrix2 = new int[N][N];
        int[][] matrix3 = new int[N][N];
        MatrixMultiplicationUtils.initialization(matrix1, matrix2, matrix3);
        
        int[] matrix1Linear = new int[N*N];
        int[] matrix2Linear = new int[N*N];
        int[] matrix3Linear = new int[N*N]; 

        WorkloadTimer.timerStart(TimerType.TOTAL.ordinal());
        WorkloadTimer.timerStart(TimerType.LINEARIZATION.ordinal());
        linearization(matrix1, matrix2, matrix3, matrix1Linear, matrix2Linear, matrix3Linear);
        WorkloadTimer.timerStop(TimerType.LINEARIZATION.ordinal());

        WorkloadTimer.timerStart(TimerType.JCUDADRIVER.ordinal());
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);
        // Create the PTX file by calling the NVCC
        String ptxFileName = JCudaSamplesUtils.preparePtxFile(
            "src/main/resources/kernels/JCudaMatrixMultiplicationKernel.cu");
            
        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        int[] maxThreadsPerBlock = new int[1];
        byte[] nameBytes = new byte[1024];

        JCudaDriver.cuDeviceGetAttribute(maxThreadsPerBlock,
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            device
        );

        JCudaDriver.cuDeviceGetName(nameBytes, nameBytes.length, device);
        String deviceName = new String(nameBytes).trim();
        
        System.out.println("GPU: " + deviceName);
        System.out.println("Máximo Threads: " + maxThreadsPerBlock[0]);

        // Load the ptx file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        // Obtain a function pointer to the "matrix_multiplication" function.
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "matrix_multiplication"); 
        WorkloadTimer.timerStop(TimerType.JCUDADRIVER.ordinal());
        
        CUdeviceptr deviceMatrix1 = new CUdeviceptr();
        CUdeviceptr deviceMatrix2 = new CUdeviceptr();
        CUdeviceptr deviceMatrix3 = new CUdeviceptr();

        WorkloadTimer.timerStart(TimerType.MEMORY_TRANSFERS.ordinal());
        // Alocação de dados para o device
        cuMemAlloc(deviceMatrix1, N * N * Sizeof.INT);
        cuMemAlloc(deviceMatrix2, N * N * Sizeof.INT); 
        cuMemAlloc(deviceMatrix3, N * N * Sizeof.INT);
        
        // Tranferencia de dados para a memoria do device
        cuMemcpyHtoD(deviceMatrix1, Pointer.to(matrix1Linear), N * N * Sizeof.INT);
        cuMemcpyHtoD(deviceMatrix2, Pointer.to(matrix2Linear), N * N * Sizeof.INT);
        cuMemcpyHtoD(deviceMatrix3, Pointer.to(matrix3Linear), N * N * Sizeof.INT);
        WorkloadTimer.timerStop(TimerType.MEMORY_TRANSFERS.ordinal());

        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParameters = Pointer.to(
            Pointer.to(deviceMatrix1),
            Pointer.to(deviceMatrix2),
            Pointer.to(deviceMatrix3),
            Pointer.to(new int[]{N})
        );

        // Call the kernel function.
        int threads = maxThreadsPerBlock[0];
        int blockSize = (int)Math.ceil((double)(N * N) / threads);
        
        WorkloadTimer.timerStart(TimerType.COMPUTATION.ordinal());
        cuLaunchKernel(function,
            blockSize,  1, 1,      // Grid dimension
            threads, 1, 1,      // Block dimension
            0, null,               // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
        WorkloadTimer.timerStop(TimerType.COMPUTATION.ordinal());
        
        WorkloadTimer.timerStart(TimerType.MEMORY_TRANSFERS.ordinal());
        cuMemcpyDtoH(Pointer.to(matrix3Linear), deviceMatrix3, N * N * Sizeof.FLOAT);
        WorkloadTimer.timerStop(TimerType.MEMORY_TRANSFERS.ordinal());

        // Deslinaraização dos dados para matrix3
        WorkloadTimer.timerStart(TimerType.DELINEARIZATION.ordinal());
        deslinearization(matrix3,matrix3Linear);
        WorkloadTimer.timerStop(TimerType.DELINEARIZATION.ordinal());

        WorkloadTimer.timerStop(TimerType.TOTAL.ordinal());

        MatrixMultiplicationUtils.verification(matrix3);
        Config.executionReport("Matrix Multiplication", 
            WorkloadTimer.timerRead(TimerType.TOTAL.ordinal()), 
            MatrixMultiplicationUtils.isPassedVerification(),
            MatrixMultiplicationUtils.getChecksumString(),
            MatrixMultiplicationUtils.getTimerString()
        );
        
        cuMemFree(deviceMatrix1);
        cuMemFree(deviceMatrix2);
        cuMemFree(deviceMatrix3);
    }

    private static void linearization(int[][] matrix1, int[][] matrix2, int[][] matrix3, int[] matrix1L, int[] matrix2L, int[] matrix3L){
        int n = Config.getWorkload().getN();
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix1L[i + j * n] = matrix1[i][j];
                matrix2L[i + j * n] = matrix2[i][j];
                matrix3L[i + j * n] = matrix3[i][j]; 
            }
        }
    }
    
    private static void deslinearization(int[][] matrix3, int[] matrix3L){
        int n = Config.getWorkload().getN();
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
               matrix3[i][j] = matrix3L[i + j * n]; 
            }
        }
    }
}
