import jcuda.*;
import jcuda.runtime.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

public class JCudaVectorAdd {
    public static void main(String[] args) {
        // Inicializa o JCuda
        JCuda.setExceptionsEnabled(true);
        cudaSetDevice(0);

        int n = 10;
        int size = n * Sizeof.FLOAT;

        // Cria vetores host (CPU)
        float hostA[] = new float[n];
        float hostB[] = new float[n];
        float hostC[] = new float[n];

        for (int i = 0; i < n; i++) {
            hostA[i] = i;
            hostB[i] = i * 2;
        }

        // Aloca memória no device (GPU)
        Pointer devA = new Pointer();
        Pointer devB = new Pointer();
        Pointer devC = new Pointer();

        cudaMalloc(devA, size);
        cudaMalloc(devB, size);
        cudaMalloc(devC, size);

        // Copia dados do host para o device
        cudaMemcpy(devA, Pointer.to(hostA), size, cudaMemcpyHostToDevice);
        cudaMemcpy(devB, Pointer.to(hostB), size, cudaMemcpyHostToDevice);

        // Chama o kernel (definido no CUDA C)
        JCudaKernelLauncher.launchVectorAdd(devA, devB, devC, n);

        // Copia resultado de volta
        cudaMemcpy(Pointer.to(hostC), devC, size, cudaMemcpyDeviceToHost);

        // Exibe resultado
        for (int i = 0; i < n; i++) {
            System.out.println(hostA[i] + " + " + hostB[i] + " = " + hostC[i]);
        }

        // Libera memória
        cudaFree(devA);
        cudaFree(devB);
        cudaFree(devC);
    }
}
