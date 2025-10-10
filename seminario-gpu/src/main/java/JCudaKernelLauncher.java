import jcuda.*;
import jcuda.driver.*;
import static jcuda.driver.JCudaDriver.*;

public class JCudaKernelLauncher {
    private static boolean initialized = false;
    private static CUmodule module;
    private static CUfunction function;

    public static void launchVectorAdd(Pointer devA, Pointer devB, Pointer devC, int n) {
        if (!initialized) {
            JCudaDriver.setExceptionsEnabled(true);
            cuInit(0);

            CUdevice device = new CUdevice();
            cuDeviceGet(device, 0);

            CUcontext context = new CUcontext();
            cuCtxCreate(context, 0, device);

            module = new CUmodule();
            cuModuleLoad(module, "vectorAdd.ptx");

            function = new CUfunction();
            cuModuleGetFunction(function, module, "vectorAdd");

            initialized = true;
        }

        int blockSize = 256;
        int gridSize = (int) Math.ceil((double) n / blockSize);

        Pointer kernelParameters = Pointer.to(
            Pointer.to(devA),
            Pointer.to(devB),
            Pointer.to(devC),
            Pointer.to(new int[]{n})
        );

        cuLaunchKernel(function,
            gridSize, 1, 1,      // grid
            blockSize, 1, 1,     // block
            0, null,             // shared memory, stream
            kernelParameters, null
        );

        cuCtxSynchronize();
    }
}
