package serial.matrixmultiplication;

import utils.MatrixMultiplicationUtils;
import utils.Config;
import utils.WorkloadTimer;
import utils.TimerType;

public class MatrixMultiplication
{
    public static void main(String args[])  
    {   
        Config.setupCommon();
        
        WorkloadTimer.timerStart(TimerType.TOTAL.ordinal());

        int N = Config.getWorkload().getN();
        int[][] matrix1 = new int[N][N];
        int[][] matrix2 = new int[N][N];
        int[][] matrix3 = new int[N][N];

        MatrixMultiplicationUtils.initialization(matrix1, matrix2, matrix3);
    
        WorkloadTimer.timerStart(TimerType.COMPUTATION.ordinal());
        matrixMultiplication(matrix1, matrix2, matrix3, N);
        WorkloadTimer.timerStop(TimerType.COMPUTATION.ordinal());

        WorkloadTimer.timerStop(TimerType.TOTAL.ordinal());
        MatrixMultiplicationUtils.verification(matrix3);

        Config.executionReport("Matrix Multiplication", 
            WorkloadTimer.timerRead(TimerType.TOTAL.ordinal()), 
            MatrixMultiplicationUtils.isPassedVerification(), 
            "", 
            MatrixMultiplicationUtils.getChecksumString(),
            MatrixMultiplicationUtils.getTimerString()
        );
    }

    public static void matrixMultiplication(int[][] matrix1, int[][] matrix2, int[][] matrix3, int N) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix3[i][j] = 0; // inicializa a posição antes de somar
                for (int k = 0; k < N; k++) {
                    matrix3[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
    } 
}
