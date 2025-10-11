package serial.matrixmultiplication;

import utils.MatrixMultiplicationUtils;
import utils.Config;
public class MatrixMultiplication
{
    /**
     * Entry point of this sample
     *
     * @param args Not used
     * @throws IOException If an IO error occurs
     */
    public static void main(String args[])  
    {
        System.out.println("Funcionando");
        
        Config.setupCommon();
        int N = Config.getWorkload().getN();
        int[][] matrix1 = new int[N][N];
        int[][] matrix2 = new int[N][N];
        int[][] matrix3 = new int[N][N];
        MatrixMultiplicationUtils.initialization(matrix1, matrix2, matrix3);
    
        
        matrixMultiplication(matrix1, matrix2, matrix3, N);

        MatrixMultiplicationUtils.verification(matrix3);

        Config.executionReport("Matrix Multiplication", 
            0.0, 
            MatrixMultiplicationUtils.isPassedVerification(), 
            "", 
            MatrixMultiplicationUtils.getChecksumString(),
            ""
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
