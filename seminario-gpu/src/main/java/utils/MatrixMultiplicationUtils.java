package utils;

import java.util.logging.Logger;
import java.lang.StringBuilder; 

import utils.Config;
import utils.WorkloadTimer;
import utils.Workload;
import utils.TimerType;

public class MatrixMultiplicationUtils
{
    
    private static final Logger logger = Logger.getLogger(MatrixMultiplicationUtils.class.getName());

    private static boolean passedVerification;
    private static String checksumString;
    private static String timerString;

    public static boolean isPassedVerification() { return passedVerification;}
    public static String getChecksumString() { return checksumString;}
    public static String getTimerString() {return timerString;}
    
    public static void initialization(int[][] matrix1, int[][] matrix2, int[][] matrix3) {
        // Config.setupCommon();
        int n = Config.getWorkload().getN();
        // initial values of the global arrays
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix1[i][j] = 4;
                matrix2[i][j] = 5;
                matrix3[i][j] = 0;

                if (i == j) {
                    matrix1[i][j] = i;
                    matrix2[i][j] = j;
                }
            }
        }
    } 

    public static void verification(int[][] matrix) {
        long value1 = 0;
        long value2 = 0;
        long value3 = 0;
        long value4 = 0;
        Workload workload = Config.getWorkload();
        int N = workload.getN();

        // total
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                value1 += matrix[i][j];
            }
        }

        // column 0
        for (int i = 0; i < N; i++) {
            value2 += matrix[i][0];
        }

        // diagonal
        for (int i = 0; i < N; i++) {
            value3 += matrix[i][i];
        }

        // line N-1
        for (int j = 0; j < N; j++) {
            value4 += matrix[N - 1][j];
        }

        // verification
        passedVerification = (
            workload.getChecksum1() == value1 &&
            workload.getChecksum2() == value2 &&
            workload.getChecksum3() == value3 &&
            workload.getChecksum4() == value4
        );

        // checksum output
        StringBuilder checksumBuilder = new StringBuilder();
        checksumBuilder.append(String.format("%25s\t%20s\t%20s%n", "Reference", "Correct", "Found"));
        checksumBuilder.append(String.format("%25s\t%20d\t%20d%n", "checksum_1", workload.getChecksum1(), value1));
        checksumBuilder.append(String.format("%25s\t%20d\t%20d%n", "checksum_2", workload.getChecksum2(), value2));
        checksumBuilder.append(String.format("%25s\t%20d\t%20d%n", "checksum_3", workload.getChecksum3(), value3));
        checksumBuilder.append(String.format("%25s\t%20d\t%20d", "checksum_4", workload.getChecksum4(), value4));

        checksumString = checksumBuilder.toString();

        // timer output
        StringBuilder timerBuilder = new StringBuilder();
        // timerBuilder.append(String.format("%25s\t%20s\t%20s%n", "Timer", "Time (s)", "Percentage"));
        // double total = WorkloadTimer.read(TimerType.TOTAL.ordinal());
        // double mem   = WorkloadTimer.read(TimerType.MEMORY_TRANSFERS.ordinal());
        // double lin   = WorkloadTimer.read(TimerType.LINEARIZATION.ordinal());
        // double comp  = WorkloadTimer.read(TimerType.COMPUTATION.ordinal());

        // timerBuilder.append(String.format("%25s\t%20f\t%19.2f%%%n", "memory_transfers", mem, mem * 100 / total));
        // timerBuilder.append(String.format("%25s\t%20f\t%19.2f%%%n", "linearization", lin, lin * 100 / total));
        // timerBuilder.append(String.format("%25s\t%20f\t%19.2f%%", "matrix_multiplication", comp, comp * 100 / total));

        timerString = timerBuilder.toString();
    }
}
