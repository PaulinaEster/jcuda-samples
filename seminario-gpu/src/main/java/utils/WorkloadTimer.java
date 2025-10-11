
package utils;

import utils.TimerType;

public class WorkloadTimer {
    private static final int MAX_TIMERS = 10; // defina conforme o número necessário
    private static final double[] elapsed = new double[MAX_TIMERS];
    private static final double[] start = new double[MAX_TIMERS];
    private static long baseTime = -1;

    private static double timerWriteTime() {
        long now = System.nanoTime(); // tempo em nanos
        if (baseTime < 0) {
            baseTime = now;
        }
        // converte para segundos desde baseTime
        return (now - baseTime) / 1.0e9;
    }

    public static double timerElapsedTime() {
        return timerWriteTime();
    }

    public static void timerClear(int n) {
        elapsed[n] = 0.0;
    }

    public static void timerStart(int n) {
        start[n] = timerElapsedTime();
    }

    public static void timerStop(int n) {
        double now = timerElapsedTime();
        double t = now - start[n];
        elapsed[n] += t;
    }

    public static double timerRead(int n) {
        return elapsed[n];
    }
}
