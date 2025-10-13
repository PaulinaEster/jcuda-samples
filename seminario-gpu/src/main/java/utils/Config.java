package utils;

import java.io.BufferedReader;
import java.io.InputStreamReader;

import utils.Workload; 

public class Config
{
    private static boolean DEBUG = false;
    private static boolean TIMER = false;
    private static Workload WORKLOAD = Workload.A;
 
    public static boolean getDebug(){ return DEBUG;}
    public static boolean getTimer(){ return TIMER;}
    public static Workload getWorkload(){ return WORKLOAD;}

    private static void setTimer(){
        boolean timerSet = Boolean.getBoolean("TIMER");
        if(timerSet == true){
            TIMER = true;
        }

        System.out.println("Timer " + (TIMER ? "ON" : "OFF"));
    }
    
    private static void setDebug(){
        boolean debugSet = Boolean.getBoolean("DEBUG"); 

        if(debugSet == true){
            DEBUG = true;
        }
        System.out.println("Debug " + (DEBUG ? "ON" : "OFF"));
    }

    private static void setWorkload(){
        String workload = System.getProperty("WORKLOAD", "A");
        if(workload != null){
            WORKLOAD = Workload.fromName(workload);
        }
        System.out.println("Workload " + WORKLOAD.getName());
    }
    
    public static void setupCommon(){
        setTimer();
        setDebug();
        setWorkload();
    }

    public static void executionReport(
        String applicationName, 
        double executionTime,
        boolean passedVerification,
        String checksumString,
        String timerString
    ) {
        String cpuName = getCpuName();
        System.out.println("----------------------------------------------------------------------------");
        System.out.println(" " + applicationName + ":");
        System.out.println();
        System.out.println(" Workload                  =     " + WORKLOAD.getName());
        System.out.printf(" Execution time in seconds =     %.6f%n", executionTime);

        if (passedVerification) { System.out.println(" Correctness verification  =     SUCCESSFUL"); } 
        else { System.out.println(" Correctness verification  =     UNSUCCESSFUL"); }

        System.out.println("----------------------------------------------------------------------------");
        System.out.println(" Hardware:");
        System.out.println();
        System.out.println(" CPU                       =     " + cpuName);
        System.out.println("----------------------------------------------------------------------------");
        System.out.println(" Flags:");
        System.out.println();
        System.out.println(" Debug flag " + (DEBUG ? "enabled" : "disabled"));
        System.out.println(" Timer flag " + (TIMER ? "enabled" : "disabled"));
        System.out.println("----------------------------------------------------------------------------");
        System.out.println(" Correctness:");
        System.out.println();
        System.out.println(checksumString);
        System.out.println("----------------------------------------------------------------------------");

        if (TIMER) {
            System.out.println(" Timers:");
            System.out.println();
            System.out.println(timerString);
            System.out.println("----------------------------------------------------------------------------");
        }
    }

    private static String getCpuName(){
       try {
            String os = System.getProperty("os.name").toLowerCase();
            String[] command;

            command = new String[]{"bash", "-c", "LANG=C lscpu | grep 'Model name' | awk -F: '{print $2}'"};

            Process process = new ProcessBuilder(command).start();
            String line;
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                while ((line = reader.readLine()) != null) {
                    return line.trim();
                }
            }
            return " - ";
        } catch (Exception e) {
            e.printStackTrace();
            return " - ";
        }
    }
}