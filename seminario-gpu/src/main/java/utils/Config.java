package utils;

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
}
