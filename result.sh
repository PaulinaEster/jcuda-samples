#!/bin/bash


declare -A DIRETORIOS 

# DIRETORIOS["./seminario-gpu"]="jcuda.matrixmultiplication.JCudaMatrixMultiplication|serial.matrixmultiplication.MatrixMultiplication"
DIRETORIOS["./seminario-gpu"]="serial.matrixmultiplication.MatrixMultiplication" 

# classes=("H" "G")
classes=("H" "H" "H" "H" "H")
timers=("matrix_multiplication" "memory_transfers" "linearization" "deslinearization" "jcuda_driver")
# Arquivo que terá a saida do resultado.
resultado="./resultados/resultado.log"
# Arquivo de log temporario
logfile="./resultados/execucao.log"

# Limpa/cria arquivos de log
> "$logfile"
# > "$resultado"

HOME_DIR="$(pwd)"

for dir in "${!DIRETORIOS[@]}"
do
    echo "---------- COMPILANDO $dir -----------"
    (source ~/.bashrc && cd "$dir" && ../libs/apache-maven-3.9.11/bin/mvn clean compile)
done

for dir in "${!DIRETORIOS[@]}"
do 
    IFS=$'|' read -ra benchmarks <<<"${DIRETORIOS[$dir]}"
    for bench in "${benchmarks[@]}"
    do 
        for workload in "${classes[@]}"
        do
            (cd "$dir" && ../libs/apache-maven-3.9.11/bin/mvn exec:java -Dexec.mainClass=$bench -DTIMER=true -DWORKLOAD=$workload) >> "$logfile" 2>&1
            if [ $? -ne 0 ]; then
                echo "Erro na execução de '../libs/apache-maven-3.9.11/bin/mvn exec:java -Dexec.mainClass=$bench -DTIMER=true -DWORKLOAD=$workload' em $dir. Abortando." | tee -a "$logfile"
                exit 1
            fi 
            echo -n -e "$bench WORKLOAD=$workload \t TEMPO: " | tee -a >> "$resultado" 

            sed -n 's/.*Execution time in seconds *= *\([0-9,]*\).*/\1/p' $logfile | tr -d '\n' >> "$resultado"
            
            for timer in "${timers[@]}"
            do  
                grep -m1 -E "$timer" "$logfile" | tr -d '\n' >> "$resultado"  
            done
            > "$logfile" 
            echo " " >> $resultado
        done
    done
done
CUDA_DIR=("./matrix-multiplication/serial/matrix-multiplication")

for dir in "${CUDA_DIR[@]}"
do 
    for workload in "${classes[@]}"
    do  
        cd $dir && make matrix_multiplication WORKLOAD=$workload TIMER=ON
        cd $HOME_DIR && $dir/matrix_multiplication.$workload.exe >> "$logfile" 2>&1
        if [ $? -ne 0 ]; then
            echo "Erro na execução do '$dir/matrix_multiplication.$workload.exe' em $dir. Abortando."
            exit 1
        fi 
        echo -n -e "$dir.$workload \t TEMPO: " | tee -a >> "$resultado" 
        
        sed -n 's/.*Execution time in seconds *= *\([0-9.]*\).*/\1/p' $logfile | tr -d '\n' >> "$resultado"

        for timer in "${timers[@]}"
        do  
            grep -m1 -E "$timer" "$logfile" | tr -d '\n' >> "$resultado" 
            ((count++)) 
        done
        > "$logfile" 
        echo " " >> $resultado 
    done
done

echo "Script finalizado. Saída salva em $logfile."
