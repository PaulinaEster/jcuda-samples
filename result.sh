#!/bin/bash


declare -A DIRETORIOS 

# DIRETORIOS["./seminario-gpu"]="jcuda.matrixmultiplication.JCudaMatrixMultiplication|serial.matrixmultiplication.MatrixMultiplication"
DIRETORIOS["./seminario-gpu"]="serial.matrixmultiplication.MatrixMultiplication" 

# classes=("H" "G")
classes=("A" "B" "C" "D" "F" "G" "H")
timers=("matrix_multiplication" "memory_transfers" "linearization" "deslinearization")
# Arquivo que terá a saida do resultado.
resultado="./resultados/resultado.log"
# Arquivo de log temporario
logfile="./resultados/execucao.log"

# Limpa/cria arquivos de log
> "$logfile"
> "$resultado"

count=1 
for dir in "${!DIRETORIOS[@]}"
do
    echo "---------- COMPILANDO $dir -----------"
    (source ~/.bashrc && cd "$dir" && ../libs/apache-maven-3.9.11/bin/mvn clean compile)
done


for dir in "${!DIRETORIOS[@]}"
do 
    IFS=$'|' read -ra benchmarks <<<"${DIRETORIOS[$dir]}"
    for workload in "${classes[@]}"
    do
        for bench in "${benchmarks[@]}"
        do
            echo "----- INICIO EXECUÇÃO DE $dir $bench" >> $resultado
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
                ((count++)) 
            done
            > "$logfile" 
            echo " " >> $resultado
        done
    done
done


# for dir in "${!cppruns[@]}"
# do
#     echo "----- INICIO EXECUÇÃO SEQUENCIAL DE $dir" >> $resultado
#     for workload in "${classes[@]}"
#     do
#         IFS=$'|' read -ra commands <<<"${cppruns[$dir]}"
#         for cmd in "${commands[@]}"
#         do
#             echo "==== $count EXECUÇÃO SEQUENCIAL $count: GCC $dir/$cmd.$workload "
#             ($dir/$cmd.$workload) >> "$logfile" 2>&1
#             if [ $? -ne 0 ]; then
#                 echo "Erro na execução do gcc '$cmd.$workload' em $dir. Abortando."
#                 exit 1
#             fi  
#             echo -n -e "$dir/$cmd.$workload \t ENTRADA: " | tee -a >> "$resultado" 
#             sed -n 's/.*Size *= *\([0-9.]*\).*/\1/p' $logfile | tr -d '\n' >> "$resultado"
#             echo -n " \t TEMPO: " | tee -a >> "$resultado" 
#             sed -n 's/.*Time in seconds *= *\([0-9.]*\).*/\1/p' $logfile >> "$resultado" 
#             > "$logfile" 
#             ((count++))
#         done
#     done
# done
echo "Script finalizado. Saída salva em $logfile."
