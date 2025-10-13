# jcuda-samples

# ALOCAR GPU
```bash
srun --time 60 --mem 3000 --cpus-per-task 1 --gpus 1 --pty bash -i
module load cuda/11.8 gcc/11.3 g++/11.3
module available 
```

## BAIXAR JDK E MAVEN
```bash 
chmod +x run.sh
./run.sh
```

## APONTAR O JAVA_HOME
```bash
export JAVA_HOME=<YOUR_PATH>/jcuda-samples/libs/openlogic-openjdk-8u462-b08-linux-x64
export PATH=$JAVA_HOME/bin:$PATH

source ~/.bashrc
```

## COMPILAR O PROJETO
```bash
../libs/apache-maven-3.9.11/bin/mvn clean compile
```

## EXECUTAR O EXEMPLO QUE DESEJA
```bash
../libs/apache-maven-3.9.11/bin/mvn exec:java -Dexec.mainClass=jcuda.driver.samples.JCudaVectorAdd

../libs/apache-maven-3.9.11/bin/mvn exec:java -Dexec.mainClass=serial.matrixmultiplication.MatrixMultiplication -DDEBUG=true -DTIMER=true -DWORKLOAD=G
```