# Seminario Java e GPU

## ALOCAR GPU
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

../libs/apache-maven-3.9.11/bin/mvn exec:java -Dexec.mainClass=jcuda.matrixmultiplication.JCudaMatrixMultiplication -DTIMER=true -DWORKLOAD=G
```

## Report

Neste trabalho você deve implementar a versão paralela de um dos programas acima listados e depois fazer: descrever o ambiente de testes e discutir os resultados obtidos. 

## Descrição do Ambiente de Testes

- SO: Debian 6.1.148-1
- Kernel Linux: Linux slurm-head 6.1.0-39-amd64
- CPU: 
- GPU:  
- Versões de Software:
  - Compilador: GCC 11.3 G++ 11.3
  - CUDA: 11.8
  - JDK: 8
  - Maven: 3.9


#### Tempo médio de execução total 
Se considerar o tempo total a execução com GPU teve duração maior que a execução em CPU. <br/>
O código executado na CPU teve duração de 0.567717s com desvio padrão de 0.004795s.  <br/>
A versão GPU teve um tempo total médio de 11.067040s com desvio padrão de 0.074159s, GPU com otimizações o tempo total médio passou a ser 2.913574s com desvio padrão de 0.008809s.

![Tempo de execução total](./resultado/speedup-tempo-total.png)


#### Tempo médio de execução Kernel
Considerando o tempo de execução apenas do Kernel a  teve resultados melhores que o código sequencial. <br/>
A execução serial teve duração de 0.326874s com desvio padrão de 0.003859s.  <br/>
O kernel de GPU deve duração média de 0.014477s com desvio padrão de 0.018403s.  <br/>
O kernel de GPU com otimizações teve um tempo total médio de 0.001845s e um desvio padrão de 0.000006s. 

![Tempo de execução total](./resultado/speedup-tempo-gray-trans.png)