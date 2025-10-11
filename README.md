# jcuda-samples

This repository contains samples for the JCuda libraries.

**Note:** Some of the samples require third-party libraries, JCuda
libraries that are not part of the [`jcuda-main`](https://github.com/jcuda/jcuda-main) 
package (for example, [`JCudaVec`](https://github.com/jcuda/jcuda-vec) or 
[`JCudnn`](https://github.com/jcuda/jcudnn)), or utility libraries
that are not available in Maven Central. In order to compile these
samples, additional setup steps may be necessary. The main goal
of this repository is to collect and maintain the samples in a 
form that allows them to serve as a collection of snippets that
can easily be copied and pasted into own projects to get started.

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
```
