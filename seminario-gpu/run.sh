#!/bin/bash

HOME=$(pwd)

URL_JCUDA="https://repo1.maven.org/maven2/org/jcuda/jcuda/11.8.0/jcuda-11.8.0.jar"
URL_JCUBLAS="https://repo1.maven.org/maven2/org/jcuda/jcublas/11.8.0/jcublas-11.8.0.jar"
URL_JCUDA_NATIVES="https://repo1.maven.org/maven2/org/jcuda/jcuda-natives/11.8.0/jcuda-natives-11.8.0.jar"

URL_JDK_8="https://builds.openlogic.com/downloadJDK/openlogic-openjdk/8u462-b08/openlogic-openjdk-8u462-b08-linux-x64.tar.gz"

URL_MAVEN="https://dlcdn.apache.org/maven/maven-3/3.9.11/binaries/apache-maven-3.9.11-bin.tar.gz"

PASTA_DESTINO=$HOME"/../libs"
mkdir -p $PASTA_DESTINO

JDK_DEST="$PASTA_DESTINO/JDK_8.tar.gz"
MAVEN_DEST="$PASTA_DESTINO/apache-maven-3.9.11-bin.tar.gz"

JCUDA_DEST="$PASTA_DESTINO/jcuda-11.8.0.jar"

JDK_HOME="$PASTA_DESTINO/openlogic-openjdk-8u462-b08-linux-x64"
JCUDA_HOME=$JCUDA_DEST
JCUBLAS_HOME="$PASTA_DESTINO/jcublas-11.8.0.jar"
JCUDA_NATIVES_HOME="$PASTA_DESTINO/jcuda-natives-11.8.0.jar"
MAVEN_HOME="$PASTA_DESTINO/apache-maven-3.9.11-bin"
# BAIXANDO DEPENDENCIAS SE NECESSÁRIO
if [ ! -f "$JDK_DEST" ]; then
  echo "Arquivo $JDK_DEST não encontrado. Baixando..."
  echo "Baixando JDK 8"
  wget -O "$JDK_DEST" "$URL_JDK_8"
fi
if [ ! -f "$URL_MAVEN" ]; then
  echo "Arquivo $URL_MAVEN não encontrado. Baixando..."
  echo "Baixando Maven 3.9.11"
  wget -O "$MAVEN_DEST" "$URL_MAVEN"
fi

# DESCOMPACTANDO ARQUIVOS SE NECESSÁRIO
if [ ! -d "$JDK_HOME" ]; then
  echo "Arquivo $JDK_HOME não encontrado. Descompactando..."
  echo "Descompactando JDK "
  tar -xzf "$JDK_DEST" -C "$PASTA_DESTINO"
fi
if [ ! -d "$MAVEN_HOME" ]; then
  echo "Arquivo $MAVEN_HOME não encontrado. Descompactando..."
  echo "Descompactando Maven "
  tar -xzf "$MAVEN_DEST" -C "$PASTA_DESTINO"
fi



# echo "REMOVENDO .class EXISTENTES"
# find . -name "*.class" -type f -exec rm -f {} \;

# echo "EXECUTANDO JAVAC"
# "$JDK_HOME/bin/javac" \
#   -cp .:$JCUDA_HOME:$JCUBLAS_HOME:$JCUDA_NATIVES_HOME  \
#   JCudaVectorAdd.java JCudaKernelLauncher.java

# echo "GERANDO PTX DO KERNEL EM CU"
# nvcc -ptx vectorAdd.cu -o vectorAdd.ptx

# echo "EXECUTANDO JAVA"

# echo "$JDK_HOME/bin/java" \
#   -cp .:$JCUDA_HOME:$JCUBLAS_HOME:$JCUDA_NATIVES_HOME \
#   JCudaVectorAdd

# "$JDK_HOME/bin/java" \
#   -cp .:$JCUDA_HOME:$JCUBLAS_HOME:$JCUDA_NATIVES_HOME \
#   JCudaVectorAdd

echo "Execute: $PASTA_DESTINO/apache-maven-3.9.11/bin/mvn clean compile"
/home/paulinarehbein/jcuda-samples/seminario-gpu/../libs/apache-maven-3.9.11/bin/mvn exec:java -Dexec.mainClass=jcuda.driver.samples.JCudaVectorAdd