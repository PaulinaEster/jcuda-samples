#!/bin/bash

HOME=$(pwd)

URL_JCUDA="https://repo1.maven.org/maven2/org/jcuda/jcuda/11.8.0/jcuda-11.8.0.jar"
URL_JDK_11="https://builds.openlogic.com/downloadJDK/openlogic-openjdk/11.0.28+6/openlogic-openjdk-11.0.28+6-linux-x64.tar.gz"

PASTA_DESTINO=$HOME"/../libs"
mkdir -p $PASTA_DESTINO

JDK_DEST="$PASTA_DESTINO/JDK_8.tar.gz"
JCUDA_DEST="$PASTA_DESTINO/jcuda-11.8.0.jar"

JDK_HOME="$PASTA_DESTINO/openlogic-openjdk-11.0.28+6-linux-x64"
JCUDA_HOME=$JCUDA_DEST

# BAIXANDO DEPENDENCIAS SE NECESSÁRIO
if [ ! -f "$JDK_DEST" ]; then
  echo "Arquivo $JDK_DEST não encontrado. Baixando..."
  echo "Baixando JDK 8"
  wget -O "$JDK_DEST" "$URL_JDK_11"
fi

if [ ! -f "$JCUDA_DEST" ]; then
  echo "Arquivo $JCUDA_DEST não encontrado. Baixando..."
  echo "Baixando JCUDA 10.1.0"
  wget -O "$JCUDA_DEST" "$URL_JCUDA"
fi

# DESCOMPACTANDO ARQUIVOS SE NECESSÁRIO
if [ ! -d "$JDK_HOME" ]; then
  echo "Arquivo $JDK_HOME não encontrado. Descompactando..."
  echo "Descompactando JDK "
  tar -xzf "$JDK_DEST" -C "$PASTA_DESTINO"
fi
# # module load cuda/11.8
# if [ ! -d "$JCUDA_HOME" ]; then
#   echo "Arquivo $JCUDA_HOME não encontrado. Descompactando..."
#   echo "Descompactando JCUDA"
#   unzip -o "$JCUDA_HOME" -d "$PASTA_DESTINO"
# fi

echo "REMOVENDO .class EXISTENTES"
find . -name "*.class" -type f -exec rm -f {} \;

echo "EXECUTANDO JAVAC"
"$JDK_HOME/bin/javac" \
  -cp $JCUDA_HOME \
  JCudaVectorAdd.java JCudaKernelLauncher.java

echo "GERANDO PTX DO KERNEL EM CU"
nvcc -ptx vectorAdd.cu -o vectorAdd.ptx

echo "EXECUTANDO JAVA"
"$JDK_HOME/bin/java" \
  -cp .:jcuda-10.1.0.jar \
  JCudaVectorAdd
