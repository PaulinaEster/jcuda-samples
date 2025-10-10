#!/bin/bash

# http://www.jcuda.org/downloads/JCuda-All-10.1.0.zip
# https://builds.openlogic.com/downloadJDK/openlogic-openjdk/8u462-b08/openlogic-openjdk-8u462-b08-linux-x64.tar.gz

HOME=$(pwd)

URL_JCUDA="http://www.jcuda.org/downloads/JCuda-All-10.1.0.zip"
URL_JDK_8="https://builds.openlogic.com/downloadJDK/openlogic-openjdk/8u462-b08/openlogic-openjdk-8u462-b08-linux-x64.tar.gz"

PASTA_DESTINO=$HOME"/../libs"
mkdir -p $PASTA_DESTINO

JDK_DEST="$PASTA_DESTINO/JDK_8.tar.gz"
JCUDA_DEST="$PASTA_DESTINO/JCuda-10-1-0.zip"

JDK_HOME="$PASTA_DESTINO/openlogic-openjdk-8u462-b08-linux-x64"
JCUDA_HOME="$PASTA_DESTINO/JCuda-All-10.1.0"

# BAIXANDO DEPENDENCIAS SE NECESSÁRIO
if [ ! -f "$JDK_DEST" ]; then
  echo "Arquivo $JDK_DEST não encontrado. Baixando..."
  echo "Baixando JDK 8"
  wget -O "$JDK_DEST" "$URL_JDK_8"
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

if [ ! -d "$JCUDA_HOME" ]; then
  echo "Arquivo $JCUDA_HOME não encontrado. Descompactando..."
  echo "Descompactando JCUDA"
  unzip -o "$JCUDA_HOME" -d "$PASTA_DESTINO"
fi

echo "REMOVENDO .class EXISTENTES"
find . -name "*.class" -type f -exec rm -f {} \;

echo "EXECUTANDO JAVAC"
"$JDK_HOME/bin/javac" \
  -cp $JCUDA_HOME/jcuda-10.1.0.jar \
  JCudaVectorAdd.java JCudaKernelLauncher.java

nvcc -ptx vectorAdd.cu -o vectorAdd.ptx

echo "EXECUTANDO JAVA"
"$JDK_HOME/bin/java" \
  -cp .:jcuda-10.1.0.jar \
  JCudaVectorAdd
