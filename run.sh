#!/bin/bash

HOME=$(pwd)

URL_JDK_8="https://builds.openlogic.com/downloadJDK/openlogic-openjdk/8u462-b08/openlogic-openjdk-8u462-b08-linux-x64.tar.gz"
URL_MAVEN="https://dlcdn.apache.org/maven/maven-3/3.9.11/binaries/apache-maven-3.9.11-bin.tar.gz"

PASTA_DESTINO=$HOME"/libs"
mkdir -p $PASTA_DESTINO

JDK_DEST="$PASTA_DESTINO/JDK_8.tar.gz"
MAVEN_DEST="$PASTA_DESTINO/apache-maven-3.9.11-bin.tar.gz"

JDK_HOME="$PASTA_DESTINO/openlogic-openjdk-8u462-b08-linux-x64"
MAVEN_HOME="$PASTA_DESTINO/apache-maven-3.9.11-bin"

# BAIXANDO JDK 8
if [ ! -f "$JDK_DEST" ]; then
  echo "Arquivo $JDK_DEST não encontrado. Baixando..."
  echo "Baixando JDK 8"
  wget -O "$JDK_DEST" "$URL_JDK_8"
fi
# DESCOMPACTANDO JDK 8
if [ ! -d "$JDK_HOME" ]; then
  echo "Arquivo $JDK_HOME não encontrado. Descompactando..."
  echo "Descompactando JDK "
  tar -xzf "$JDK_DEST" -C "$PASTA_DESTINO"
  echo "Apagando $JDK_DEST"
  rm $JDK_DEST
fi


# BAIXANDO MAVEN 3.9.11
if [ ! -f "$URL_MAVEN" ]; then
  echo "Arquivo $URL_MAVEN não encontrado. Baixando..."
  echo "Baixando Maven 3.9.11"
  wget -O "$MAVEN_DEST" "$URL_MAVEN"
fi
 
# DESCOMPACTANDO MAVEN 3.9.11
if [ ! -d "$MAVEN_HOME" ]; then
  echo "Arquivo $MAVEN_HOME não encontrado. Descompactando..."
  echo "Descompactando Maven "
  tar -xzf "$MAVEN_DEST" -C "$PASTA_DESTINO"
  echo "Apagando $MAVEN_DEST"
  rm $MAVEN_DEST
fi
