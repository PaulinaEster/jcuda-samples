# exercícios

## Repository organization
- serial: contains the serial applications
- serial/config: contains the definition of the C compiler and compilation flags
- include: contains the definitions for each application

## Applications
- matrix-multiplication
- array-add
- array-reduction
- search
- pi-number
- row-wise-reduction

## How to compile
1. Go to the directory `serial/config`, edit the file `make.def`, and define the compiler and compilation flags:
    ```
    CCOMPILER = gcc
    CFLAGS = -Wall -O3 -mcmodel=large -lm
    ```
2. Go to the directory of the desired application and perform the following command:
    ```
    make APPLICATION_NAME WORKLOAD=_WORKLOAD_VALUE DEBUG=_DEBUG_VALUE TIMER=_TIMER_VALUE
    ```
    `_WORKLOAD_VALUE` are:
        A, B, C, D, E, ... \
    `_DEBUG_VALUE` are:
        ON, OFF \
    `_TIMER_VALUE` are:
        ON, OFF

3. Command example:
    ```
    cd serial/array-add
    make array_add WORKLOAD=A DEBUG=ON TIMER=ON
    ./array_add.A.exe
