#include "SimpleMultiplication.h"
#include <memory>

#include <stdio.h>
#include <iostream>

void matMultiplyOnHost(float* A, float* B, float* C, int L, int M, int N) {
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0;
            for (int k = 0; k < M; k++) {
                C[i * N + j] += A[i * M + k] * B[k * N + j];
            }
        }
    }
    return;
}

int main()
{
    constexpr unsigned int L = 256, M = 256, N = 256;
    float* a;
    float* b;
    float* cCheck;
    float* cSimple;
    float* cShared;
    float* cWarp;

    a = (float*)malloc(sizeof(float) * L * M);
    b = (float*)malloc(sizeof(float) * M * N);

    cCheck = (float*)malloc(sizeof(float) * L * N);
    cSimple = (float*)malloc(sizeof(float) * L * N);
    cShared = (float*)malloc(sizeof(float) * L * N);
    cWarp = (float*)malloc(sizeof(float) * L * N);
   // std::cout << "a = {";
    for (int i = 0; i < L * M; i++) {
        a[i] = (rand() % 9999) / 2.0;
        //std::cout << a[i] << ((i != L * M - 1) ? ", " : "");
    }
    //std::cout << "}" << std::endl;
    //std::cout << "b = {";
    for (int i = 0; i < M * N; i++) {
        b[i] = (rand() % 9999) / 2.0;
       // std::cout << b[i] << ((i != M * N - 1) ? ", " : "");
    }
   // std::cout << "}" << std::endl;

    matMultiplyOnHost(a, b, cCheck, L, M, N);

    // Add vectors in parallel.
    cudaError_t cudaStatus = cudaMath::matrixMult(cSimple, a, b, L, M, N, cudaMath::MatrixMultiplicationMode::Simple);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    for (int i = 0; i < L * N; i++) {
        if (cCheck[i] != cSimple[i]) {
            printf("Simple Mismatch at Row = %d Col = %d real[] = %f --device[] %f\n", i / N,
                i % N, cCheck[i], cSimple[i]);
            break;
        }
    }

    cudaStatus = cudaMath::matrixMult(cShared, a, b, L, M, N, cudaMath::MatrixMultiplicationMode::SharedMemory);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    for (int i = 0; i < L * N; i++) {
        if (cCheck[i] != cShared[i]) {
            printf("Shared Mismatch at Row = %d Col = %d real[] = %f --device[] %f\n", i / N,
                i % N, cCheck[i], cShared[i]);
            break;
        }
    }
    cudaStatus = cudaMath::matrixMult(cWarp, a, b, L, M, N, cudaMath::MatrixMultiplicationMode::WarpInstrincts);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    for (int i = 0; i < L * N; i++) {
        if (cCheck[i] != cWarp[i]) {
            printf("Simple Mismatch at Row = %d Col = %d real[] = %f --device[] %f\n", i / N,
                i % N, cCheck[i], cWarp[i]);
            break;
        }
    }

    free(a);
    free(b);
    free(cCheck);
    free(cSimple);
    free(cShared);
    free(cWarp);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}