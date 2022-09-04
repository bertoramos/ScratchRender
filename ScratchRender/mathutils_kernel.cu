
#include <math.h>
#include <iostream>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "./mathutils_kernel.cuh"

__global__ void mult_mm_kernel(float* A, float* B, float* C, int N) {

    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
}


void mult_mm(float* A, float* B, float* C, int N) {

    // Initialize device pointers.
    float* d_A, * d_B, * d_C;

    // Allocate device memory.
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // Transfer arrays a and b to device.
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block

    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
    if (N * N > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(float(N) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(float(N) / double(threadsPerBlock.y));
    }

    mult_mm_kernel << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void mult_mv_kernel(float* A, float* B, float* C, int N) {

    int ROW = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpSum = 0;

    if (ROW < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW*N + i] * B[i];
        }
        C[ROW] = tmpSum;
    }
    
}


void mult_mv(float* A, float* B, float* C, int N) {

    float* d_A, * d_B, * d_C;

    // Allocate device memory.
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // Transfer arrays a and b to device.
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    dim3 blocks(1, 1);
    dim3 threadsPerBlock(N, N);
    if (N * N > 512) {
        blocks.x = 512;
        blocks.y = 512;
        threadsPerBlock.x = ceil(double(N) / double(blocks.x));
        threadsPerBlock.y = ceil(double(N) / double(blocks.y));
    }

    mult_mv_kernel << <blocks, threadsPerBlock >> > (d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
