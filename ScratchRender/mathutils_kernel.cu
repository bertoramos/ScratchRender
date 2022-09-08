
#include <math.h>
#include <iostream>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cublas.h>
#include <cublas_api.h>
#include <cublas_v2.h>

#include "./mathutils_kernel.cuh"

#define cudacall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            std::cerr <<"CUDA Error:\nFile " << __FILE__ << "\nLine " << __LINE__ << "\nError " << cudaGetErrorString(err);    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define cublascall(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cublasStatus_t status = (call);                                                                         \
        if(CUBLAS_STATUS_SUCCESS != status)                                                                     \
        {                                                                                                       \
            std::cerr <<"CUBLAS Error:\nFile " << __FILE__ << "\nLine " << __LINE__ << "\nCode " << status; \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)



void mult_mm(float* A, float* B, float* C, int N) {

    cudaError_t error;
    cublasStatus_t state;
    cublasHandle_t handle;
    cublascall(cublasCreate(&handle));

    // Initialize device pointers.
    float* d_A, * d_B, * d_C;

    // Allocate device memory.
    cudacall(cudaMalloc((void**)&d_A, N * sizeof(float)));
    cudacall(cudaMalloc((void**)&d_B, N * sizeof(float)));
    cudacall(cudaMalloc((void**)&d_C, N * sizeof(float)));

    // Transfer arrays a and b to device.
    cudacall(cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice));
    cudacall(cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice));

    // *********** multiplication A*B

    int m = N, n = N, k = N;
    int lda = N, ldb = N, ldc = N;
    const float alf = 1;
    const float bet = 0;
    const float* alpha = &alf;
    const float* beta = &bet;

    cublascall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_B, ldb, d_A, lda, beta, d_C, ldc));

    // ***********

    cudacall(cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudacall(cudaFree(d_A));
    cudacall(cudaFree(d_B));
    cudacall(cudaFree(d_C));

    cublascall(cublasDestroy(handle));
    //cudaDeviceReset();
}

void mult_mv(float* A, float* B, float* C, int N) {

    cudaError_t error;
    cublasStatus_t state;
    cublasHandle_t handle;
    cublascall(cublasCreate(&handle));

    float* d_A, * d_B, * d_C;

    // Allocate device memory.
    cudacall(cudaMalloc((void**)&d_A, N * N * sizeof(float)));
    cudacall(cudaMalloc((void**)&d_B, N * sizeof(float)));
    cudacall(cudaMalloc((void**)&d_C, N * sizeof(float)));

    // Transfer arrays a and b to device.
    cudacall(cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    cudacall(cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice));

    // *********** multiplication A*v
    int m = N, n = N;
    int lda = N;
    const float alf = 1;
    const float bet = 0;
    const float* alpha = &alf;
    const float* beta = &bet;

    cublascall(cublasSgemv(handle, CUBLAS_OP_T, m, n, alpha, d_A, lda, d_B, 1, beta, d_C, 1));

    // ***********

    cudacall(cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudacall(cudaFree(d_A));
    cudacall(cudaFree(d_B));
    cudacall(cudaFree(d_C));

    cublascall(cublasDestroy(handle));
    //cudaDeviceReset();
}

void invert(float** src, float** dst, int n, int batchSize)
{
    cublasHandle_t handle;
    cublascall(cublasCreate_v2(&handle));

    int* P, * INFO;

    cudacall(cudaMalloc(&P, n * batchSize * sizeof(int)));
    cudacall(cudaMalloc(&INFO, batchSize * sizeof(int)));

    int lda = n;

    float** A = (float**)malloc(batchSize * sizeof(float*));
    float** A_d, * A_dflat;

    cudacall(cudaMalloc(&A_d, batchSize * sizeof(float*)));
    cudacall(cudaMalloc(&A_dflat, n * n * batchSize * sizeof(float)));

    A[0] = A_dflat;
    for (int i = 1; i < batchSize; i++)
        A[i] = A[i - 1] + (n * n);

    cudacall(cudaMemcpy(A_d, A, batchSize * sizeof(float*), cudaMemcpyHostToDevice));

    for (int i = 0; i < batchSize; i++)
        cudacall(cudaMemcpy(A_dflat + (i * n * n), src[i], n * n * sizeof(float), cudaMemcpyHostToDevice));


    cublascall(cublasSgetrfBatched(handle, n, A_d, lda, P, INFO, batchSize));

    int INFOh[1];
    cudacall(cudaMemcpy(INFOh, INFO, batchSize * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < batchSize; i++)
        if (INFOh[i] != 0)
        {
            fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
            cudacall(cudaDeviceReset());
            exit(EXIT_FAILURE);
        }

    float** C = (float**)malloc(batchSize * sizeof(float*));
    float** C_d, * C_dflat;

    cudacall(cudaMalloc(&C_d, batchSize * sizeof(float*)));
    cudacall(cudaMalloc(&C_dflat, n * n * batchSize * sizeof(float)));
    C[0] = C_dflat;
    for (int i = 1; i < batchSize; i++)
        C[i] = C[i - 1] + (n * n);
    cudacall(cudaMemcpy(C_d, C, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    cublascall(cublasSgetriBatched(handle, n, (const float**)A_d, lda, P, C_d, lda, INFO, batchSize));

    cudacall(cudaMemcpy(INFOh, INFO, batchSize * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < batchSize; i++)
        if (INFOh[i] != 0)
        {
            fprintf(stderr, "Inversion of matrix %d Failed: Matrix may be singular\n", i);
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }
    for (int i = 0; i < batchSize; i++)
        cudacall(cudaMemcpy(dst[i], C_dflat + (i * n * n), n * n * sizeof(float), cudaMemcpyDeviceToHost));

    cudacall(cudaFree(A_d)); cudacall(cudaFree(A_dflat)); free(A);
    cudacall(cudaFree(C_d)); cudacall(cudaFree(C_dflat)); free(C);
    cudacall(cudaFree(P)); cudacall(cudaFree(INFO));
    
    cublascall(cublasDestroy(handle));
}