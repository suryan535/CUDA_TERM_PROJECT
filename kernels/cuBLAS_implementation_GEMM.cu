%%cuda --name cuBLAS_MatrixMul.cu
#include <bits/stdc++.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int WA, HA, WB, HB, WC, HC;
} sMatrixSize;


void matrix_Mul_CPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    unsigned int i,j,k;
    for ( i = 0; i < hA; i++)
        for (j = 0; j < wB; j++)
        {
            double sum = 0;
            for ( k = 0; k < wA; k++)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum =sum+a * b;
            }

            C[i * wB + j] = (float)sum;
        }
}

// Allocates a matrix with random float entries.
void random_mat_fill(float *data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}
void initializeCUDA(int argc, char **argv, int &devID, int &iSizeMultiple, sMatrixSize &matrix_size)
{ 
    devID = 0;
    cudaSetDevice(devID);
    iSizeMultiple = 5;

    // use a larger block size for Fermi and above
    int block_size = 32;

    matrix_size.WA = 5 * block_size * iSizeMultiple;
    matrix_size.HA = 3 * block_size * iSizeMultiple;
    matrix_size.WB = 2 * block_size * iSizeMultiple;
    matrix_size.HB = 5 * block_size * iSizeMultiple;
    matrix_size.WC = 2 * block_size * iSizeMultiple;
    matrix_size.HC = 3 * block_size * iSizeMultiple;

    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
           matrix_size.HA, matrix_size.WA,
           matrix_size.HB, matrix_size.WB,
           matrix_size.HC, matrix_size.WC);

    if( matrix_size.WA != matrix_size.HB ||
        matrix_size.HA != matrix_size.HC ||
        matrix_size.WB != matrix_size.WC)
    {
       printf("ERROR: Matrix sizes do not match!\n");
       exit(-1);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test matrix multiply using CUBLAS
////////////////////////////////////////////////////////////////////////////////
void matrix_Multiply(int argc, char **argv, int devID, sMatrixSize &matrix_size)
{
 

    // use a larger block size for Fermi and above
    int block_size = 32;

    // set seed for rand()
    srand(1550);

    // allocate host memory for matrices A and B
    unsigned int size_A = matrix_size.WA * matrix_size.HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = matrix_size.WB * matrix_size.HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // set seed for rand()
    srand(1550);

    // initialize host memory
    random_mat_fill(h_A, size_A);
    random_mat_fill(h_B, size_B);

    // allocate device memory
    float *d_A, *d_B, *d_C;
    unsigned int size_C = matrix_size.WC * matrix_size.HC;
    unsigned int mem_size_C = sizeof(float) * size_C;

    // allocate host memory for the result
    float *h_C      = (float *) malloc(mem_size_C);
    float *h_CUBLAS = (float *) malloc(mem_size_C);

   cudaMalloc((void **) &d_A, mem_size_A);
    cudaMalloc((void **) &d_B, mem_size_B);
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_C, mem_size_C);

    // setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(matrix_size.WC / threads.x, matrix_size.HC / threads.y);

    // create and start timer
    printf("Computing result using CUBLAS...");

    // execute the kernel
    int nIter = 30;

    // CUBLAS version 2.0
    {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        cublasHandle_t handle;
        cudaEvent_t start, stop;

        cublasCreate(&handle);

        //Perform warmup operation with cublas
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.WB, matrix_size.HA, matrix_size.WA, &alpha, d_B, matrix_size.WB, d_A, matrix_size.WA, &beta, d_C, matrix_size.WB);

        // Allocate CUDA events that we'll use for timing
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Record the start event
        cudaEventRecord(start, NULL);

        for (int j = 0; j < nIter; j++)
        {
            //note cublas is column primary!
            //need to transpose the order
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.WB, matrix_size.HA, matrix_size.WA, &alpha, d_B, matrix_size.WB, d_A, matrix_size.WA, &beta, d_C, matrix_size.WB);

        }

        printf("done.\n");

        // Record the stop event
      cudaEventRecord(stop, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stop);

        float msecTotal = 0.0f;
        cudaEventElapsedTime(&msecTotal, start, stop);

        // Compute and print the performance
        float msecPerMatrixMul = msecTotal / nIter;
        double flopsPerMatrixMul = 2.0 * (double)matrix_size.HC * (double)matrix_size.WC * (double)matrix_size.HB;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
        printf(
            "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
            gigaFlops,
            msecPerMatrixMul,
            flopsPerMatrixMul);

        // copy result from device to host
        cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost);

        // Destroy the handle
 cublasDestroy(handle);       
    }

    // compute reference solution
    printf("Computing result using host CPU...");
   cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1, NULL);
    float *reference = (float *)malloc(mem_size_C);
    matrix_Mul_CPU(reference, h_A, h_B, matrix_size.HA, matrix_size.WA, matrix_size.WB);
    printf("done.\n");
   cudaEventRecord(stop1, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stop1);
  float msecTotal1 = 0.0f;
        cudaEventElapsedTime(&msecTotal1, start1, stop1);
 float msecPerMatrixMul1 = msecTotal1;
 printf("Time= %.3f msec\n",msecPerMatrixMul1);
    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(reference);
    cudaFree(d_A);
  cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();

   
}

int main(int argc, char **argv)
{
    printf("[Matrix Multiply CUBLAS] - Starting...\n");

    int devID = 0, sizeMult = 5;
    sMatrixSize matrix_size;

    initializeCUDA(argc, argv, devID, sizeMult, matrix_size);

     matrix_Multiply(argc, argv, devID, matrix_size);

    return 0;
}