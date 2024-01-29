#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call) {                                              \
    const cudaError_t error = call;                                \
    if (error != cudaSuccess) {                                    \
    printf( "Error: %s:%d, ", __FILE__, __LINE__);         \
    printf( "code:%d, reason: %s\n", error,                \
    cudaGetErrorString(error));                                    \
    exit(1);                                                       \
    }                                                              \
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0e-8;
    bool match = 1;
    for (int i = 0; i < N; ++i) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu: %5.2f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");

}


void initialDate(float *ip, const int size) {
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; ++i) {
        ip[i] = (float)(rand() % 0xFF)/10.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main(int argc, char **argv) {
    printf("> %s Starting...\n", argv[0]);

    //set up device
    int dev = 0;
    cudaSetDevice(dev);

    // int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev) );
    printf("Device name: %s\n", deviceProp.name);
    // CHECK(cudaSetDevice(dev));

    //set up data size of arrays
    int nElem = 32;
    printf("Size of data %d\n", nElem);
    //malloc host memory
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    initialDate(h_A, nElem);
    initialDate(h_B, nElem);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    //malloc device global memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    //tranfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    //invoke kernel at host side
    dim3 block(nElem);
    dim3 grid((nElem + block.x - 1) / block.x);

    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    printf("> Executed kernel <<<%d, %d>>>\n", grid.x, block.x);

    //copy back data from device global memory to host memory
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    //check device results against reference values
    checkResult(hostRef, gpuRef, nElem);

    //free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    //free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;

}