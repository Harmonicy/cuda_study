#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
// #include <windows.h>

#include<unistd.h>

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec * 1000 + (double)tp.tv_usec/1000;
}

int recursiveReduce(int *idata, int const size){
    // terminate check
    if (size == 1) return idata[0];

    //renew the stride
    int const stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; ++i){
        idata[i] += idata[stride + i];
    }
    
    //call recursively
    return recursiveReduce(idata, stride);
}

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n){
    //set thread ID
    unsigned int tid = threadIdx.x;

    //convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x*blockDim.x;

    //boundary check
    if(tid >= n) return;

    //in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2){
        if((tid % (2*stride)) == 0){
            idata[tid] += idata[tid + stride];
        }
        //synchronize within threadblock
        __syncthreads();
    }
    //write result for this block to global mem
    if(tid == 0)
        g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n){
    //set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    //convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x*blockDim.x;

    //boundary check
    if(idx >= n) return;

    //in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2){
        int index = 2*stride*tid;

        if(index < blockDim.x){
            idata[index] += idata[index + stride];
        }
        //synchronize within threadblock
        __syncthreads();
    }
    //write result for this block to global mem
    if(tid == 0)
        g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    //convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    // boundary check
    if(idx >=n) return;

    //in-place reduction in global memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(tid < stride){
            idata[tid] += idata[tid + stride];
        }
        //synchronize within threadblock
        __syncthreads();
    }

    //write result for this block to global mem
    if(tid == 0)
        g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n){
    //set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x*blockDim.x*2 + threadIdx.x;

    //convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x*blockDim.x*2;

    //unrolling 2
    if(idx + blockDim.x < n)
        g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads();

    //in-place reduction in global memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(tid < stride){
            idata[tid] += idata[tid + stride];
        }
        //synchronize within threadblock
        __syncthreads();
    }

    //write result for this block to global mem
    if(tid == 0)
        g_odata[blockIdx.x] = idata[0];
}


int main(int argc, char *argv[]) {
    //set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    bool bResult = false;
    
    
    //initialization
    int size = 1<<24;
    printf("Matrix size: %d\n", size);

    //execution configuration
    int blocksize = 512;
    if(argc > 1){
        blocksize = atoi(argv[1]);
    }

    dim3 block (blocksize, 1);
    dim3 grid ((size + block.x - 1) / block.x, 1);
    printf("grid: %d, block: %d\n", grid.x, block.x);

    //allocate memory on host
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x*sizeof(int));
    int *tmp = (int *) malloc(bytes);

    //initialize the array
    for(int i = 0; i < size; ++i){
        //maks off high 2 bytes to force max number to 255
        h_idata[i] = (int) (rand() & 0xFF);
    }
    memcpy(tmp, h_idata, bytes);

    size_t iStart, iElaps;
    int gpu_sum=0;

    //allocate memory on GPU
    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void **) &d_idata, bytes);
    cudaMalloc((void **) &d_odata, grid.x*sizeof(int));

    //cpu reduction
    iStart = cpuSecond();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = cpuSecond() - iStart;
    // iElaps = static_cast<double>(end_time.QuadPart - start_time.QuadPart) / frequency.QuadPart / 1000;
    // iElaps = end_time - start_time;
    
    printf("CPU reduce elapsed %d ms cpu_sum: %d\n", iElaps, cpu_sum);

    //warmup
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i = 0; i < grid.x; ++i) 
        gpu_sum += h_odata[i];
    printf("GPU warmup elapsed %d ms gpu_sum: %d <<<grid %d block %d>>> \n", iElaps, gpu_sum, grid.x, block.x);

    //kernel 1: reduceNeighbored
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i = 0; i < grid.x; ++i) 
        gpu_sum += h_odata[i];
    printf("GPU reduceNeighbored elapsed %d ms gpu_sum: %d <<<grid %d block %d>>> \n", iElaps, gpu_sum, grid.x, block.x);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x/8*sizeof(int), cudaMemcpyDeviceToHost);


    //kernel 2: reduceNeighboredLess
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i = 0; i < grid.x; ++i) 
        gpu_sum += h_odata[i];
    printf("GPU reduceNeighbored2 elapsed %d ms gpu_sum: %d <<<grid %d block %d>>> \n", iElaps, gpu_sum, grid.x, block.x);
    

    //kernel 3: reduceInterleaved
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i = 0; i < grid.x; ++i) 
        gpu_sum += h_odata[i];
    printf("GPU reduceInterleaved elapsed %d ms gpu_sum: %d <<<grid %d block %d>>> \n", iElaps, gpu_sum, grid.x, block.x);
    cudaDeviceSynchronize();

    //kernel 4: reduceUnrolling2
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceUnrolling2<<<grid.x/2, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x/2*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i = 0; i < grid.x/2; ++i) 
        gpu_sum += h_odata[i];
    printf("GPU reduceUnrolling2 elapsed %d ms gpu_sum: %d <<<grid %d block %d>>> \n", iElaps, gpu_sum, grid.x/2, block.x);
    cudaDeviceSynchronize();
    

    // gpu_sum = 0;
    // for(int i = 0; i < grid.x/8; ++i) 
    //     gpu_sum += h_odata[i];

    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);

    cudaDeviceReset();
    //check the reuslt
    bResult = (gpu_sum == cpu_sum);
    if(!bResult)
        printf(">> Test FAILED!\n");
    return EXIT_SUCCESS;
    
    

}
