#include<stdio.h>

__global__ void hello_world(void)
{
    printf("Hello world from GPU\n");
}

int main(void)
{
    printf("Hello world from CPU\n");

    hello_world<<<1, 10>>>();

    cudaDeviceReset();
    // cudaDeviceSynchronize();    
    return 0;

}