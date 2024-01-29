// check for errors
#define CHECK(call) {                                              \
    const cudaError_t error = call;                                \
    if (error != cudaSuccess) {                                    \
    printf( "Error: %s:%d, ", __FILE__, __LINE__);         \
    printf( "code:%d, reason: %s\n", error,                \
    cudaGetErrorString(error));                                    \
    exit(1);                                                       \
    }                                                              \
}

// check for GPU informations
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev) );
    printf("Device %d name: \"%s\"\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

//time
#include <sys/time.h>
double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec/1000000;
}