# GPUAbstractionLayer

This is a first version of the abstraction layer for CUDA and OpenCL as introduced in the Paper "A Framework for Transparent Execution of Massively-Parallel Applications on CUDA and OpenCL" on the EuroVR 2015.
It is the independend version of what we used to port the ProtoSphere sphere-packing-library to OpenCL.

# DEPENDENCIES
CUDA v6.0 or higher
boost (the current official release)

# BUILDING
The GPUAbstractionLayer uses CMake (https://cmake.org/) to provide platform independent build support.

# USAGE
simple example program:

create a .cu file

```
// number of threads in one execution block on the device
const unsigned int threads_per_block = 256;

// CUDA kernel
__global__ void myKernel( int nrOfThreads, float* device_array )
{
    ...
}

// OpenCL kernel
const std::string strOCLKernel =
"__kernel void myKernel( int nrOfThreads, __global float* device_array )\n"
"{\n"
"   ...\n"
"}\n";

// kernel caller function
void myKernelCaller( int nrOfThreads, DevMem<float>* device_array)
{
    GPUWrapper* gpu = GPUWrapper::getSingletonPtr();
    if( gpu->getType() == GPCuda )
    {
        unsigned int numBlocks = (nrOfThreads + threadsPerBlock - 1) / threadsPerBlock;
        myKernel<<< numBlocks, threadsPerBlock >>>( nrOfThreads, device_array->getCUDA());
    }
    else if(gpu->getType() == GPOpenCL )
    {
        static Kernel* spKernel = NULL;
        if( spKernel == NULL )	  spKernel = gpu->getRawOpenCL()->createKernel( strOCLKernel );
        spKernel->execute( nrOfThreads, threadsPerBlock, nrOfThreads, device_array->getOCL() );
    }
}

// main application function
void main()
{
    // initialize with OpenCL as GPGPU method
    GPUWrapper::init( GPOpenCL );
    GPUWrapper* gpu = GPUWrapper::getSingletonPtr();

    // generate random numbers on the host
    float host_array[ 1000000 ];
        ...
    // create a new array on the device and copy the array to the device
    DevMem<float>* device_array = gpu->copyToDev( host_array, 1000000 );
    // sort the array and download it from the device
    gpu->sort( device_array );
    myKernelCaller( 1000000, device_array );
    device_array->copyToHost( host_array );
}
```
