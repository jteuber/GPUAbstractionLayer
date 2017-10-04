#ifdef USE_CUDA

#include "Cuda_.h"

#include "CudaTools.cuh"
#include "ArgumentTools.h"


/// Class methods

bool Cuda::init(int argc, const char** argv)
{
	unsigned int uiDevice = 0;
	if( checkArgumentExists(argc, argv, "-device") )
		uiDevice = getArgumentInt(argc, argv, "-device");

	return init( uiDevice );
}

bool Cuda::init( unsigned int uiDevice )
{
	//First print the available devices:
	cudaDeviceProp deviceProp;
	int iNumDevices = 0;

	if( pseCheckCudaErrors( cudaGetDeviceCount( &iNumDevices ) ) || iNumDevices == 0 )
	{
		Log::getLog("GPUAbstractionLayer").logError("No CUDA device found!");
		return false;
	}

	for( int current_device = 0; current_device < iNumDevices; current_device++ )
	{
		cudaGetDeviceProperties( &deviceProp, current_device );

		Log::getLog("GPUAbstractionLayer") << Log::EL_INFO << "Device " << current_device << " " << deviceProp.name << Log::endl;
	}

	// following code was taken from helper_cuda.h (findCudaDevice()) and slightly modified
	int devID = 0;

	// If the command-line has a device number specified, use it
	if( uiDevice > 0 )
	{
		devID = uiDevice;

		if( devID < 0 || devID >= iNumDevices )
		{
			Log::getLog("GPUAbstractionLayer").logError("Specified CUDA device does not exist!");
			return false;
		}
		else
		{
			if( pseCheckCudaErrors(cudaGetDeviceProperties(&m_cudaDeviceProps, devID)) )
				return false;

			if( m_cudaDeviceProps.major < 2 )
			{
				Log::getLog("GPUAbstractionLayer").logError("Specified GPU device does not support the necessary CUDA compute model 2.0!");
				return false;
			}

			if( pseCheckCudaErrors(cudaSetDevice(devID)) )
				return false;

			Log::getLog("GPUAbstractionLayer") << Log::EL_INFO << "using GPU Device " << devID << ": \"" << m_cudaDeviceProps.name << "\" with compute capability " << m_cudaDeviceProps.major << "." << m_cudaDeviceProps.minor << Log::endl;
		}
	}
	else
	{
		// Otherwise pick the device with highest Gflops/s
		devID = gpuGetMaxGflopsDeviceId();
		if( pseCheckCudaErrors(cudaSetDevice(devID)) || pseCheckCudaErrors(cudaGetDeviceProperties(&m_cudaDeviceProps, devID)) )
			return false;

		Log::getLog("GPUAbstractionLayer") << Log::EL_INFO << "using GPU Device " << devID << ": \"" << m_cudaDeviceProps.name << "\" with compute capability " << m_cudaDeviceProps.major << "." << m_cudaDeviceProps.minor << Log::endl;
	}

	return true;
}


void Cuda::devDelete(void* devPtr, size_t sizeInBytes)
{
	pseCheckCudaErrors( cudaFree( devPtr ) );
	m_reservedMemory -= sizeInBytes;
}


unsigned int Cuda::getTotalAvailableVRAM()
{
	size_t freeMem, totalMem;
	pseCheckCudaErrors( cudaMemGetInfo( &freeMem, &totalMem ) );
	return totalMem;
}

int Cuda::getFreeVRAM()
{
	size_t freeMem, totalMem;
	pseCheckCudaErrors( cudaMemGetInfo( &freeMem, &totalMem ) );
	return freeMem-m_reservedMemory;
}


void Cuda::scan( unsigned int* dOut, unsigned int* dIn, size_t numElements )
{
	thrustScan( dOut, dIn, numElements );
}

unsigned int Cuda::reduce( unsigned int* dIn, size_t numElements)
{
	return thrustReduce( dIn, numElements );
}


void Cuda::sort( unsigned int* dKeys, unsigned int* dValues, size_t numElements )
{
	sortUIntUInt( dKeys, dValues, numElements );
}

/*void Cuda::sort( unsigned int* dKeys, CellOnDevice* dValues, size_t numElements )
{
	sortUIntCells( dKeys, dValues, numElements );
}*/

void Cuda::sort( float* dKeys, unsigned int* dValues, size_t numElements )
{
	sortFloatUInt( dKeys, dValues, numElements );
}


Cuda::Cuda()
    : m_reservedMemory( 0 )
{
}

Cuda::~Cuda()
{
}

#endif // USE_CUDA
