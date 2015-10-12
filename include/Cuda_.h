#ifndef CUDA_H
#define CUDA_H

#ifdef linux
#include <execinfo.h>
#include <signal.h>
#endif

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "Log.h"

#include "GPUAbstractionLayer_global.h"

class GAL_EXPORT Cuda
{
	friend class GPUWrapper;

public:
	bool init( int argc, const char **argv );
	bool init( unsigned int uiDevice );

	template<typename T> T* devNew( unsigned int size = 1 );
	template<typename T> static void devMemSet( T* deviceData, unsigned int size = 1, int data = 0 );

	static void devDelete(void *devPtr);

	template<typename T> static void copyFromDev( T* hostData, const T* deviceData, unsigned int size = 1 );
	template<typename T> static void copyToDev( T* deviceData, const T* hostData, unsigned int size = 1 );
	template<typename T> static void copyDevToDev( T* dest, const T* src, unsigned int size = 1, unsigned int offset = 0 );

	static void scan( unsigned int* dIn, unsigned int* dOut, size_t numElements );
	static unsigned int reduce( unsigned int* dIn, size_t numElements );

	static void sort( unsigned int* dKeys, unsigned int* dValues, size_t numElements );
	static void sort( float* dKeys, unsigned int* dValues, size_t numElements );

	unsigned int getTotalAvailableVRAM() const;

private:
	Cuda();
	~Cuda();

private:
	cudaDeviceProp m_cudaDeviceProps;
};


static const char *_pseCudaGetErrorEnum(cudaError_t error)
{
    switch (error)
    {
        case cudaSuccess:
            return "cudaSuccess";

        case cudaErrorMissingConfiguration:
            return "cudaErrorMissingConfiguration";

        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";

        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";

        case cudaErrorLaunchFailure:
            return "cudaErrorLaunchFailure";

        case cudaErrorPriorLaunchFailure:
            return "cudaErrorPriorLaunchFailure";

        case cudaErrorLaunchTimeout:
            return "cudaErrorLaunchTimeout";

        case cudaErrorLaunchOutOfResources:
            return "cudaErrorLaunchOutOfResources";

        case cudaErrorInvalidDeviceFunction:
            return "cudaErrorInvalidDeviceFunction";

        case cudaErrorInvalidConfiguration:
            return "cudaErrorInvalidConfiguration";

        case cudaErrorInvalidDevice:
            return "cudaErrorInvalidDevice";

        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";

        case cudaErrorInvalidPitchValue:
            return "cudaErrorInvalidPitchValue";

        case cudaErrorInvalidSymbol:
            return "cudaErrorInvalidSymbol";

        case cudaErrorMapBufferObjectFailed:
            return "cudaErrorMapBufferObjectFailed";

        case cudaErrorUnmapBufferObjectFailed:
            return "cudaErrorUnmapBufferObjectFailed";

        case cudaErrorInvalidHostPointer:
            return "cudaErrorInvalidHostPointer";

        case cudaErrorInvalidDevicePointer:
            return "cudaErrorInvalidDevicePointer";

        case cudaErrorInvalidTexture:
            return "cudaErrorInvalidTexture";

        case cudaErrorInvalidTextureBinding:
            return "cudaErrorInvalidTextureBinding";

        case cudaErrorInvalidChannelDescriptor:
            return "cudaErrorInvalidChannelDescriptor";

        case cudaErrorInvalidMemcpyDirection:
            return "cudaErrorInvalidMemcpyDirection";

        case cudaErrorAddressOfConstant:
            return "cudaErrorAddressOfConstant";

        case cudaErrorTextureFetchFailed:
            return "cudaErrorTextureFetchFailed";

        case cudaErrorTextureNotBound:
            return "cudaErrorTextureNotBound";

        case cudaErrorSynchronizationError:
            return "cudaErrorSynchronizationError";

        case cudaErrorInvalidFilterSetting:
            return "cudaErrorInvalidFilterSetting";

        case cudaErrorInvalidNormSetting:
            return "cudaErrorInvalidNormSetting";

        case cudaErrorMixedDeviceExecution:
            return "cudaErrorMixedDeviceExecution";

        case cudaErrorCudartUnloading:
            return "cudaErrorCudartUnloading";

        case cudaErrorUnknown:
            return "cudaErrorUnknown";

        case cudaErrorNotYetImplemented:
            return "cudaErrorNotYetImplemented";

        case cudaErrorMemoryValueTooLarge:
            return "cudaErrorMemoryValueTooLarge";

        case cudaErrorInvalidResourceHandle:
            return "cudaErrorInvalidResourceHandle";

        case cudaErrorNotReady:
            return "cudaErrorNotReady";

        case cudaErrorInsufficientDriver:
            return "cudaErrorInsufficientDriver";

        case cudaErrorSetOnActiveProcess:
            return "cudaErrorSetOnActiveProcess";

        case cudaErrorInvalidSurface:
            return "cudaErrorInvalidSurface";

        case cudaErrorNoDevice:
            return "cudaErrorNoDevice";

        case cudaErrorECCUncorrectable:
            return "cudaErrorECCUncorrectable";

        case cudaErrorSharedObjectSymbolNotFound:
            return "cudaErrorSharedObjectSymbolNotFound";

        case cudaErrorSharedObjectInitFailed:
            return "cudaErrorSharedObjectInitFailed";

        case cudaErrorUnsupportedLimit:
            return "cudaErrorUnsupportedLimit";

        case cudaErrorDuplicateVariableName:
            return "cudaErrorDuplicateVariableName";

        case cudaErrorDuplicateTextureName:
            return "cudaErrorDuplicateTextureName";

        case cudaErrorDuplicateSurfaceName:
            return "cudaErrorDuplicateSurfaceName";

        case cudaErrorDevicesUnavailable:
            return "cudaErrorDevicesUnavailable";

        case cudaErrorInvalidKernelImage:
            return "cudaErrorInvalidKernelImage";

        case cudaErrorNoKernelImageForDevice:
            return "cudaErrorNoKernelImageForDevice";

        case cudaErrorIncompatibleDriverContext:
            return "cudaErrorIncompatibleDriverContext";

        case cudaErrorPeerAccessAlreadyEnabled:
            return "cudaErrorPeerAccessAlreadyEnabled";

        case cudaErrorPeerAccessNotEnabled:
            return "cudaErrorPeerAccessNotEnabled";

        case cudaErrorDeviceAlreadyInUse:
            return "cudaErrorDeviceAlreadyInUse";

        case cudaErrorProfilerDisabled:
            return "cudaErrorProfilerDisabled";

        case cudaErrorProfilerNotInitialized:
            return "cudaErrorProfilerNotInitialized";

        case cudaErrorProfilerAlreadyStarted:
            return "cudaErrorProfilerAlreadyStarted";

        case cudaErrorProfilerAlreadyStopped:
            return "cudaErrorProfilerAlreadyStopped";

#if __CUDA_API_VERSION >= 0x4000

        case cudaErrorAssert:
            return "cudaErrorAssert";

        case cudaErrorTooManyPeers:
            return "cudaErrorTooManyPeers";

        case cudaErrorHostMemoryAlreadyRegistered:
            return "cudaErrorHostMemoryAlreadyRegistered";

        case cudaErrorHostMemoryNotRegistered:
            return "cudaErrorHostMemoryNotRegistered";
#endif

        case cudaErrorStartupFailure:
            return "cudaErrorStartupFailure";

        case cudaErrorApiFailureBase:
            return "cudaErrorApiFailureBase";
    }

    return "<unknown>";
}


template< typename T >
bool pseCudaCheck(T result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		Log::getLog("GPUAbstractionLayer") << Log::EL_ERROR << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result) << "(" << _pseCudaGetErrorEnum(result) << ") \"" << func << "\"" << Log::endl;
		return true;
	}
	else
		return false;
}

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define pseCheckCudaErrors(val)           pseCudaCheck ( (val), #val, __FILE__, __LINE__ )



template<typename T>
T* Cuda::devNew(unsigned int size)
{
	T* pTemp;
	unsigned int uiCompleteSize = sizeof(T) * size;

	// check whether there is enough memory available
	size_t freeMem, totalMem;
	pseCheckCudaErrors( cudaMemGetInfo( &freeMem, &totalMem ) );
	if( freeMem < uiCompleteSize )
		Log::getLog("GPUAbstractionLayer").logWarning( "Device will probably run out of memory now..." );
	// for now just go on even if there probably is not enough

	// allocate the memory
	if( !pseCheckCudaErrors(cudaMalloc((void**)&pTemp, uiCompleteSize)) )
		return pTemp;


	Log::getLog("GPUAbstractionLayer").logError( "Memory allocation failed!" );

#if defined(linux) && defined(DEBUG)
	void *array[10];
	size_t traceSize = backtrace(array, 10);

	Log::getLog("GPUAbstractionLayer").logError( "Stacktrace:" );

	// print out all the frames to stderr
	char** pcBacktrace = backtrace_symbols(array, traceSize);
	for( unsigned int i = 0; i < traceSize; ++i )
	{
		Log::getLog("GPUAbstractionLayer").logError( pcBacktrace[i] );
	}
#endif

	return NULL;
}

template<typename T>
void Cuda::devMemSet(T* deviceData, unsigned int size, int data)
{
	pseCheckCudaErrors(cudaMemset( deviceData, data, size * sizeof(T) ));
}

template<typename T>
void Cuda::copyFromDev(T* hostData, const T* deviceData, unsigned int size)
{
	pseCheckCudaErrors( cudaMemcpy(hostData, deviceData, sizeof(T) * size, cudaMemcpyDeviceToHost) );
}

template<typename T>
void Cuda::copyToDev(T* deviceData, const T* hostData, unsigned int size)
{
	pseCheckCudaErrors( cudaMemcpy((char *) deviceData, hostData, sizeof(T) * size, cudaMemcpyHostToDevice) );
}

template<typename T>
void Cuda::copyDevToDev(T* dest, const T* src, unsigned int size, unsigned int offset)
{
	pseCheckCudaErrors( cudaMemcpy( (char *) dest + ( sizeof(T) * offset ), src, sizeof(T) * size, cudaMemcpyDeviceToDevice ) );
}


#endif // CUDA_H
