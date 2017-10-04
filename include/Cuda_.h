#ifndef CUDA_H
#define CUDA_H

#ifdef USE_CUDA

#ifdef linux
#include <execinfo.h>
#include <signal.h>
#endif

#include <cuda_runtime.h>
#include <driver_types.h>
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

	void devDelete(void *devPtr, size_t sizeInBytes);

	template<typename T> static void copyFromDev( T* hostData, const T* deviceData, unsigned int size = 1 );
	template<typename T> static void copyToDev( T* deviceData, const T* hostData, unsigned int size = 1 );
	template<typename T> static void copyDevToDev( T* dest, const T* src, unsigned int size = 1, unsigned int offset = 0 );

	static void scan( unsigned int* dIn, unsigned int* dOut, size_t numElements );
	static unsigned int reduce( unsigned int* dIn, size_t numElements );

	static void sort( unsigned int* dKeys, unsigned int* dValues, size_t numElements );
	static void sort( float* dKeys, unsigned int* dValues, size_t numElements );

	unsigned int getTotalAvailableVRAM();
	int getFreeVRAM();

private:
	Cuda();
	~Cuda();

private:
	cudaDeviceProp m_cudaDeviceProps;
	size_t m_reservedMemory;
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
	if( freeMem < uiCompleteSize + m_reservedMemory )
		Log::getLog("GPUAbstractionLayer").logWarning( "Device will probably run out of memory now..." );
	// for now just go on even if there probably is not enough

	m_reservedMemory += uiCompleteSize;

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

#else

__declspec(align(8)) struct short3_t
{
	short x, y, z;
};
typedef struct short3_t short3;

__declspec(align(8)) struct short4_t
{
	short x, y, z, w;
};
typedef struct short4_t short4;

__declspec(align(16)) struct int3_t
{
	int x, y, z;
};
typedef struct int3_t int3;

typedef unsigned int uint;

__declspec(align(16)) struct uint3_t
{
	unsigned int x, y, z;
};
typedef struct uint3_t uint3;

__declspec(align(16)) struct uint4_t
{
	unsigned int x, y, z, w;
};
typedef struct uint4_t uint4;

__declspec(align(16)) struct float3_t
{
	float x, y, z;
};
typedef struct float3_t float3;

__declspec(align(16)) struct float4_t
{
	float x, y, z, w;
};
typedef struct float4_t float4;

inline float3 operator+(const float3& a, const float3& b)
{
	return float3{ a.x + b.x, a.y + b.y, a.z + b.z };
}

inline float3 operator+=(float3& a, const float b)
{
	a.x += b;
	a.y += b;
	a.z += b;
	return a;
}

inline float3 operator-(const float3& a, const float3& b)
{
	return float3{ a.x - b.x, a.y - b.y, a.z - b.z };
}

inline uint4 operator-(const uint4& a, const uint4& b)
{
	return uint4{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
}

inline float3 operator-=(float3& a, const float b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
	return a;
}

inline float3 operator*(const float3& a, const float& b)
{
	return float3{ a.x * b, a.y * b, a.z * b };
}

inline uint3 operator*=(uint3 &a, const uint& b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	return a;
}

inline float3 operator*=(float3 &a, float3& b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}

inline float3 operator*=(float3 &a, const float& b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	return a;
}

inline uint3 make_uint3( const uint& a )
{
	return uint3{ a, a, a };
}

inline short4 make_short4( const short& x, const short& y, const short& z, const short& w )
{
	return short4{ x, y, z, w };
}

inline uint4 make_uint4( const uint& a )
{
	return uint4{ a, a, a, a };
}

inline uint4 make_uint4( const uint3& a, const uint& b )
{
	return uint4{ a.x, a.y, a.z, b };
}

inline uint4 make_uint4( const uint& x, const uint& y, const uint& z, const uint& w )
{
	return uint4{ x, y, z, w };
}

inline float3 make_float3( const float& a )
{
	return float3{ a, a, a };
}

inline float3 make_float3( const float& x, const float& y, const float& z )
{
	return float3{ x, y, z };
}

inline float3 make_float3( const float4& a )
{
	return float3{ a.x, a.y, a.z };
}

inline float4 make_float4( const float& a )
{
	return float4{ a, a, a, a };
}

inline float4 make_float4( const float3& a, const float& b )
{
	return float4{ a.x, a.y, a.z, b };
}

inline float4 make_float4( const float& x, const float& y, const float& z, const float& w )
{
	return float4{ x, y, z, w };
}

inline float dot( float3 a, float3 b )
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float3 cross( const float3& a, const float3& b )
{
	return float3{ a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
}

#endif // USE_CUDA

#endif // CUDA_H
