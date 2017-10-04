#ifndef DEVMEM_H
#define DEVMEM_H

#include <assert.h>

#include "Cuda_.h"
#include "OpenCL.h"

class GAL_EXPORT GPUUser
{
	friend class GPUWrapper;

protected:
#ifdef USE_CUDA
	static Cuda* m_pCuda;
#endif // USE_CUDA
#ifdef USE_OPENCL
	static OpenCL* m_pOpenCL;
#endif // USE_OPENCL
};


/**
 * @brief Simple wrapper class for the different memory models of CUDA and OpenCL.
 */
template<typename T>
class DevMem : public GPUUser
{
	friend class GPUWrapper;

public:
	void memSet( int data = 0 );

	void copyToHost( T* hostData );
	void copyToHost( T* hostData, unsigned int uiNumElementsToCopy, unsigned int uiOffset = 0 );
	T* copyToHost();

	void overwrite( T* hostData );
	void overwrite( DevMem<T>* dSrc );

	void copyToOffset( DevMem<T>* dSrc, unsigned int uiOffset, unsigned int uiNumElements );


	unsigned int getSize() const;

	T*		getCUDA() const;
#ifdef USE_OPENCL
	cl_mem	getOCL() const;
#endif // USE_OPENCL
	~DevMem();

private:
	DevMem( T* pCUDAMem, unsigned int uiSize );
#ifdef USE_OPENCL
	DevMem( cl_mem openCLMem, unsigned int uiSize );
#endif // USE_OPENCL

	bool isValid() const;

private:
	T*		m_pCUDAMem;
#ifdef USE_OPENCL
	cl_mem	m_openCLMem;
#endif // USE_OPENCL

	unsigned int m_uiSize;
};


struct ObjectOnDevice
{
	DevMem<float4>* m_dVertices;
	DevMem<uint4>* m_dVertexIndices;

	unsigned int m_uiNumVertices;
	unsigned int m_uiNumTriangles;
};


template<typename T>
void DevMem<T>::memSet( int data )
{
#ifdef USE_CUDA
	if( m_pCUDAMem != NULL )
	{
		m_pCuda->devMemSet( m_pCUDAMem, m_uiSize, data );
	}
#ifdef USE_OPENCL
	else
#endif // USE_OPENCL
#endif // USE_CUDA
#ifdef USE_OPENCL
	if( m_openCLMem != NULL )
	{
		m_pOpenCL->devMemSet<T>( m_openCLMem, m_uiSize, data );
	}
#endif // USE_OPENCL
}

template<typename T>
void DevMem<T>::copyToHost( T* hostData )
{
#ifdef USE_CUDA
	if ( m_pCUDAMem != NULL )
	{
		m_pCuda->copyFromDev( hostData, m_pCUDAMem, m_uiSize );
	}
#ifdef USE_OPENCL
	else
#endif // USE_OPENCL
#endif // USE_CUDA
#ifdef USE_OPENCL
	if ( m_openCLMem != NULL )
	{
		m_pOpenCL->copyFromDev( hostData, m_openCLMem, m_uiSize );
	}
#endif // USE_OPENCL
}

template<typename T>
void DevMem<T>::copyToHost(T* hostData, unsigned int uiNumElementsToCopy, unsigned int uiOffset)
{
#ifdef USE_CUDA
	if ( m_pCUDAMem != NULL )
	{
		m_pCuda->copyFromDev( hostData, m_pCUDAMem + uiOffset, uiNumElementsToCopy );
	}
#ifdef USE_OPENCL
	else
#endif // USE_OPENCL
#endif // USE_CUDA
#ifdef USE_OPENCL
	if ( m_openCLMem != NULL )
	{
		m_pOpenCL->copyFromDev( hostData, m_openCLMem, uiNumElementsToCopy, uiOffset );
	}
#endif // USE_OPENCL
}

template<typename T>
T* DevMem<T>::copyToHost()
{
#if !defined(USE_CUDA) && !defined(USE_OPENCL)
	return 0;
#else
	if (
#ifdef USE_CUDA
	    m_pCUDAMem != NULL
#ifdef USE_OPENCL
	    ||
#endif // USE_OPENCL
#endif // USE_CUDA
#ifdef USE_OPENCL
	    m_openCLMem != NULL
#endif // USE_OPENCL
	)
	{
		T* pTemp = new T[m_uiSize];
		copyToHost( pTemp );

		return pTemp;
	}

	return NULL;
#endif // !USE_CUDA && !USE_OPENCL
}

template<typename T>
void DevMem<T>::overwrite(T* hostData)
{
#ifdef USE_CUDA
	if( m_pCUDAMem != NULL )
	{
		m_pCuda->copyToDev( m_pCUDAMem, hostData, m_uiSize );
	}
#ifdef USE_OPENCL
	else
#endif // USE_OPENCL
#endif // USE_CUDA
#ifdef USE_OPENCL
	if( m_openCLMem != NULL )
	{
		m_pOpenCL->copyToDev( m_openCLMem, hostData, m_uiSize );
	}
#endif // USE_OPENCL
}

template<typename T>
void DevMem<T>::overwrite(DevMem<T>* dSrc)
{
	assert( dSrc->m_uiSize <= m_uiSize );

#ifdef USE_CUDA
	if ( m_pCUDAMem != NULL )
	{
		m_pCuda->copyDevToDev( m_pCUDAMem, dSrc->m_pCUDAMem, dSrc->m_uiSize );
	}
#ifdef USE_OPENCL
	else
#endif // USE_OPENCL
#endif // USE_CUDA
#ifdef USE_OPENCL
	if ( m_openCLMem != NULL )
	{
		m_pOpenCL->copyDevToDev<T>( m_openCLMem, dSrc->m_openCLMem, dSrc->m_uiSize );
	}
#endif // USE_OPENCL
}

template<typename T>
void DevMem<T>::copyToOffset(DevMem<T>* dSrc, unsigned int uiOffset, unsigned int uiNumElements)
{
#ifdef USE_CUDA
	if ( m_pCUDAMem != NULL )
	{
		m_pCuda->copyDevToDev( m_pCUDAMem, dSrc->m_pCUDAMem, uiNumElements, uiOffset );
	}
#ifdef USE_OPENCL
	else
#endif // USE_OPENCL
#endif // USE_CUDA
#ifdef USE_OPENCL
	if ( m_openCLMem != NULL )
	{
		m_pOpenCL->copyDevToDev<T>( m_openCLMem, dSrc->m_openCLMem, uiNumElements, uiOffset );
	}
#endif // USE_OPENCL
}


template<typename T>
unsigned int DevMem<T>::getSize() const
{
	return m_uiSize;
}

template<typename T>
T* DevMem<T>::getCUDA() const
{
#ifdef USE_CUDA
	return m_pCUDAMem;
#else
	return NULL;
#endif // USE_CUDA
}

#ifdef USE_OPENCL
template<typename T>
cl_mem DevMem<T>::getOCL() const
{
	return m_openCLMem;
}
#endif // USE_OPENCL


template<typename T>
DevMem<T>::DevMem( T* pCUDAMem, unsigned int uiSize )
    : m_uiSize( uiSize )
#ifdef USE_CUDA
    , m_pCUDAMem( pCUDAMem )
#endif // USE_CUDA
#ifdef USE_OPENCL
    , m_openCLMem( NULL )
#endif // USE_OPENCL
{

}

#ifdef USE_OPENCL
template<typename T>
DevMem<T>::DevMem( cl_mem openCLMem, unsigned int uiSize )
    : m_uiSize( uiSize )
#ifdef USE_CUDA
    , m_pCUDAMem( NULL )
#endif // USE_CUDA
    , m_openCLMem( openCLMem )
{

}
#endif // USE_OPENCL

template<typename T>
DevMem<T>::~DevMem()
{
#ifdef USE_CUDA
	if( m_pCUDAMem != NULL )
	{
		m_pCuda->devDelete( m_pCUDAMem, m_uiSize * sizeof(T) );
		m_pCUDAMem = NULL;
	}
#ifdef USE_OPENCL
	else
#endif // USE_OPENCL
#endif // USE_CUDA
#ifdef USE_OPENCL
	if( m_openCLMem != NULL )
	{
		m_pOpenCL->devDelete( m_openCLMem, m_uiSize * sizeof(T) );
		m_openCLMem = NULL;
	}
#endif // USE_OPENCL
}


template<typename T>
bool DevMem<T>::isValid() const
{
#if !defined(USE_CUDA) && !defined(USE_OPENCL)
	return false;
#else
	return
#ifdef USE_CUDA
	m_pCUDAMem != NULL
#ifdef USE_OPENCL
	||
#endif // USE_OPENCL
#endif // USE_CUDA
#ifdef USE_OPENCL
	m_openCLMem != NULL
#endif // USE_OPENCL
	;
#endif // !USE_CUDA && !USE_OPENCL
}

#endif // GPUMEMORY_H
