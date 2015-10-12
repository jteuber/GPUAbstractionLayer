#ifndef DEVMEM_H
#define DEVMEM_H

#include <assert.h>

#include "Cuda_.h"
#include "OpenCL.h"

class GAL_EXPORT GPUUser
{
	friend class GPUWrapper;

protected:
	static Cuda* m_pCuda;
	static OpenCL* m_pOpenCL;
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
	cl_mem	getOCL() const;

	~DevMem();

private:
	DevMem( T* pCUDAMem, unsigned int uiSize );
	DevMem( cl_mem openCLMem, unsigned int uiSize );

	bool isValid() const;

private:
	T*		m_pCUDAMem;
	cl_mem	m_openCLMem;

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
	if( m_pCUDAMem != NULL )
	{
		m_pCuda->devMemSet( m_pCUDAMem, m_uiSize, data );
	}
	else if( m_openCLMem != NULL )
	{
		m_pOpenCL->devMemSet<T>( m_openCLMem, m_uiSize, data );
	}
}

template<typename T>
void DevMem<T>::copyToHost( T* hostData )
{
	if( m_pCUDAMem != NULL )
	{
		m_pCuda->copyFromDev( hostData, m_pCUDAMem, m_uiSize );
	}
	else if( m_openCLMem != NULL )
	{
		m_pOpenCL->copyFromDev( hostData, m_openCLMem, m_uiSize );
	}
}

template<typename T>
void DevMem<T>::copyToHost(T* hostData, unsigned int uiNumElementsToCopy, unsigned int uiOffset)
{
	if( m_pCUDAMem != NULL )
	{
		m_pCuda->copyFromDev( hostData, m_pCUDAMem+uiOffset, uiNumElementsToCopy );
	}
	else if( m_openCLMem != NULL )
	{
		m_pOpenCL->copyFromDev( hostData, m_openCLMem, uiNumElementsToCopy, uiOffset );
	}
}

template<typename T>
T* DevMem<T>::copyToHost()
{
	if( m_pCUDAMem != NULL || m_openCLMem != NULL )
	{
		T* pTemp = new T[m_uiSize];
		copyToHost( pTemp );

		return pTemp;
	}

	return NULL;
}

template<typename T>
void DevMem<T>::overwrite(T* hostData)
{
	if( m_pCUDAMem != NULL )
	{
		m_pCuda->copyToDev( m_pCUDAMem, hostData, m_uiSize );
	}
	else if( m_openCLMem != NULL )
	{
		m_pOpenCL->copyToDev( m_openCLMem, hostData, m_uiSize );
	}
}

template<typename T>
void DevMem<T>::overwrite(DevMem<T>* dSrc)
{
	assert( dSrc->m_uiSize <= m_uiSize );

	if( m_pCUDAMem != NULL )
	{
		m_pCuda->copyDevToDev( m_pCUDAMem, dSrc->m_pCUDAMem, dSrc->m_uiSize );
	}
	else if( m_openCLMem != NULL )
	{
		m_pOpenCL->copyDevToDev<T>( m_openCLMem, dSrc->m_openCLMem, dSrc->m_uiSize );
	}
}

template<typename T>
void DevMem<T>::copyToOffset(DevMem<T>* dSrc, unsigned int uiOffset, unsigned int uiNumElements)
{
	if( m_pCUDAMem != NULL )
	{
		m_pCuda->copyDevToDev( m_pCUDAMem, dSrc->m_pCUDAMem, uiNumElements, uiOffset );
	}
	else if( m_openCLMem != NULL )
	{
		m_pOpenCL->copyDevToDev<T>( m_openCLMem, dSrc->m_openCLMem, uiNumElements, uiOffset );
	}
}


template<typename T>
unsigned int DevMem<T>::getSize() const
{
	return m_uiSize;
}

template<typename T>
T* DevMem<T>::getCUDA() const
{
	return m_pCUDAMem;
}

template<typename T>
cl_mem DevMem<T>::getOCL() const
{
	return m_openCLMem;
}


template<typename T>
DevMem<T>::DevMem( T* pCUDAMem, unsigned int uiSize )
	: m_pCUDAMem( pCUDAMem )
	, m_openCLMem( NULL )
	, m_uiSize( uiSize )
{

}

template<typename T>
DevMem<T>::DevMem( cl_mem openCLMem, unsigned int uiSize )
	: m_pCUDAMem( NULL )
	, m_openCLMem( openCLMem )
	, m_uiSize( uiSize )
{

}

template<typename T>
DevMem<T>::~DevMem()
{
	if( m_pCUDAMem != NULL )
	{
		m_pCuda->devDelete( m_pCUDAMem );
		m_pCUDAMem = NULL;
	}
	else if( m_openCLMem != NULL )
	{
		m_pOpenCL->devDelete( m_openCLMem );
		m_openCLMem = NULL;
	}
}


template<typename T>
bool DevMem<T>::isValid() const
{
	return m_pCUDAMem != NULL || m_openCLMem != NULL;
}



#endif // GPUMEMORY_H
