#ifndef IGPGPU_H
#define IGPGPU_H

#include "GPUAbstractionLayer_global.h"

#include "Cuda_.h"
#include "OpenCL.h"

#include "DevMem.h"

enum EGPGPUType
{
	GPCuda = 0,
	GPOpenCL,
	GPNone
};

class GAL_EXPORT GPUWrapper
{
public:
	static GPUWrapper* getSingletonPtr()
	{
		if( sm_pInstance == NULL )
			Log::getLog("GPUAbstractionLayer").logFatalError("GPGPU singleton not initialized! Call one of the static init() methods first.");

		return sm_pInstance;
	}


	static bool init( int argc, const char **argv );
	static bool init( EGPGPUType eGPGPUType, int argc, const char **argv );
	static bool init( EGPGPUType eGPGPUType = GPCuda, unsigned int uiDeviceID = 0, unsigned int uiPlatformID = 0 );

	template<typename T> DevMem<T>* devNew( unsigned int size = 1 );
	template<typename T> DevMem<T>* devNew( unsigned int size, int data );

	template<typename T> DevMem<T>* copyToDev( const T* hostData, unsigned int uiNrOfElements = 1 );

	void scan( DevMem<unsigned int>* dIn, DevMem<unsigned int>* dOut );
	DevMem<unsigned int>* scan( DevMem<unsigned int>* dIn );

	unsigned int reduce( DevMem<unsigned int>* dIn );

	void sort( DevMem<unsigned int>* dKeys, DevMem<unsigned int>* dValues, size_t numElements = 0 );
	void sort( DevMem<float>* dKeys, DevMem<unsigned int>* dValues, size_t numElements = 0 );


	unsigned int getTotalAvailableVRAM();

	EGPGPUType getType() const;
	Cuda* getRawCUDA() { return m_pCuda; }
	OpenCL* getRawOpenCL() { return m_pOpenCL; }

private:
	static GPUWrapper* sm_pInstance;

	Cuda* m_pCuda;
	OpenCL* m_pOpenCL;

private:
	GPUWrapper();
	virtual ~GPUWrapper();
};


template<typename T>
DevMem<T>* GPUWrapper::devNew( unsigned int size )
{
	if( size == 0 )
		return NULL;

	DevMem<T>* pTemp = NULL;
	if( m_pCuda != NULL )
	{
		pTemp = new DevMem<T>( m_pCuda->devNew<T>( size ), size );
	}
	else if( m_pOpenCL != NULL )
	{
		pTemp = new DevMem<T>( m_pOpenCL->devNew<T>( size ), size );
	}

	// make sure that the memory is valid before returning
	if( pTemp != NULL )
	{
		if( pTemp->isValid() )
			return pTemp;
		else // if not, delete the invalid object
			delete pTemp;
	}

	return NULL;
}

template<typename T>
DevMem<T>* GPUWrapper::devNew(unsigned int size, int data)
{
	DevMem<T>* pTemp = devNew<T>( size );

	if( pTemp != NULL )
	{
		if( m_pCuda != NULL )
		{
			m_pCuda->devMemSet( pTemp->getCUDA(), size, data );
		}
		else if( m_pOpenCL != NULL )
		{
			m_pOpenCL->devMemSet<T>( pTemp->getOCL(), size, data );
		}
	}

	return pTemp;
}


template<typename T>
DevMem<T>* GPUWrapper::copyToDev(const T* hostData, unsigned int uiNrOfElements)
{
	if( m_pCuda != NULL )
	{
		DevMem<T>* pTemp = devNew<T>( uiNrOfElements );
		m_pCuda->copyToDev( pTemp->m_pCUDAMem, hostData, uiNrOfElements );

		return pTemp;
	}
	else if( m_pOpenCL != NULL )
	{
		return new DevMem<T>( m_pOpenCL->copyToDev( hostData, uiNrOfElements ), uiNrOfElements );
	}

	return NULL;
}

#endif // IGPGPU_H
