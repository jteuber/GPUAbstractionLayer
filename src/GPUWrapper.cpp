
#include "GPUWrapper.h"

#include "ArgumentTools.h"


GPUWrapper* GPUWrapper::sm_pInstance = NULL;


bool GPUWrapper::init(int argc, const char** argv)
{
#ifdef USE_OPENCL
#ifdef USE_CUDA
	// CUDA as default, OpenCL optional
	if ( checkArgumentExists( argc, argv, "-gpgpu" ) )
	{
		std::string strType = getArgumentString( argc, argv, "-gpgpu" );
		if ( strType == "opencl" )
			return init( GPOpenCL, argc, argv );
	}

	return init( GPCuda, argc, argv ); // CUDA only
#else
	return init( GPOpenCL, argc, argv ); // OpenCL only
#endif // USE_CUDA
#else
#ifdef USE_CUDA
	return init( GPCuda, argc, argv ); // CUDA only
#else
	return init( GPNone, argc, argv ); // None available
#endif // USE_CUDA
#endif // USE_OPENCL
}


bool GPUWrapper::init( EGPGPUType eGPGPUType, int argc, const char** argv )
{
	sm_pInstance = new GPUWrapper();

#ifdef USE_CUDA
	if ( eGPGPUType == GPCuda )
	{
		sm_pInstance->m_pCuda = new Cuda();
		GPUUser::m_pCuda = sm_pInstance->m_pCuda;
		return sm_pInstance->m_pCuda->init( argc, argv );
	}
#ifdef USE_OPENCL
	else
#endif // USE_OPENCL
#endif // USE_CUDA
#ifdef USE_OPENCL
	if ( eGPGPUType == GPOpenCL)
	{
		sm_pInstance->m_pOpenCL = new OpenCL();
		GPUUser::m_pOpenCL = sm_pInstance->m_pOpenCL;
		return sm_pInstance->m_pOpenCL->init( argc, argv );
	}
#endif // USE_OPENCL

	return false;
}

bool GPUWrapper::init( EGPGPUType eGPGPUType, unsigned int uiDeviceID, unsigned int uiPlatformID )
{
	sm_pInstance = new GPUWrapper();

#ifdef USE_CUDA
	if( eGPGPUType == GPCuda )
	{
		sm_pInstance->m_pCuda = new Cuda();
		GPUUser::m_pCuda = sm_pInstance->m_pCuda;
		return sm_pInstance->m_pCuda->init( uiDeviceID );
	}
#ifdef USE_OPENCL
	else
#endif // USE_OPENCL
#endif // USE_CUDA
#ifdef USE_OPENCL
	if ( eGPGPUType == GPOpenCL )
	{
		sm_pInstance->m_pOpenCL = new OpenCL();
		GPUUser::m_pOpenCL = sm_pInstance->m_pOpenCL;
		return sm_pInstance->m_pOpenCL->init( uiPlatformID, uiDeviceID );
	}
#endif // USE_OPENCL

	return false;
}

void GPUWrapper::scan( DevMem<unsigned int>* dIn, DevMem<unsigned int>* dOut )
{
	// argument checks
	if( dIn == NULL || dIn->m_uiSize == 0 || dOut == NULL )
	{
		Log::getLog("GPUAbstractionLayer") << Log::EL_WARNING << "Called scan with invalid argument (input pointer == NULL || array size == 0 || output pointer == NULL)." << Log::endl;
		return;
	}

#ifdef USE_CUDA
	if ( m_pCuda != NULL )
	{
		m_pCuda->scan( dOut->m_pCUDAMem, dIn->m_pCUDAMem, dIn->m_uiSize );
	}
#ifdef USE_OPENCL
	else
#endif // USE_OPENCL
#endif // USE_CUDA
#ifdef USE_OPENCL
	if ( m_pOpenCL != NULL )
	{
		m_pOpenCL->scan( dOut->m_openCLMem, dIn->m_openCLMem, dIn->m_uiSize );
	}
#endif // USE_OPENCL
}

DevMem<unsigned int>* GPUWrapper::scan( DevMem<unsigned int>* dIn )
{
	// argument checks
	if( dIn == NULL || dIn->m_uiSize == 0 )
	{
		Log::getLog("GPUAbstractionLayer") << Log::EL_WARNING << "Called scan with invalid argument (input pointer == NULL || array size == 0)." << Log::endl;
		return NULL;
	}

	DevMem<unsigned int>* pRet = NULL;
#ifdef USE_CUDA
	if ( m_pCuda != NULL )
	{
		pRet = new DevMem<unsigned int>( m_pCuda->devNew<unsigned int>( dIn->m_uiSize ), dIn->m_uiSize );
	}
#ifdef USE_OPENCL
	else
#endif // USE_OPENCL
#endif // USE_CUDA
#ifdef USE_OPENCL
	if ( m_pOpenCL != NULL )
	{
		pRet = new DevMem<unsigned int>( m_pOpenCL->devNew<unsigned int>( dIn->m_uiSize ), dIn->m_uiSize );
	}
#endif // USE_OPENCL
	scan( dIn, pRet );

	return pRet;
}

unsigned int GPUWrapper::reduce( DevMem<unsigned int>* dIn )
{
	// argument checks
	if( dIn == NULL || dIn->m_uiSize == 0 )
	{
		Log::getLog("GPUAbstractionLayer") << Log::EL_WARNING << "Called reduce with invalid argument (input pointer == NULL || array size == 0)." << Log::endl;
		return 0;
	}

	// delegate
#ifdef USE_CUDA
	if( m_pCuda != NULL )
	{
		return m_pCuda->reduce( dIn->m_pCUDAMem, dIn->m_uiSize );
	}
#ifdef USE_OPENCL
	else
#endif // USE_OPENCL
#endif // USE_CUDA
#ifdef USE_OPENCL
	if ( m_pOpenCL != NULL )
	{
		return m_pOpenCL->reduce( dIn->m_openCLMem, dIn->m_uiSize );
	}
#endif // USE_OPENCL

	return 0;
}

void GPUWrapper::sort( DevMem<unsigned int>* dKeys, DevMem<unsigned int>* dValues, size_t numElements )
{
	// argument checks
	if( dKeys == NULL || dKeys->m_uiSize == 0 || dValues == NULL )
	{
		Log::getLog("GPUAbstractionLayer") << Log::EL_WARNING << "Called sort (UInt|UInt) with invalid argument (key pointer == NULL || array size == 0 || value pointer == NULL)." << Log::endl;
		return;
	}

	// delegate
#ifdef USE_CUDA
	if ( m_pCuda != NULL )
	{
		m_pCuda->sort( dKeys->getCUDA(), dValues->getCUDA(), (numElements > 0) ? numElements : dKeys->m_uiSize );
	}
#ifdef USE_OPENCL
	else
#endif // USE_OPENCL
#endif // USE_CUDA
#ifdef USE_OPENCL
	if( m_pOpenCL != NULL )
	{
		m_pOpenCL->sortUIntUInt( dKeys->m_openCLMem, dValues->m_openCLMem, ( numElements > 0 ) ? numElements : dKeys->m_uiSize );
	}
#endif // USE_OPENCL
}

//void GPUWrapper::sort( DevMem<unsigned int>* dKeys, DevMem<CellOnDevice>* dValues, size_t numElements )
//{
//	Profiler::getProfilerPtr()->startSubSection( "UIntCells sort" );
//	// argument checks
//	if( dKeys == NULL || dKeys->m_uiSize == 0 || dValues == NULL )
//	{
//		Log::getLog("GPUAbstractionLayer") << Log::EL_WARNING << "Called sort (UInt|Cell) with invalid argument (key pointer == NULL || array size == 0 || value pointer == NULL)." << Log::endl;
//		return;
//	}
//
//	// delegate
//	if( m_pCuda != NULL )
//	{
//		m_pCuda->sort( dKeys->m_pCUDAMem, dValues->m_pCUDAMem, ( numElements > 0 ) ? numElements : dKeys->m_uiSize );
//	}
//	else if( m_pOpenCL != NULL )
//	{
//		m_pOpenCL->sortUIntCells( dKeys->m_openCLMem, dValues->m_openCLMem, ( numElements > 0 ) ? numElements : dKeys->m_uiSize );
//	}
//	Profiler::getProfilerPtr()->stopSubSection();
//}

void GPUWrapper::sort( DevMem<float>* dKeys, DevMem<unsigned int>* dValues, size_t numElements )
{
	// argument checks
	if( dKeys == NULL || dKeys->m_uiSize == 0 || dValues == NULL )
	{
		Log::getLog("GPUAbstractionLayer") << Log::EL_WARNING << "Called sort (Float|UInt) with invalid argument (key pointer == NULL || array size == 0 || value pointer == NULL)." << Log::endl;
		return;
	}

	// delegate
#ifdef USE_CUDA
	if ( m_pCuda != NULL )
	{
		m_pCuda->sort( dKeys->getCUDA(), dValues->getCUDA(), (numElements > 0) ? numElements : dKeys->m_uiSize );
	}
#ifdef USE_OPENCL
	else
#endif // USE_OPENCL
#endif // USE_CUDA
#ifdef USE_OPENCL
	if ( m_pOpenCL != NULL )
	{
		m_pOpenCL->sortFloatUInt( dKeys->m_openCLMem, dValues->m_openCLMem, ( numElements > 0 ) ? numElements : dKeys->m_uiSize );
	}
#endif // USE_OPENCL
}


unsigned int GPUWrapper::getTotalAvailableVRAM()
{
#ifdef USE_CUDA
	if( m_pCuda != NULL )
	{
		return m_pCuda->getTotalAvailableVRAM();
	}
#ifdef USE_OPENCL
	else
#endif // USE_OPENCL
#endif // USE_CUDA
#ifdef USE_OPENCL
	if( m_pOpenCL != NULL )
	{
		return m_pOpenCL->getTotalAvailableVRAM();
	}
#endif // USE_OPENCL

	return 0;
}

int GPUWrapper::getFreeVRAM()
{
#ifdef USE_CUDA
	if( m_pCuda != NULL )
	{
		return m_pCuda->getFreeVRAM();
	}
#ifdef USE_OPENCL
	else
#endif // USE_OPENCL
#endif // USE_CUDA
#ifdef USE_OPENCL
	if( m_pOpenCL != NULL )
	{
		return m_pOpenCL->getFreeVRAM();
	}
#endif // USE_OPENCL

	return 0;
}


GPUWrapper::GPUWrapper()
#ifdef USE_CUDA
	: m_pCuda( NULL )
#ifdef USE_OPENCL
	, m_pOpenCL( NULL )
#endif // USE_OPENCL
#elif defined(USE_OPENCL)
	: m_pOpenCL( NULL )
#endif // USE_CUDA
{
}


GPUWrapper::~GPUWrapper()
{
#ifdef USE_CUDA
	if ( m_pCuda != NULL )
		delete m_pCuda;
#ifdef USE_OPENCL
	else
#endif // USE_OPENCL
#endif // USE_CUDA
#ifdef USE_OPENCL
	if ( m_pOpenCL != NULL )
		delete m_pOpenCL;
#endif // USE_CUDA
}

EGPGPUType GPUWrapper::getType() const
{
#ifdef USE_CUDA
	if( m_pCuda != NULL )
		return GPCuda;
#ifdef USE_OPENCL
	else
#endif // USE_OPENCL
#endif // USE_CUDA
#ifdef USE_OPENCL
	if ( m_pOpenCL != NULL )
		return GPOpenCL;
#endif // USE_CUDA

	return GPNone;
}
