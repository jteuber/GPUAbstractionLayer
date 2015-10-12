
#include "GPUWrapper.h"

#include "ArgumentTools.h"


GPUWrapper* GPUWrapper::sm_pInstance = NULL;


bool GPUWrapper::init(int argc, const char** argv)
{
	if( checkArgumentExists(argc, argv, "-gpgpu") )
	{
		std::string strType = getArgumentString( argc, argv, "-gpgpu" );
		if( strType == "opencl" )
			return init( GPOpenCL, argc, argv );
	}

	return init( GPCuda, argc, argv );
}


bool GPUWrapper::init(EGPGPUType eGPGPUType, int argc, const char** argv)
{
	sm_pInstance = new GPUWrapper();

	if( eGPGPUType == GPCuda )
	{
		sm_pInstance->m_pCuda = new Cuda();
		GPUUser::m_pCuda = sm_pInstance->m_pCuda;
		return sm_pInstance->m_pCuda->init( argc, argv );
	}
	else // GPOpenCL
	{
		sm_pInstance->m_pOpenCL = new OpenCL();
		GPUUser::m_pOpenCL = sm_pInstance->m_pOpenCL;
		return sm_pInstance->m_pOpenCL->init( argc, argv );
	}

	return false;
}

bool GPUWrapper::init( EGPGPUType eGPGPUType, unsigned int uiDeviceID, unsigned int uiPlatformID )
{
	sm_pInstance = new GPUWrapper();

	if( eGPGPUType == GPCuda )
	{
		sm_pInstance->m_pCuda = new Cuda();
		GPUUser::m_pCuda = sm_pInstance->m_pCuda;
		return sm_pInstance->m_pCuda->init( uiDeviceID );
	}
	else // GPOpenCL
	{
		sm_pInstance->m_pOpenCL = new OpenCL();
		GPUUser::m_pOpenCL = sm_pInstance->m_pOpenCL;
		return sm_pInstance->m_pOpenCL->init( uiPlatformID, uiDeviceID );
	}

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

	if( m_pCuda != NULL )
	{
		m_pCuda->scan( dOut->m_pCUDAMem, dIn->m_pCUDAMem, dIn->m_uiSize );
	}
	else if( m_pOpenCL != NULL )
	{
		m_pOpenCL->scan( dOut->m_openCLMem, dIn->m_openCLMem, dIn->m_uiSize );
	}
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
	if( m_pCuda != NULL )
	{
		pRet = new DevMem<unsigned int>( m_pCuda->devNew<unsigned int>( dIn->m_uiSize ), dIn->m_uiSize );
	}
	else if( m_pOpenCL != NULL )
	{
		pRet = new DevMem<unsigned int>( m_pOpenCL->devNew<unsigned int>( dIn->m_uiSize ), dIn->m_uiSize );
	}
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
	if( m_pCuda != NULL )
	{
		return m_pCuda->reduce( dIn->m_pCUDAMem, dIn->m_uiSize );
	}
	else if( m_pOpenCL != NULL )
	{
		return m_pOpenCL->reduce( dIn->m_openCLMem, dIn->m_uiSize );
	}

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
	if( m_pCuda != NULL )
	{
		m_pCuda->sort( dKeys->m_pCUDAMem, dValues->m_pCUDAMem, ( numElements > 0 ) ? numElements : dKeys->m_uiSize );
	}
	else if( m_pOpenCL != NULL )
	{
		m_pOpenCL->sortUIntUInt( dKeys->m_openCLMem, dValues->m_openCLMem, ( numElements > 0 ) ? numElements : dKeys->m_uiSize );
	}
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
	if( m_pCuda != NULL )
	{
		m_pCuda->sort( dKeys->m_pCUDAMem, dValues->m_pCUDAMem, ( numElements > 0 ) ? numElements : dKeys->m_uiSize );
	}
	else if( m_pOpenCL != NULL )
	{
		m_pOpenCL->sortFloatUInt( dKeys->m_openCLMem, dValues->m_openCLMem, ( numElements > 0 ) ? numElements : dKeys->m_uiSize );
	}
}


unsigned int GPUWrapper::getTotalAvailableVRAM()
{
	if( m_pCuda != NULL )
	{
		return m_pCuda->getTotalAvailableVRAM();
	}
	else if( m_pOpenCL != NULL )
	{
		return m_pOpenCL->getTotalAvailableVRAM();
	}

	return 0;
}


GPUWrapper::GPUWrapper()
	: m_pCuda( NULL )
	, m_pOpenCL( NULL )
{

}


GPUWrapper::~GPUWrapper()
{
	if( m_pCuda != NULL )
		delete m_pCuda;
	else if( m_pOpenCL != NULL )
		delete m_pOpenCL;
}

EGPGPUType GPUWrapper::getType() const
{
	if( m_pCuda != NULL )
		return GPCuda;
	else if( m_pOpenCL != NULL )
		return GPOpenCL;

	return GPNone;
}
