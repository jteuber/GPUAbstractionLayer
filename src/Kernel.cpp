#ifdef USE_OPENCL

#include "Kernel.h"

#include <algorithm>

bool Kernel::addSharedMem( const unsigned int uiSize )
{
	int iErr = clSetKernelArg( m_kernel, m_uiNumArguments, uiSize, NULL );
	++m_uiNumArguments;

	if(iErr < 0)
	{
		Log::getLog("GPUAbstractionLayer") << Log::EL_ERROR << "Unable to set kernel argument number " << m_uiNumArguments << ". Error code: " << errorNumberToString(iErr) << Log::endl;
		return false;
	}

	return true;
}

bool Kernel::execute( const unsigned int uiTotalNrOfWorkItems, const unsigned int uiNrOfWorkItemsPerUnit )
{
	unsigned int uiTotalNrOfWorkItemsExec = uiTotalNrOfWorkItems;
	unsigned int uiNrOfWorkItemsPerUnitExec = std::min( std::min( uiNrOfWorkItemsPerUnit, m_uiMaxWorkGroupSize ), uiTotalNrOfWorkItemsExec );

	if( uiNrOfWorkItemsPerUnitExec > 0 && uiTotalNrOfWorkItemsExec % uiNrOfWorkItemsPerUnitExec != 0 )
		uiTotalNrOfWorkItemsExec += uiNrOfWorkItemsPerUnitExec - ( uiTotalNrOfWorkItemsExec % uiNrOfWorkItemsPerUnitExec );

	size_t aGlobal[] = { uiTotalNrOfWorkItemsExec };
	size_t aLocal[] = { uiNrOfWorkItemsPerUnitExec };

	int iErr = clEnqueueNDRangeKernel( m_cmdQueue, m_kernel, 1, NULL, aGlobal, ( uiNrOfWorkItemsPerUnit > 0 ) ? aLocal : NULL, 0, NULL, NULL );
	m_uiNumArguments = 0;

	if(iErr < 0)
	{
		Log::getLog("GPUAbstractionLayer") << Log::EL_ERROR << "Failed to enqueue the kernel. Error code: " << errorNumberToString(iErr) << Log::endl;
		return false;
	}

	return true;
}

bool Kernel::execute( const size_t* pTotalNrOfWorkItems, const size_t* pNrOfWorkItemsPerUnit )
{
	int iErr = clEnqueueNDRangeKernel( m_cmdQueue, m_kernel, 1, NULL, pTotalNrOfWorkItems, pNrOfWorkItemsPerUnit, 0, NULL, NULL );
	m_uiNumArguments = 0;

	if(iErr < 0)
	{
		Log::getLog("GPUAbstractionLayer") << Log::EL_ERROR << "Failed to enqueue the kernel. Error code: " << errorNumberToString(iErr) << Log::endl;
		return false;
	}

	return true;
}

bool Kernel::finish()
{
    int iErr = clFinish( m_cmdQueue );
    if( iErr != CL_SUCCESS )
    {
		Log::getLog("GPUAbstractionLayer") << Log::EL_ERROR << "Failed to wait for command queue to finish. Error code: " << errorNumberToString(iErr) << Log::endl;
        return false;
    }

    return true;
}

Kernel::~Kernel()
{
	clReleaseKernel( m_kernel );
}

Kernel::Kernel( cl_command_queue cmdQueue, cl_kernel kernel, unsigned int uiMaxWorkGroupSize )
	: m_cmdQueue( cmdQueue )
	, m_kernel( kernel )
	, m_uiNumArguments( 0 )
	, m_uiMaxWorkGroupSize( uiMaxWorkGroupSize )
{
}

#endif // USE_OPENCL
