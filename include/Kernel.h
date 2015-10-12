#ifndef KERNEL_H
#define KERNEL_H

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "GPUAbstractionLayer_global.h"

#include "OpenCLTools.h"
#include "Log.h"


class GAL_EXPORT Kernel
{
	friend class OpenCL;

public:
	template<typename T> bool addArgument( T arg );
	bool addSharedMem( const unsigned int uiSize );

	bool execute( const unsigned int uiTotalNrOfWorkItems, const unsigned int uiNrOfWorkItemsPerUnit );
	bool execute( const size_t* pTotalNrOfWorkItems, const size_t* pNrOfWorkItemsPerUnit );

	template<typename... Args> bool execute( const unsigned int uiTotalNrOfWorkItems, const unsigned int uiNrOfWorkItemsPerUnit, Args... args );
	template<typename... Args> bool executeWithSharedMem( const unsigned int uiTotalNrOfWorkItems, const unsigned int uiNrOfWorkItemsPerUnit, const unsigned int uiSharedMemSize, Args... args );

	bool finish();

	~Kernel();

private:
	Kernel( cl_command_queue cmdQueue, cl_kernel kernel, unsigned int uiMaxWorkGroupSize );

	template<typename T, typename... Args> bool addArgument( T arg, Args... args );

private:
	cl_command_queue m_cmdQueue;
	cl_kernel m_kernel;

	unsigned int m_uiNumArguments;
	unsigned int m_uiMaxWorkGroupSize;
};

template<typename T>
bool Kernel::addArgument( T arg )
{
	int iErr = clSetKernelArg( m_kernel, m_uiNumArguments, sizeof(T), &arg );
	++m_uiNumArguments;

	if(iErr < 0)
	{
		Log::getLog("GPUAbstractionLayer") << Log::EL_ERROR << "Unable to set kernel argument number " << m_uiNumArguments << ". Error code: " << errorNumberToString(iErr) << Log::endl;
		return false;
	}

	return true;
}

template<typename T, typename... Args>
bool Kernel::addArgument( T arg, Args... args )
{
	if( !addArgument( arg ) )
		return false;

	return addArgument( args... );
}

template<typename... Args> 
bool Kernel::execute( const unsigned int uiTotalNrOfWorkItems, const unsigned int uiNrOfWorkItemsPerUnit, Args... args )
{
	if( !addArgument( args... ) )
		return false;

	return execute( uiTotalNrOfWorkItems, uiNrOfWorkItemsPerUnit );
}

template<typename... Args> 
bool Kernel::executeWithSharedMem( const unsigned int uiTotalNrOfWorkItems, const unsigned int uiNrOfWorkItemsPerUnit, const unsigned int uiSharedMemSize, Args... args )
{
	if( !addSharedMem( uiSharedMemSize ) )
		return false;

	return execute( uiTotalNrOfWorkItems, uiNrOfWorkItemsPerUnit, args... );
}

#endif // KERNEL_H
