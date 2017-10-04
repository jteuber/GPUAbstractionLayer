#ifndef OPENCL_H
#define OPENCL_H

#ifdef USE_OPENCL

#include <string>
#include <map>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif // MAC

#include "Kernel.h"
#include "Log.h"
#include "OpenCLTools.h"


class GAL_EXPORT OpenCL
{
	friend class GPUWrapper;

public:
	bool init( int argc, const char **argv );
	bool init( unsigned int uiPlatform = 0, unsigned int uiDevice = 0 );

	template<typename T> cl_mem devNew( unsigned int uiNrOfElements );
	template<typename T> cl_mem copyToDev( T* hData, unsigned int uiNrOfElements );
	template<typename T> cl_mem copyToConstant( T* hData, unsigned int uiNrOfElements );

	void devDelete( cl_mem dData, size_t sizeInBytes );
	template<typename T> void devMemSet( cl_mem dData, unsigned int uiNrOfElements, int data = 0 );


	template<typename T> void copyFromDev( T* hData, const cl_mem dData, unsigned int uiNrOfElements, unsigned int offset = 0 );
	template<typename T> void copyToDev( cl_mem dData, const T* hData, unsigned int uiNrOfElements );
	template<typename T> void copyDevToDev( cl_mem dest, const cl_mem src, unsigned int uiNrOfElements, unsigned int offset = 0 );


	void scan( cl_mem dOut, cl_mem dIn, size_t numElements );
	unsigned int reduce( cl_mem dIn, size_t numElements );

	void sortUIntUInt( cl_mem dKeys, cl_mem dValues, size_t numElements );
	void sortFloatUInt( cl_mem dKeys, cl_mem dValues, size_t numElements );


	unsigned int getTotalAvailableVRAM();
	int getFreeVRAM();

	Kernel* createKernel( std::string strKernelName, std::string strFileName );
	Kernel* createKernel( std::string strKernelSource );


	cl_command_queue getCommandQueue();

	void setKernelFolder( std::string strKernelFolder );

private:
	OpenCL();
	~OpenCL();

private:
	cl_device_id	m_device;
	cl_context		m_context;
	std::string 	m_strPlatformName;

	cl_command_queue  m_commandQueue;

	std::map<std::string, cl_program> m_mapPrograms;

	unsigned int m_uiMaxWorkGroupSize;
	std::string m_strKernelFolder;

	size_t m_reservedMemory;
};


template<typename T>
cl_mem OpenCL::devNew(unsigned int uiNrOfElements)
{
	int iErr;
	cl_mem mem = clCreateBuffer( m_context, CL_MEM_READ_WRITE, uiNrOfElements * sizeof(T), NULL, &iErr );

	m_reservedMemory += uiNrOfElements * sizeof(T);

	if( iErr != CL_SUCCESS )
	{
		Log::getLog("GPUAbstractionLayer") << Log::EL_ERROR << "Error creating an OpenCL buffer (" << errorNumberToString( iErr ) << ")" << Log::endl;
		return NULL;
	}

	return mem;
}

template<typename T>
cl_mem OpenCL::copyToDev(T* hData, unsigned int uiNrOfElements)
{
	int iErr;
	cl_mem mem = clCreateBuffer( m_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, uiNrOfElements * sizeof( T ), (void*)hData, &iErr );

	m_reservedMemory += uiNrOfElements * sizeof(T);

	if( iErr != CL_SUCCESS )
	{
		Log::getLog("GPUAbstractionLayer") << Log::EL_ERROR << "Error creating an OpenCL buffer (" << errorNumberToString( iErr ) << ")" << Log::endl;
		return NULL;
	}

	return mem;
}

template<typename T>
cl_mem OpenCL::copyToConstant(T* hData, unsigned int uiNrOfElements)
{
	int iErr;
	cl_mem mem = clCreateBuffer( m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, uiNrOfElements * sizeof(T), (void*)hData, &iErr );

	if( iErr != CL_SUCCESS )
	{
		Log::getLog("GPUAbstractionLayer") << Log::EL_ERROR << "Error creating an OpenCL read-only buffer (" << errorNumberToString( iErr ) << ")" << Log::endl;
		return NULL;
	}

	return mem;
}

template<typename T>
void OpenCL::devMemSet(cl_mem dData, unsigned int uiNrOfElements, int data)
{
	// TODO: Does not work on NVIDIA NVS 5400M
//#if defined(CL_VERSION_1_2)
//	int iErr = clEnqueueFillBuffer( m_commandQueue, dData, (void*)&data, sizeof( int ), 0, uiNrOfElements * sizeof( int ), 0, NULL, NULL );
//	if( iErr != CL_SUCCESS )
//		Log::getLog("ProtoSphere") << Log::EL_ERROR << "Error mem-setting an OpenCL buffer (" << errorNumberToString( iErr ) << ")" << Log::endl;
//#else
	T* result = (T*)malloc(uiNrOfElements*sizeof(T));
	memset(result, data, uiNrOfElements*sizeof(T));

	copyToDev( dData, result, uiNrOfElements );

	free(result);
//#endif
}


template<typename T>
void OpenCL::copyFromDev(T* hData, const cl_mem dData, unsigned int uiNrOfElements, unsigned int offset)
{
	int iErr = clEnqueueReadBuffer( m_commandQueue, dData, CL_TRUE, offset * sizeof(T), uiNrOfElements * sizeof(T), hData, 0, NULL, NULL );
	if( iErr != CL_SUCCESS )
		Log::getLog("GPUAbstractionLayer") << Log::EL_ERROR << "Error copying an OpenCL buffer to the host (" << errorNumberToString( iErr ) << ")" << Log::endl;
}

template<typename T>
void OpenCL::copyToDev(cl_mem dData, const T* hData, unsigned int uiNrOfElements)
{
	int iErr = clEnqueueWriteBuffer( m_commandQueue, dData, CL_TRUE, 0, uiNrOfElements * sizeof(T), hData, 0, NULL, NULL );
	if( iErr != CL_SUCCESS )
		Log::getLog("GPUAbstractionLayer") << Log::EL_ERROR << "Error writing to an OpenCL buffer (" << errorNumberToString( iErr ) << ")" << Log::endl;
}

template<typename T>
void OpenCL::copyDevToDev( cl_mem dest, const cl_mem src, unsigned int uiNrOfElements, unsigned int offset )
{
	int iErr = clEnqueueCopyBuffer( m_commandQueue, src, dest, 0, offset * sizeof(T), uiNrOfElements * sizeof(T), 0, NULL, NULL );
	if( iErr != CL_SUCCESS )
		Log::getLog("GPUAbstractionLayer") << Log::EL_ERROR << "Error copying an OpenCL buffer to another buffer (" << errorNumberToString( iErr ) << ")" << Log::endl;
}

#endif // USE_OPENCL

#endif // OPENCL_H
