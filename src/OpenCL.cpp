#ifdef USE_OPENCL

#include "OpenCL.h"

#include <algorithm>

#include <vector_types.h>

#include <boost/compute.hpp>
#include <boost/serialization/static_warning.hpp>

#include "Log.h"
#include "ArgumentTools.h"


//BOOST_COMPUTE_TYPE_NAME(short4_ocl, cl_short4);
BOOST_COMPUTE_ADAPT_STRUCT(short4, short4, (x,y,z,w));


bool OpenCL::init(int argc, const char** argv)
{
	unsigned int uiPlatform = 0;
	if( checkArgumentExists(argc, argv, "-platform") )
		uiPlatform = getArgumentInt(argc, argv, "-platform");

	unsigned int uiDevice = 0;
	if( checkArgumentExists(argc, argv, "-device") )
		uiDevice = getArgumentInt(argc, argv, "-device");

	return init( uiPlatform, uiDevice );
}

bool OpenCL::init( unsigned int uiPlatform, unsigned int uiDevice )
{
	int iErr;
	cl_platform_id * pPlatformIDs = NULL;
	cl_uint uiNrOfPlatforms = 0;
	char acNameBuffer[128] = { 0 };

	// get number of available platforms
	cl_int err = clGetPlatformIDs( 0, NULL, &uiNrOfPlatforms );
	if ( CL_SUCCESS != err )
		return false;

	// get all platform IDs
	pPlatformIDs = new cl_platform_id[uiNrOfPlatforms];
	iErr = clGetPlatformIDs( uiNrOfPlatforms, pPlatformIDs, NULL );
	if ( iErr != CL_SUCCESS )
	{
		Log::getLog( "ProtoSphere" ) << Log::EL_ERROR << "Error getting OpenCL platform (" << errorNumberToString( iErr ) << ")" << Log::endl;
		return false;
	}

	// get the platform. Either the one given by the user via the command line or simply the first one
	int platformID = 1;
	if ( uiPlatform > 0 )
		platformID = uiPlatform;

	cl_platform_id platformInUse = pPlatformIDs[platformID - 1];

	// print out the list of platforms and devices and mark the entry we will use
	for ( cl_uint i = 0; i < uiNrOfPlatforms; ++i )
	{
		err = clGetPlatformInfo( pPlatformIDs[i],
			CL_PLATFORM_VENDOR,
			128 * sizeof( char ),
			acNameBuffer,
			NULL );
		if ( CL_SUCCESS != err )
			return false;

		if ( acNameBuffer != NULL )
		{
			Log::getLog( "ProtoSphere" ) << Log::EL_INFO << "OpenCL platform id " << i + 1 << ": " << acNameBuffer << Log::endl;

			cl_uint num_devices = 0;
			iErr = clGetDeviceIDs( pPlatformIDs[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices );

			cl_device_id* pDeviceIDs = new cl_device_id[num_devices];
			iErr = clGetDeviceIDs( pPlatformIDs[i], CL_DEVICE_TYPE_ALL, num_devices, pDeviceIDs, NULL );

			cl_device_id defaultDevice;
			iErr = clGetDeviceIDs( pPlatformIDs[i], CL_DEVICE_TYPE_DEFAULT, 1, &defaultDevice, NULL );
			if ( iErr != CL_SUCCESS )
			{
				Log::getLog( "ProtoSphere" ) << Log::EL_ERROR << "Error getting default OpenCL device (" << errorNumberToString( iErr ) << ")" << Log::endl;
				return false;
			}

			for ( unsigned j = 0; j < num_devices; ++j )
			{
				char deviceName[48];
				cl_device_type devType;

				clGetDeviceInfo( pDeviceIDs[j], CL_DEVICE_NAME, 48, &deviceName, NULL );
				clGetDeviceInfo( pDeviceIDs[j], CL_DEVICE_TYPE, sizeof( cl_device_type ), &devType, NULL );

				Log::getLog( "ProtoSphere" ) << Log::EL_INFO;
				if ( i == platformID - 1 && (uiDevice == 0 ? pDeviceIDs[j] == defaultDevice : uiDevice == j + 1) )
				{
					Log::getLog( "ProtoSphere" ) << "->";
					m_device = pDeviceIDs[j];
				}
				Log::getLog( "ProtoSphere" ) << "\t";
				Log::getLog( "ProtoSphere" ) << std::string( (devType == CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU" );
				Log::getLog( "ProtoSphere" ) << " Device " << std::string( deviceName );
				if ( pDeviceIDs[j] == defaultDevice )
					Log::getLog( "ProtoSphere" ) << " (default)";
				Log::getLog( "ProtoSphere" ) << Log::endl;
			}
		}
	}

	// create the context
	cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platformInUse, 0 };

	m_context = clCreateContext( properties, 1, &m_device, NULL, NULL, &iErr );
	if ( iErr != CL_SUCCESS )
	{
		Log::getLog( "ProtoSphere" ) << Log::EL_ERROR << "Error creating the OpenCL context (" << errorNumberToString( iErr ) << ")" << Log::endl;
		return false;
	}

	m_commandQueue = clCreateCommandQueue( m_context, m_device, 0, &iErr );
	if ( iErr != CL_SUCCESS )
	{
		Log::getLog( "ProtoSphere" ) << Log::EL_ERROR << "Error creating an OpenCL command queue within the just created context (" << errorNumberToString( iErr ) << ")" << Log::endl;
		return false;
	}

	size_t returned_size = 0;
	size_t max_workgroup_size = 0;
	iErr = clGetDeviceInfo( m_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof( size_t ), &max_workgroup_size, &returned_size );
	if ( iErr != CL_SUCCESS )
	{
		Log::getLog( "ProtoSphere" ) << Log::EL_ERROR << "Failed to retrieve device info (" << errorNumberToString( iErr ) << ")" << Log::endl;
		return false;
	}

	m_uiMaxWorkGroupSize = max_workgroup_size;

	return true;
}

void OpenCL::devDelete( cl_mem dData, size_t sizeInBytes )
{
	int iErr = clReleaseMemObject( dData );
	if( iErr != CL_SUCCESS )
		Log::getLog("GPUAbstractionLayer") << Log::EL_ERROR << "Error releasing an OpenCL buffer (" << errorNumberToString( iErr ) << ")" << Log::endl;

	m_reservedMemory -= sizeInBytes;
}

void OpenCL::scan(cl_mem dOut, cl_mem dIn, size_t numElements)
{
	boost::compute::command_queue queue( m_commandQueue );
	boost::compute::buffer bufIn( dIn );
	boost::compute::buffer bufOut( dOut );

	boost::compute::exclusive_scan( boost::compute::make_buffer_iterator<unsigned int>( bufIn, 0 ), boost::compute::make_buffer_iterator<unsigned int>( bufIn, numElements ),
			boost::compute::make_buffer_iterator<unsigned int>( bufOut, 0 ), queue );

	queue.finish();

	//m_pScanner->scan( dOut, dIn, numElements );
}

unsigned int OpenCL::reduce( cl_mem dIn, size_t numElements )
{
	unsigned int uiRet = 0;

	boost::compute::command_queue queue( m_commandQueue );
	boost::compute::buffer bufIn( dIn );

	boost::compute::reduce( boost::compute::make_buffer_iterator<unsigned int>( bufIn, 0 ), boost::compute::make_buffer_iterator<unsigned int>( bufIn, numElements ), &uiRet, queue );

	queue.finish();

	return uiRet;
}

void OpenCL::sortUIntUInt(cl_mem dKeys, cl_mem dValues, size_t numElements)
{
	if( m_strPlatformName == "NVIDIA CUDA")
	{
		// As long as boost::compute doesn't fix the sort_by_key function, use the cpu
		unsigned int* pKeys = new unsigned int[ numElements ];
		copyFromDev( pKeys, dKeys, numElements );

		unsigned int* pValues = new unsigned int[ numElements ];
		copyFromDev( pValues, dValues, numElements );

		std::multimap<unsigned int, unsigned int> mapCells;

		for( unsigned int i = 0; i < numElements; ++i )
		{
			mapCells.insert( std::make_pair( pKeys[ i ], pValues[ i ] ) );
		}

		unsigned int i = 0;
		for( std::multimap<unsigned int, unsigned int>::iterator it = mapCells.begin(); it != mapCells.end(); ++it, ++i )
		{
			pKeys[ i ] = it->first;
			pValues[ i ] = it->second;
		}

		copyToDev( dKeys, pKeys, numElements );
		copyToDev( dValues, pValues, numElements );

		delete[] pKeys;
		delete[] pValues;
	}
	else
	{
		boost::compute::command_queue queue( m_commandQueue );

		boost::compute::buffer bufKeys( dKeys );
		boost::compute::buffer bufValues( dValues );

		boost::compute::sort_by_key( boost::compute::make_buffer_iterator<unsigned int>( bufKeys, 0 ), boost::compute::make_buffer_iterator<unsigned int>( bufKeys, numElements ),
			boost::compute::make_buffer_iterator<unsigned int>( bufValues, 0 ), queue );

		queue.finish();
	}

}


void OpenCL::sortFloatUInt(cl_mem dKeys, cl_mem dValues, size_t numElements)
{
	boost::compute::command_queue queue( m_commandQueue );

	boost::compute::buffer bufKeys( dKeys );
	boost::compute::buffer bufValues( dValues );

	boost::compute::sort_by_key( boost::compute::make_buffer_iterator<float>( bufKeys, 0 ), boost::compute::make_buffer_iterator<float>( bufKeys, numElements ),
									boost::compute::make_buffer_iterator<unsigned int>( bufValues, 0 ), queue );
}

unsigned int OpenCL::getTotalAvailableVRAM()
{
	cl_ulong size;
	clGetDeviceInfo( m_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &size, nullptr );
	return size;
}

int OpenCL::getFreeVRAM()
{
	cl_ulong size;
	clGetDeviceInfo( m_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &size, nullptr );
	return size-m_reservedMemory;
}


Kernel* OpenCL::createKernel(std::string strKernelName, std::string strFileName)
{
	if( m_commandQueue == 0 )
	{
		Log::getLog("GPUAbstractionLayer") << Log::EL_ERROR << "OpenCL is not initialized yet." << Log::endl;
		return NULL;
	}

	cl_program program;
	int iErr;

	// check the filename, if empty, abort
	if( strFileName.empty() )
	{
		Log::getLog("GPUAbstractionLayer") << Log::EL_ERROR << "Filename necessary for kernel " << strKernelName << Log::endl;
		return NULL;
	}

	// find or create the program
	std::map<std::string, cl_program>::iterator it = m_mapPrograms.find( strFileName );
	if( it == m_mapPrograms.end() )
	{
		// load the program from the given file
		FILE *program_handle;
		char *program_buffer;
		size_t program_size, log_size;

		// Read program file and place content into buffer
		program_handle = fopen( (m_strKernelFolder + strFileName).c_str(), "r" );
		if(program_handle == NULL)
		{
			Log::getLog("GPUAbstractionLayer") << Log::EL_ERROR << "Unable to find file " << strFileName << " in the folder " << m_strKernelFolder << Log::endl;
			return NULL;
		}
		fseek( program_handle, 0, SEEK_END );
		program_size = ftell( program_handle );
		rewind( program_handle );
		program_buffer = (char*)malloc( program_size + 1 );
		program_buffer[program_size] = '\0';
		fread( program_buffer, sizeof( char ), program_size, program_handle );
		fclose( program_handle );

		// Create program from file
		program = clCreateProgramWithSource( m_context, 1, (const char**)&program_buffer, &program_size, &iErr );
		if ( iErr != CL_SUCCESS )
		{
			Log::getLog("GPUAbstractionLayer") << Log::EL_ERROR << "Unable to create the program from file " << strFileName << Log::endl;
			return NULL;
		}
		free(program_buffer);

		// Build program
		std::string strBuildParams( "-I " + m_strKernelFolder );

		// if this is a debug build and we are on a CPU use the debug option!
#ifdef DEBUG
		cl_device_type devType;
		clGetDeviceInfo( m_device, CL_DEVICE_TYPE, sizeof(cl_device_type), &devType, NULL );
		if( devType == CL_DEVICE_TYPE_CPU )
			strBuildParams += " -g";
#else
		strBuildParams += " -cl-unsafe-math-optimizations -cl-mad-enable -cl-no-signed-zeros";
#endif // DEBUG

#ifdef MAC
		strBuildParams += " -DMAC";
#endif // MAC

		iErr = clBuildProgram(program, 0, NULL, strBuildParams.c_str(), NULL, NULL);
		if( iErr != CL_SUCCESS )
		{
			char* program_log;
			/* Find size of log and print to std output */
			clGetProgramBuildInfo(program, m_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
			program_log = (char*) malloc(log_size + 1);
			program_log[log_size] = '\0';
			clGetProgramBuildInfo(program, m_device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
			Log::getLog("GPUAbstractionLayer") << Log::EL_FATAL_ERROR << "Error compiling the file " << strFileName << ":\n" << program_log << "\n" << Log::endl;
			free(program_log);

			return NULL;
		}
#ifdef DEBUG
		else
		{
			char* program_log;
			/* Find size of log and print to std output */
			clGetProgramBuildInfo(program, m_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
			program_log = (char*) malloc(log_size + 1);
			program_log[log_size] = '\0';
			clGetProgramBuildInfo(program, m_device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
			Log::getLog("GPUAbstractionLayer") << Log::EL_INFO << "Build log for file " << strFileName << ":\n" << program_log << "\n" << Log::endl;
			free(program_log);
		}
#endif // DEBUG

		// insert program into map
		m_mapPrograms.insert( std::make_pair( strFileName, program ) );
	}
	else
	{
		program = it->second;
	}

	cl_kernel kernel = clCreateKernel(program, strKernelName.c_str(), &iErr);
	if( iErr != CL_SUCCESS )
	{
		Log::getLog("GPUAbstractionLayer") << Log::EL_ERROR << "Unable to create the kernel " << strKernelName << " in file " << strFileName << Log::endl;
		return NULL;
	}

	size_t wgSize;
	iErr = clGetKernelWorkGroupInfo( kernel, m_device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wgSize, NULL);
	if( iErr != CL_SUCCESS )
	{
		Log::getLog("GPUAbstractionLayer") << Log::EL_ERROR << "Failed to get kernel work group size (" << errorNumberToString( iErr ) << ")" << Log::endl;
		return NULL;
	}

	return new Kernel( m_commandQueue, kernel, std::min( (unsigned int)wgSize, m_uiMaxWorkGroupSize ) );
}

Kernel* OpenCL::createKernel( std::string strKernelSource )
{
	int iErr = 0;

	size_t sKernelLength = strKernelSource.length();
	const char* program_buffer = strKernelSource.c_str();
	cl_program program = clCreateProgramWithSource( m_context, 1, (const char**)&program_buffer, &sKernelLength, &iErr );
	if( iErr != CL_SUCCESS )
	{
		Log::getLog( "GPUAbstractionLayer" ) << Log::EL_ERROR << "Unable to create the program from the given source: " << strKernelSource.substr( 0, strKernelSource.find( '\n' ) ) << Log::endl;
		return NULL;
	}

	/* Build program */
	std::string strBuildParams;

	// if this is a debug build and we are on a CPU use the debug option!
#ifdef DEBUG
	cl_device_type devType;
	clGetDeviceInfo( m_device, CL_DEVICE_TYPE, sizeof( cl_device_type ), &devType, NULL );
	if( devType == CL_DEVICE_TYPE_CPU )
		strBuildParams += " -g";
#else
	strBuildParams += " -cl-unsafe-math-optimizations -cl-mad-enable -cl-no-signed-zeros";
#endif

#ifdef MAC
	strBuildParams += " -DMAC";
#endif

	iErr = clBuildProgram( program, 0, NULL, strBuildParams.c_str(), NULL, NULL );
	if( iErr != CL_SUCCESS )
	{
		char* program_log;
		size_t log_size;
		/* Find size of log and print to std output */
		clGetProgramBuildInfo( program, m_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size );
		program_log = (char*)malloc( log_size + 1 );
		program_log[ log_size ] = '\0';
		clGetProgramBuildInfo( program, m_device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL );
		Log::getLog( "GPUAbstractionLayer" ) << Log::EL_FATAL_ERROR << "Error compiling the source:\n" << program_log << "\n" << Log::endl;
		free( program_log );

		return NULL;
	}
#ifdef DEBUG
	else
	{
		char* program_log;
		size_t log_size;
		/* Find size of log and print to std output */
		clGetProgramBuildInfo( program, m_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size );
		program_log = (char*)malloc( log_size + 1 );
		program_log[ log_size ] = '\0';
		clGetProgramBuildInfo( program, m_device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL );
		Log::getLog( "GPUAbstractionLayer" ) << Log::EL_INFO << "Build log:\n" << program_log << "\n" << Log::endl;
		free( program_log );
	}
#endif

	cl_kernel kernel;
	iErr = clCreateKernelsInProgram( program, 1, &kernel, NULL );
	if( iErr != CL_SUCCESS )
	{
		Log::getLog( "GPUAbstractionLayer" ) << Log::EL_ERROR << "Unable to create a kernel from the given source code" << Log::endl;
		return NULL;
	}

	size_t wgSize;
	iErr = clGetKernelWorkGroupInfo( kernel, m_device, CL_KERNEL_WORK_GROUP_SIZE, sizeof( size_t ), &wgSize, NULL );
	if( iErr != CL_SUCCESS )
	{
		Log::getLog( "GPUAbstractionLayer" ) << Log::EL_ERROR << "Failed to get kernel work group size (" << errorNumberToString( iErr ) << ")" << Log::endl;
		return NULL;
	}

	return new Kernel( m_commandQueue, kernel, std::min( (unsigned int)wgSize, m_uiMaxWorkGroupSize ) );
}


cl_command_queue OpenCL::getCommandQueue()
{
	return m_commandQueue;
}


// TODO: move this somewhere else! This is not the right class for this
/*BOOST_COMPUTE_FUNCTION(bool, isActiveOCL, (struct CellOnDevice x),
{
	return (x.m_status) > -1;
});*/

/*unsigned int OpenCL::countActiveCells( cl_mem dCells, int nrOfGridCells )
{
	boost::compute::command_queue queue( m_commandQueue );

	boost::compute::buffer bufCells( dCells );

	return boost::compute::count_if( boost::compute::make_buffer_iterator<CellOnDevice>( bufCells, 0 ), boost::compute::make_buffer_iterator<CellOnDevice>( bufCells, nrOfGridCells ),
			isActiveOCL, queue );
}

void OpenCL::copyActiveCells( cl_mem dCells, cl_mem dActiveCells, int nrOfGridCells )
{
	boost::compute::command_queue queue( m_commandQueue );

	boost::compute::buffer bufCells( dCells );
	boost::compute::buffer bufActiveCells( dActiveCells );

	boost::compute::copy_if( boost::compute::make_buffer_iterator<CellOnDevice>( bufCells, 0 ), boost::compute::make_buffer_iterator<CellOnDevice>( bufCells, nrOfGridCells ),
			boost::compute::make_buffer_iterator<CellOnDevice>( bufActiveCells, 0 ), isActiveOCL, queue );
}*/


void OpenCL::setKernelFolder( std::string strKernelFolder )
{
	m_strKernelFolder = strKernelFolder;
}


OpenCL::OpenCL()
	: m_device( NULL )
	, m_context( NULL )
	, m_commandQueue( NULL )
	, m_uiMaxWorkGroupSize( 32 )
	, m_strKernelFolder( "./oclKernels/" )
{
}

OpenCL::~OpenCL()
{
	m_mapPrograms.clear();

	clReleaseCommandQueue( m_commandQueue );
	clReleaseContext( m_context );
}

#endif // USE_OPENCL
