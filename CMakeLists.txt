#####################################
# CMake File for GPUAbstractionLayer
#
# Requires:
# - helper_cuda.h
#   CUDA_EXAMPLE_directory: /path/to/CUDA Samples/v7.5/common/inc
# - if (BOOST < 1.61)
#   BOOST_EXT_COMPUTE_directory: /path/to/external/compute/include
#
# Workarounds:
# - OpenCL was added to cmake 3.1
#   provide FindOpenCL.cmake for lower versions
# - CMAKE_CXX_FLAGS are forwarded to NVCC in cmake 3.6.1? (3.4.3 works also)
#   provide FindCUDA.cmake + FindCUDA/ of CMake 3.4.3 for lower versions
#   https://github.com/Kitware/CMake/commit/b4e54f9b8c748f78d16e9da055a7e0436d7654ef
# - CMake 3.1 supports set(CMAKE_CXX_STANDARD 11)
#   add -std=c++11 manually for lower versions
# - CMake 3.1 supports target_link_libraries(<target> Threads::Threads)
#   add -pthread and link library extra
#
# Usage: see README.md
#
# https://cmake.org/cmake/help/v3.0/module/FindCUDA.html
# https://cmake.org/cmake/help/v3.1/module/FindOpenCL.html
# https://cmake.org/cmake/help/v3.0/module/FindBoost.html
#
# Philipp Dittmann <dittmann@tzi.de>
# 4th of August 2016
#####################################
cmake_minimum_required (VERSION 2.8.12.2)

if(UNIX)
	set(CMAKE_CXX_COMPILER "/usr/bin/g++-5")
	set(CMAKE_C_COMPILER "/usr/bin/gcc-5")
endif() # UNIX

project (GPUAbstractionLayer)

message(STATUS "CMAKE_VERSION: ${CMAKE_VERSION}")

# TODO: individual builds
OPTION(BUILD_CUDA_SUPPORT   "If TRUE some algorithm runs with CUDA"   ON)
OPTION(BUILD_OPENCL_SUPPORT "If TRUE some algorithm runs with OpenCL" ON)
SET(CMAKE_DEBUG_POSTFIX "d")

# OpenCL was introduced at 3.1
if(BUILD_CUDA_SUPPORT AND CMAKE_VERSION VERSION_LESS "3.1")
	message(STATUS "CMake < 3.1: CMAKE_MODULE_PATH += cmake/BELOW_3_1_FindOpenCL")
	list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/BELOW_3_1_FindOpenCL")
endif()

# CMAKE_CXX_FLAGS forward to nvcc (i.e. -fPIC); introduced at 3.4.3:
if(CMAKE_VERSION VERSION_LESS "3.4.3")
	message(STATUS "CMake < 3.4.3: CMAKE_MODULE_PATH += cmake/BELOW_3_4_3_FindCUDA")
	list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/BELOW_3_4_3_FindCUDA")
endif()

# Some preprocessor makros for source code
add_definitions(-DPSE_LIBRARY)
if(WIN32)
	# avoid conflicts between CUDA <helper_math.h> and c:\Program Files (x86)\Windows Kits\8.1\Include\shared\minwindef.h
	add_definitions(-DNOMINMAX)
elseif(UNIX)
	add_definitions(-Dlinux)
endif()

# Setting (i.e. external references=
if (EXISTS "${PROJECT_SOURCE_DIR}/settings.cmake")
	include("${PROJECT_SOURCE_DIR}/settings.cmake")
endif()

FILE(GLOB_RECURSE HEADER_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/include include/*.h)
FILE(GLOB_RECURSE HEADER_FILES_CUDA RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/include include/*.cuh)

# collect all .cpp source
FILE(GLOB SOURCES src/*.cpp include/*.h)

# .cu files in src as sources
FILE(GLOB SOURCES_CUDA src/*.cu include/*.cuh)

if(BUILD_CUDA_SUPPORT)
	#set(CUDA_HOST_COMPILATION_CPP OFF) # deprecated since CUDA 3.0
	set(CUDA_USE_STATIC_CUDA_RUNTIME OFF) # link cudart instead of cuda_static
	#set(CUDA_64_BIT_DEVICE_CODE OFF) # set 64 bit (ON) or 32 bit (OFF) for cuda (default: match host bit size)
	set(CUDA_SEPARABLE_COMPILATION ON) # avoid compile errors, caused by cross references within .cu files
	#set(CUDA_PROPAGATE_HOST_FLAGS OFF) # Stop forwarding CMAKE_{C,CXX}_FLAGS to NVCC
	#set(CUDA_VERBOSE_BUILD ON) # Show all executed NVCC calls

	add_definitions(-DUSE_CUDA)
	
	find_package(CUDA 7.0)
	if (CUDA_FOUND)
		message(STATUS "CUDA Version: ${CUDA_VERSION}")
		message(STATUS "CUDA Include: ${CUDA_INCLUDE_DIRS}")
		message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")

		# check default pathes
		if (NOT CUDA_EXAMPLE_INCLUDE_DIR)
			find_path(CUDA_EXAMPLE_INCLUDE_DIR helper_cuda.h
				HINTS "${CUDA_EXAMPLE_INCLUDE_DIR}"
				PATHS
					"C:/ProgramData/NVIDIA Corporation/CUDA Samples/v${CUDA_VERSION}/common/inc"
					"$ENV{HOME}/NVIDIA_CUDA-${CUDA_VERSION}_Samples/common/inc"
					"/opt/cuda/samples/common/inc"
			)
		endif()
		
		if (NOT CUDA_EXAMPLE_INCLUDE_DIR)
			message(SEND_ERROR "Path to CUDA Sample include 'CUDA_EXAMPLE_INCLUDE_DIR' not provided!")
		else()
			message(STATUS "Load helper_cuda.h from CUDA samples: ${CUDA_EXAMPLE_INCLUDE_DIR}")
			include_directories(${CUDA_EXAMPLE_INCLUDE_DIR})
		endif()
		
		# Compile for current graphics card
		CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS Auto)
		list(APPEND CUDA_NVCC_FLAGS "${CUDA_ARCH_FLAGS}")
	endif() # CUDA_FOUND
else()
	# set CUDA annotations to nothing
	add_definitions(-D__global__= -D__constant__= -D__device__= -D__host__=)
endif()

if(BUILD_OPENCL_SUPPORT)
	add_definitions(-DUSE_OPENCL)
	
	find_package(OpenCL)
	if (OpenCL_FOUND)
		message(STATUS "OpenCL Version: ${OpenCL_VERSION_STRING}")
		message(STATUS "OpenCL Include: ${OpenCL_INCLUDE_DIRS}")
		message(STATUS "OpenCL Libraries: ${OpenCL_LIBRARIES}")
		include_directories(${OpenCL_INCLUDE_DIRS})

		# OpenCL part uses Boost
		find_package(Boost)
		if (Boost_FOUND)
			message(STATUS "Boost Version: ${Boost_VERSION}")
			message(STATUS "Boost Include: ${Boost_INCLUDE_DIRS}")
			message(STATUS "Boost Libraries: ${Boost_LIBRARIES}")

			include_directories(${Boost_INCLUDE_DIRS})
		
			if (Boost_VERSION VERSION_LESS 106100) # compute was integrated in 1.61, was external before
				if (NOT BOOST_EXT_COMPUTE_directory)
					message(SEND_ERROR "Boost < 1.61, but BOOST_EXT_COMPUTE_directory not provided!")
				else()
					message(STATUS "Boost < 1.61: include external compute: ${BOOST_EXT_COMPUTE_directory}")
					include_directories(${BOOST_EXT_COMPUTE_directory})
				endif()
			endif() # Boost_VERSION < 1.61
		endif() # BOOST_FOUND
	endif() # OpenCL_FOUND
endif()

# declare project headers
include_directories(include)

if(UNIX)
	if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
		# Use c++11 code convention
		if(CMAKE_VERSION VERSION_LESS "3.1")
			message(STATUS "CMake ${CMAKE_VERSION} < 3.1: CMAKE_CXX_FLAGS += -std=c++11")
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
		else()
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
			set(CMAKE_CXX_STANDARD 11)
		endif()

		# set methods default visibility to hidden
		message(STATUS "CMAKE_C_FLAGS += -fvisibility=hidden")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")

		# set Position Independent Code (PIC) in shared libraries
		message(STATUS "CMAKE_CXX_FLAGS += -fPIC")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

		# UNIX needs pthread
		set(THREADS_PREFER_PTHREAD_FLAG ON)
	endif() # GNU or Clang compiler
endif() # UNIX


### shared library: GPUAbstractionLayer ###
if(BUILD_CUDA_SUPPORT) # TODO: better separation of files?
	cuda_add_library(GPUAbstractionLayer SHARED ${SOURCES} ${SOURCES_CUDA})
else()
	# Handle .cu-files as typical c++ files
	set_source_files_properties(${SOURCES_CUDA} PROPERTIES LANGUAGE CXX)
	add_library(GPUAbstractionLayer SHARED ${SOURCES} ${SOURCES_CUDA})
endif()

if(BUILD_OPENCL_SUPPORT)
	# TODO: Load OpenCL and Boost only, if they are available
	target_link_libraries(GPUAbstractionLayer ${OpenCL_LIBRARIES} ${Boost_LIBRARIES}) # ${CUDA_LIBRARIES} linked by cuda_add_library
endif()


INSTALL(TARGETS GPUAbstractionLayer
ARCHIVE DESTINATION "lib/"
LIBRARY DESTINATION "bin/"
RUNTIME DESTINATION "bin/"
)

macro(INSTALL_HEADERS_WITH_DIRECTORY HEADER_LIST)
	foreach(HEADER ${${HEADER_LIST}})
		string(REGEX MATCH "(.*)[/]" DIR ${HEADER})
		install(FILES include/${HEADER} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${DIR})
	endforeach(HEADER)
endmacro(INSTALL_HEADERS_WITH_DIRECTORY)

INSTALL_HEADERS_WITH_DIRECTORY(HEADER_FILES)
if (BUILD_CUDA_SUPPORT)
	INSTALL_HEADERS_WITH_DIRECTORY(HEADER_FILES_CUDA)
endif(BUILD_CUDA_SUPPORT)
