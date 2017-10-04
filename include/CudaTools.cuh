/*
 * File:   CudaTools.cuh
 *
 * Created on July 21, 2012, 1:23 PM
 */
#ifndef CUDATOOLS_CUH
#define	CUDATOOLS_CUH

#include "GPUAbstractionLayer_global.h"

//#include "GridStructs.h"


GAL_EXPORT void thrustScan( unsigned int* dOut, unsigned int* dIn, size_t numElements );
GAL_EXPORT unsigned int thrustReduce( unsigned int* dIn, size_t numElements );

GAL_EXPORT void sortUIntUInt( unsigned int* dKeys, unsigned int* dValues, size_t numElements );
GAL_EXPORT void sortFloatUInt( float* dKeys, unsigned int* dValues, size_t numElements );


#endif	/* CUDATOOLS_CUH */
