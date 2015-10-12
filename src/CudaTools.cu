
#include "CudaTools.cuh"

#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>


void thrustScan(unsigned int* dOut, unsigned int* dIn, size_t numElements)
{
	thrust::exclusive_scan( thrust::device_ptr<unsigned int>(dIn), thrust::device_ptr<unsigned int>(dIn + numElements), thrust::device_ptr<unsigned int>(dOut) );
}

unsigned int thrustReduce(unsigned int* dIn, size_t numElements)
{
	return thrust::reduce( thrust::device_ptr<unsigned int>(dIn), thrust::device_ptr<unsigned int>(dIn + numElements) );
}

void sortUIntUInt( unsigned int* dKeys, unsigned int* dValues, size_t numElements )
{
	thrust::stable_sort_by_key( thrust::device_ptr<unsigned int>(dKeys), thrust::device_ptr<unsigned int>(dKeys + numElements), thrust::device_ptr<unsigned int>(dValues) );
}

//void sortUIntCells( unsigned int* dKeys, CellOnDevice* dValues, size_t numElements )
//{
//	thrust::stable_sort_by_key( thrust::device_ptr<unsigned int>(dKeys), thrust::device_ptr<unsigned int>(dKeys + numElements), thrust::device_ptr<CellOnDevice>(dValues) );
//}

void sortFloatUInt( float* dKeys, unsigned int* dValues, size_t numElements )
{
	thrust::stable_sort_by_key( thrust::device_ptr<float>(dKeys), thrust::device_ptr<float>(dKeys + numElements), thrust::device_ptr<unsigned int>(dValues) );
}

