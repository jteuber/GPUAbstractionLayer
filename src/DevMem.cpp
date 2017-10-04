#include "DevMem.h"

#ifdef USE_CUDA
Cuda* GPUUser::m_pCuda = NULL;
#endif // USE_CUDA
#ifdef USE_OPENCL
OpenCL* GPUUser::m_pOpenCL = NULL;
#endif // USE_OPENCL
