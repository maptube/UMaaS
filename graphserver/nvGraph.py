"""
Header file for nvGraph library.
Contains data structures and SSSP functions, which are the only ones we use.
"""

import ctypes as c
from enum import Enum
import platform

class nvgraphStatus(Enum):
    #all NVGRAPH_STATUS_* in docs
    #nvgraph.h
    #typedef enum{NVGRAPH_STATUS_SUCCESS=0,NVGRAPH_STATUS_NOT_INITIALIZED=1,NVGRAPH_STATUS_ALLOC_FAILED=2,NVGRAPH_STATUS_INVALID_VALUE=3,NVGRAPH_STATUS_ARCH_MISMATCH=4,NVGRAPH_STATUS_MAPPING_ERROR=5,NVGRAPH_STATUS_EXECUTION_FAILED=6,NVGRAPH_STATUS_INTERNAL_ERROR=7,NVGRAPH_STATUS_TYPE_NOT_SUPPORTED=8,NVGRAPH_STATUS_NOT_CONVERGED=9} nvgraphStatus_t;
    Success = c.c_uint(0)
    NotInitialized = c.c_uint(1)
    AllocFailed = c.c_uint(2)
    InvalidValue = c.c_uint(3)
    ArchMismatch = c.c_uint(4)
    MappingError = c.c_uint(5)
    ExecutionFailed = c.c_uint(6)
    InternalError = c.c_uint(7)
    TypeNotSupported = c.c_uint(8)
    NotConverged = c.c_uint(9)

class nvgraphTopologyType(Enum):
    #typedef enum {   NVGRAPH_CSR_32 = 0,   NVGRAPH_CSC_32 = 1, } nvgraphTopologyType_t;
    CSR_32 = c.c_uint(0)
    CSC_32 = c.c_uint(1)
    COO_32 = c.c_uint(2)

#struct nvgraphCSRTopology32I_st {  int nvertices;  int nedges;  int *source_offsets;  int *destination_indices; }; typedef struct nvgraphCSRTopology32I_st *nvgraphCSRTopology32I_t;
class nvgraphCSRTopology32I_st(c.Structure):
    _fields_ = [
        ("nvertices", c.c_int),
        ("nedges", c.c_int),
        ("source_offsets", c.POINTER(c.c_int)),
        ("destination_indices", c.POINTER(c.c_int))
    ]


#struct nvgraphCSCTopology32I_st {  int nvertices;  int nedges;  int *destination_offsets;  int *source_indices; }; typedef struct nvgraphCSCTopology32I_st *nvgraphCSCTopology32I_t;
class nvgraphCSCTopology32I_st(c.Structure):
    _fields_ = [
        ("nvertices", c.c_int),
        ("nedges", c.c_int),
        ("destination_offsets", c.POINTER(c.c_int)),
        ("source_indices", c.POINTER(c.c_int))            
        ]


#Opaque structure holding nvGRAPH library context */
#struct nvgraphContext; typedef struct nvgraphContext *nvgraphHandle_t;
class nvgraphContext(c.Structure):
    pass
nvgraphHandle_t = c.POINTER(nvgraphContext) # opaque struct

#Opaque structure holding the graph descriptor */
#struct nvgraphGraphDescr; typedef struct nvgraphGraphDescr *nvgraphGraphDescr_t;
class nvgraphGraphDescr(c.Structure):
    pass
nvgraphDescr_t = c.POINTER(nvgraphGraphDescr) # opaque struct

class cudaDataType(Enum):
    CUDA_R_16F = 2 # 16 bit real
    CUDA_C_16F = 6 # 16 bit complex
    CUDA_R_32F = 0 # 32 bit real
    CUDA_C_32F = 4 # 32 bit complex
    CUDA_R_64F = 1 # 64 bit real
    CUDA_C_64F = 5 # 64 bit complex
    CUDA_R_8I = 3  # 8 bit real as a signed integer
    CUDA_C_8I = 7  # 8 bit complex as a pair of signed integers
    CUDA_R_8U = 8  # 8 bit real as a signed integer
    CUDA_C_8U = 9  # 8 bit complex as a pair of signed integers


################################################################################
#Function pointers into nvgraph dll

#NOTE: dll is in D:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin\nvgraph64_90.dll on windows
#this MUST be on the path
#For Linux RHEL, /usr/local/cuda-10.1/bin and /usr/local/cuda/bin
#File is in /usr/local/cuda/lib64/libnvgraph.so

print("Platform: ",platform.system())
if platform.system()=="Windows": #should return Windows or Linux
    lib_nvGraph = "nvgraph64_10.dll" #"nvgraph64_90.dll" #Windows - how do you do multiple version detection?
else:
    lib_nvGraph = "libnvgraph.so" #Linux

#nvgraphPagerank = c.CDLL(lib_nvGraph).nvgraphPagerank #don't need this

#nvgraphStatus_t nvgraphSssp(nvgraphHandle_t handle, const nvgraphGraphDescr_t descrG, const size_t weight_index, const int *source_vert, const size_t sssp_index);
nvgraphSssp = c.CDLL(lib_nvGraph).nvgraphSssp

nvgraphCreate = c.CDLL(lib_nvGraph).nvgraphCreate

nvgraphCreateGraphDescr = c.CDLL(lib_nvGraph).nvgraphCreateGraphDescr

#nvgraphStatus_t nvgraphSetGraphStructure(nvgraphHandle_t handle, nvgraphGraphDescr_t descrG, void* topologyData, nvgraphTopologyType_t TType);
#nvgraphStatus nvgraphSetGraphStructure(nvgraphContext handle, nvgraphGraphDescr descrG, ref nvgraphCSCTopology32I topologyData, nvgraphTopologyType TType);
nvgraphSetGraphStructure = c.CDLL(lib_nvGraph).nvgraphSetGraphStructure

#nvgraphStatus_t nvgraphAllocateVertexData(nvgraphHandle_t handle, nvgraphGraphDescr_t descrG, size_t numsets, cudaDataType_t *settypes);
nvgraphAllocateVertexData = c.CDLL(lib_nvGraph).nvgraphAllocateVertexData

#nvgraphStatus_t nvgraphAllocateEdgeData(nvgraphHandle_t handle, nvgraphGraphDescr_t descrG, size_t numsets, cudaDataType_t *settypes);
nvgraphAllocateEdgeData = c.CDLL(lib_nvGraph).nvgraphAllocateEdgeData

#nvgraphStatus_t nvgraphSetEdgeData(nvgraphHandle_t handle, nvgraphGraphDescr_t descrG, void *edgeData, size_t setnum);
nvgraphSetEdgeData = c.CDLL(lib_nvGraph).nvgraphSetEdgeData

#nvgraphStatus_t nvgraphGetVertexData(nvgraphHandle_t handle, nvgraphGraphDescr_t descrG, void *vertexData, size_t setnum);
nvgraphGetVertexData = c.CDLL(lib_nvGraph).nvgraphGetVertexData

#nvgraphStatus_t nvgraphDestroy(nvgraphHandle_t handle);
nvgraphDestroy = c.CDLL(lib_nvGraph).nvgraphDestroy

#nvgraphStatus_t nvgraphDestroyGraphDescr(nvgraphHandle_t handle, nvgraphGraphDescr_t descrG);
nvgraphDestroyGraphDescr = c.CDLL(lib_nvGraph).nvgraphDestroyGraphDescr

################################################################################

def check_status(msg,status):
    if status!=0:
        print("ERROR: ", msg, " status=",status)

################################################################################

