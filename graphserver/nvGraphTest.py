"""
Test of SSSP shortest paths in nvGraph

read this...
https://docs.python.org/3/library/ctypes.html#module-ctypes
"""

import ctypes as c
from enum import Enum

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

lib_nvGraph = "nvgraph64_10.dll" #"nvgraph64_90.dll" #Windows
#lib_nvGraph = "libnvgraph.so" #Linux

nvgraphPagerank = c.CDLL(lib_nvGraph).nvgraphPagerank

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


###main program - from TestSSSP in nvGraph.cs
n = 6
nnz = 10
vertex_numsets = 1 #c.c_size_t(3)
edge_numsets = 1 #c.c_size_t(1)

#init host data
#sssp_1 = c.c_float(n) #sssp_1_h = (float*)malloc(n*sizeof(float));
#sssp_1_h = c.pointer(sssp_1)
##
sssp_1 = [0.0,0.0,0.0,0.0,0.0,0.0]
sssp_1_seq = c.c_float * n
sssp_1_h = sssp_1_seq(*sssp_1)


#void** vertex_dim;
#vertex_dim = int(vertex_numsets)
#vertex_dim = c.c_void_p * 3
vertex_dimT = [cudaDataType.CUDA_R_32F.value]
vertex_dimT_seq = c.c_int * len(vertex_dimT)
vertex_dimT_h = vertex_dimT_seq(*vertex_dimT)
edge_dimT = [cudaDataType.CUDA_R_32F.value]
edge_dimT_seq = c.c_int * len(edge_dimT)
edge_dimT_h = edge_dimT_seq(*edge_dimT)

#note: this is the NVidia example from their SSSP example program            
weights = [ 0.333333, 0.5, 0.333333, 0.5, 0.5, 1.0, 0.333333, 0.5, 0.5, 0.5 ]
weights_seq = c.c_float * len(weights)
weights_h = weights_seq(*weights)
destination_offsets = [ c.c_int(0), c.c_int(1), c.c_int(3), c.c_int(4), c.c_int(6), c.c_int(8), c.c_int(10) ]
destination_offsets_seq = c.c_int*len(destination_offsets)
destination_offsets_h = destination_offsets_seq(*destination_offsets)
#destination_offsets_p = c.pointer(destination_offsets_h)
#destination_offsets_p.contents = destination_offsets_h

source_indices = [ 2, 0, 2, 0, 4, 5, 2, 3, 3, 4 ]
source_indices_seq = c.c_int*len(source_indices)
source_indices_h = source_indices_seq(*source_indices)
#source_indices_p = c.pointer(source_indices_h)
#source_indices_p.contents = source_indices_h

handle = nvgraphHandle_t()
handle_p = c.pointer(handle)
handle_p.contents = handle
graph = nvgraphDescr_t()
graph_p = c.pointer(graph)
graph_p.contents = graph

check_status("nvgraphCreate",nvgraphCreate(handle_p)) #now we create the graph with a graph pointer handle for the return
check_status("nvgraphCreateGraphDescr",nvgraphCreateGraphDescr(handle, graph_p)) #and then do the same with a graph descriptor handle

CSC_input = nvgraphCSCTopology32I_st() #or as params nvgraphCSCTopology32I_st(6,10,s_i_h,d_o_h)
CSC_input.nvertices = n
CSC_input.nedges = nnz
CSC_input.destination_offsets = destination_offsets_h
CSC_input.source_indices = source_indices_h
#CSC_p = c.pointer(CSC_input)
#CSC_p.contents = CSC_input
#print("CSC=",CSC_p,CSC_input)

#CSR_input = nvgraphCSRTopology32I_st()
#CSR_input.nvertices = n
#CSR_input.nedges = nnz
#CSR_input.source_offsets = destination_offsets_h
#CSR_input.destination_indices = source_indices_h

#GCHandle pin_weights_h = GCHandle.Alloc(weights_h, GCHandleType.Pinned);
#GCHandle pin_destination_offsets_h = GCHandle.Alloc(destination_offsets_h, GCHandleType.Pinned);
#GCHandle pin_source_indices_h = GCHandle.Alloc(source_indices_h, GCHandleType.Pinned);
#CSC_input.destination_offsets = pin_destination_offsets_h.AddrOfPinnedObject();
#CSC_input.source_indices = pin_source_indices_h.AddrOfPinnedObject();
#CSC_input.destination_offsets = destination_offsets_h
#CSC_input.source_indices = source_indices_h



# Set graph connectivity and properties (tranfers)
check_status("nvgraphSetGraphStructure",nvgraphSetGraphStructure(handle, graph, c.pointer(CSC_input), nvgraphTopologyType.CSC_32.value))
#success = nvgraphSetGraphStructure(handle, graph, CSR_input, c.c_int(nvgraphTopologyType.CSR_32.value))
check_status("nvgraphAllocateVertexData",nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT_h))
check_status("nvgraphAllocateEdgeData",nvgraphAllocateEdgeData(handle, graph, edge_numsets, edge_dimT_h))
check_status("nvgraphSetEdgeData",nvgraphSetEdgeData(handle, graph, weights_h, 0, nvgraphTopologyType.CSC_32.value))
# Solve
source_vert = c.c_int(0)
source_vert_h = c.pointer(source_vert)
check_status("nvgraphSssp",nvgraphSssp(handle, graph, 0,  source_vert_h, 0))
# Get and print result
check_status("nvgraphGetVertexData",nvgraphGetVertexData(handle, graph, sssp_1_h, 0, nvgraphTopologyType.CSC_32.value))
#for i in range(0,n):
#    #print(source_vert + " -> ", int(i) , " " , sssp_1[int(i)])
#    print(i,sssp_1)
print(list(sssp_1_h))



#Clean up - all variables are python managed
check_status("nvgraphDestroyGraphDescr",nvgraphDestroyGraphDescr(handle, graph))
check_status("nvgraphDestroy",nvgraphDestroy(handle))
