from io cimport nitf_IOHandle
from types cimport *

cdef extern from "nitf/DataSource.h":
    ctypedef struct nitf_DataSource:
        pass

    void nitf_DataSource_destruct(nitf_DataSource** dataSource)

cdef extern from "nitf/SegmentSource.h":
    ctypedef struct nitf_SegmentSource:
        pass

    nitf_SegmentSource* nitf_SegmentMemorySource_construct(const char* data, nitf_Off size, nitf_Off start, int byteSkip, NITF_BOOL copyData, nitf_Error* error)
    nitf_SegmentSource* nitf_SegmentFileSource_construct(nitf_IOHandle handle, nitf_Off start, int byteSkip, nitf_Error* error)

