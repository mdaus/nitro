from types cimport *

cdef extern from "nitf/DataSource.h":
    ctypedef struct nitf_DataSource:
        pass

    void nitf_DataSource_destruct(nitf_DataSource** dataSource)

cdef extern from "nitf/BandSource.h":
    ctypedef struct nitf_BandSource:
        pass

    nitf_BandSource* nitf_MemorySource_construct(const void* data, nitf_Off size, nitf_Off start, int numBytesPerPixel, int pixelSkip, nitf_Error* error);

cdef extern from "nitf/ImageSource.h":
    ctypedef struct nitf_ImageSource:
        pass

    nitf_ImageSource * nitf_ImageSource_construct(nitf_Error* error)
    void nitf_ImageSource_destruct(nitf_ImageSource**)
    NITF_BOOL nitf_ImageSource_addBand(nitf_ImageSource* imageSource, nitf_BandSource* bandSource, nitf_Error* error)
    nitf_BandSource * nitf_ImageSource_getBand(nitf_ImageSource* imageSource, int n, nitf_Error* error);
