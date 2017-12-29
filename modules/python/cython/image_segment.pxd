from types cimport *
from header cimport nitf_ImageSubheader


cdef extern from "nitf/ImageSegment.h":
    ctypedef struct nitf_ImageSegment:
        nitf_ImageSubheader *subheader
        nitf_Uint64 imageOffset
        nitf_Uint64 imageEnd

    void nitf_ImageSegment_destruct(nitf_ImageSegment ** segment);
