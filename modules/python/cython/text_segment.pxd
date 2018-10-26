from types cimport *
from header cimport nitf_TextSubheader


cdef extern from "nitf/TextSegment.h":
    ctypedef struct nitf_TextSegment:
        nitf_TextSubheader *subheader
        nitf_Uint64 offset
        nitf_Uint64 end
