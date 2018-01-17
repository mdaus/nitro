from types cimport *
from header cimport nitf_DESubheader


cdef extern from "nitf/DESegment.h":
    ctypedef struct nitf_DESegment:
        nitf_DESubheader *subheader
        nitf_Uint64 offset
        nitf_Uint64 end
