cdef extern from "nitf/System.h":
    ctypedef struct nitf_Error:
        char* message
        char* file
        int line
        char* func
        int level
