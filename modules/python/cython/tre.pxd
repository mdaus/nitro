from error cimport nitf_Error
from field cimport nitf_Field
from types cimport *

cdef extern from "nitf/TRE.h":
    cdef int NITF_MAX_TAG

    ctypedef nitf_Pair* (*NITF_TRE_ITERATOR_INCREMENT)(nitf_TREEnumerator*, nitf_Error*)
    ctypedef NITF_BOOL (*NITF_TRE_ITERATOR_HAS_NEXT)(nitf_TREEnumerator**)
    ctypedef const char* (*NITF_TRE_ITERATOR_GET_DESCRIPTION)(nitf_TREEnumerator*, nitf_Error*)

    ctypedef struct nitf_TREEnumerator:
        NITF_TRE_ITERATOR_INCREMENT next
        NITF_TRE_ITERATOR_HAS_NEXT hasNext
        NITF_TRE_ITERATOR_GET_DESCRIPTION getFieldDescription
        NITF_DATA* data

    ctypedef struct nitf_TRE:
        void* handler
        NITF_DATA* priv
        char* tag

    nitf_TRE* nitf_TRE_construct(const char* tag, const char* id, nitf_Error* error)
    int nitf_TRE_getCurrentSize(nitf_TRE* tre, nitf_Error* error)
    NITF_BOOL nitf_TRE_exists(nitf_TRE* tre, const char* tag)
    nitf_Field* nitf_TRE_getField(nitf_TRE* tre, const char* tag)
    NITF_BOOL nitf_TRE_setField(nitf_TRE* tre, const char* tag, NITF_DATA* data, size_t dataLength, nitf_Error* error)
    nitf_TREEnumerator* nitf_TRE_begin(nitf_TRE* tre, nitf_Error* error)
