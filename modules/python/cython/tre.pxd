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
    void nitf_TRE_destruct(nitf_TRE* tre)
    int nitf_TRE_getCurrentSize(nitf_TRE* tre, nitf_Error* error)
    NITF_BOOL nitf_TRE_exists(nitf_TRE* tre, const char* tag)
    nitf_Field* nitf_TRE_getField(nitf_TRE* tre, const char* tag)
    NITF_BOOL nitf_TRE_setField(nitf_TRE* tre, const char* tag, NITF_DATA* data, size_t dataLength, nitf_Error* error)
    nitf_TREEnumerator* nitf_TRE_begin(nitf_TRE* tre, nitf_Error* error)


cdef extern from "nitf/Extensions.h":
    ctypedef struct nitf_Extensions:
        pass

    ctypedef struct nitf_ExtensionsIterator:
        pass

    NITF_BOOL nitf_Extensions_appendTRE(nitf_Extensions * ext, nitf_TRE * tre, nitf_Error * error)
    nitf_List* nitf_Extensions_getTREsByName(nitf_Extensions * ext, const char *name)
    void nitf_Extensions_removeTREsByName(nitf_Extensions* ext, const char* name)
    NITF_BOOL nitf_Extensions_exists(nitf_Extensions * ext, const char *name)
    nitf_ExtensionsIterator nitf_Extensions_begin(nitf_Extensions *ext)
    nitf_ExtensionsIterator nitf_Extensions_end(nitf_Extensions *ext)
    void nitf_ExtensionsIterator_increment(nitf_ExtensionsIterator *extIt)
    nitf_TRE* nitf_ExtensionsIterator_get(nitf_ExtensionsIterator *extIt)
    NITF_BOOL nitf_ExtensionsIterator_notEqualTo(nitf_ExtensionsIterator *it1, nitf_ExtensionsIterator *it2)
