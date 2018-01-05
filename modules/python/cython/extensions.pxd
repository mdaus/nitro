from error cimport nitf_Error
from tre cimport nitf_TRE
from types cimport *

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
