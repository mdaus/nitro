from libc.stdint cimport *
from error cimport nitf_Error

cdef extern from "nitf/Types.h":
    ctypedef void NITF_DATA

cdef extern from "nitf/System.h":
    ctypedef uint8_t nitf_Uint8
    ctypedef uint16_t nitf_Uint16
    ctypedef uint32_t nitf_Uint32
    ctypedef uint64_t nitf_Uint64
    ctypedef int8_t nitf_Int8
    ctypedef int16_t nitf_Int16
    ctypedef int32_t nitf_Int32
    ctypedef int64_t nitf_Int64
    ctypedef int NITF_BOOL
    ctypedef struct nitf_List:
        pass

    ctypedef struct nitf_ListIterator:
        pass

    nitf_ListIterator nitf_List_begin(nitf_List * chain)
    NITF_BOOL nitf_ListIterator_equals(nitf_ListIterator * it1, nitf_ListIterator * it2)
    nitf_ListIterator nitf_List_end(nitf_List * this_chain)
    nitf_Uint32 nitf_List_size(nitf_List * list)
    NITF_DATA * nitf_List_get(nitf_List * list, int index, nitf_Error * error)
    void nitf_ListIterator_increment(nitf_ListIterator * this_iter)
    NITF_DATA * nitf_ListIterator_get(nitf_ListIterator * this_iter)

