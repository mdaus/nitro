cimport header
cimport record
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer
from error cimport nitf_Error
from dataextension_segment cimport *

from . import header
from .error import NitfError


cdef class DESegment:
    cdef nitf_DESegment* _c_des

    @staticmethod
    def from_record(c):
        cdef record.nitf_Record* c_rec
        cdef nitf_Error error
        assert(PyCapsule_IsValid(c, "Record"))
        c_rec = <record.nitf_Record*>PyCapsule_GetPointer(c, "Record")
        obj = DESegment()
        obj._c_des = record.nitf_Record_newDataExtensionSegment(c_rec, &error)
        if obj._c_des is NULL:
            raise NitfError(error)
        return obj  # allow chaining

    @staticmethod
    cdef from_ptr(nitf_DESegment* ptr):
        obj = DESegment()
        obj._c_des = ptr
        return obj

    @property
    def subheader(self):
        cdef header.nitf_DESubheader* hdr
        hdr = self._c_des.subheader
        h = header.DESubheader(PyCapsule_New(hdr, "DESubheader", NULL))
        return h
