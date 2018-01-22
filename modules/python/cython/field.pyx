#cython: language_level=3, c_string_type=unicode, c_string_encoding=ascii

cimport field
from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from error cimport nitf_Error

import contextlib
import numpy as np
from deprecated import deprecated
from .error import NitfError


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    """Useful context manager for temporarily changing numpy print options"""
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


cdef class Field:
    cdef field.nitf_Field* _c_field
    cdef Py_ssize_t _shape[1]
    cdef Py_ssize_t _strides[1]
    cdef void* _buf_value
    cdef int _ref_count

    def __cinit__(self, capsule=None):
        self._c_field = NULL
        self._buf_value = NULL
        self._strides[0] = 1
        self._shape[0] = 0
        self._ref_count = 0

        if capsule is not None:
            if not PyCapsule_IsValid(capsule, "Field"):
                raise TypeError("Invalid C pointer type")
            self._c_field = <field.nitf_Field*>PyCapsule_GetPointer(capsule, "Field")

    @staticmethod
    cdef from_ptr(field.nitf_Field* ptr):
        obj = Field()
        obj._c_field = ptr
        return obj

    @staticmethod
    def from_capsule(c):
        assert(PyCapsule_IsValid(c, "Field"))
        return Field.from_ptr(<field.nitf_Field*>PyCapsule_GetPointer(c, "Field"))

    @property
    def type(self):
        if self._c_field is NULL:
            return FieldType(NITF_BINARY)
        return FieldType(self._c_field.type)

    def __len__(self):
        return self._c_field.length

    def __str__(self):
        if self.type in [NITF_BCS_N, NITF_BCS_A]:
            return self.get_string()
        # this generates a nice list of hex byte values truncating at 200 bytes like numpy array pretty printing
        with printoptions(threshold=200, edgeitems=100, formatter={'int_kind':lambda v: '%02x' % v}):
            return str(np.frombuffer(self.get_raw(), dtype="uint8"))

    def __repr__(self):
        return "<{} [{}]: {}>".format(self.__class__.__name__, self.type.name, self.get_string())

    @deprecated("Old SWIG API")
    def intValue(self):
        return self.get_int()

    def get_uint(self):
        cdef nitf_Error error
        cdef uint64_t val
        if not field.nitf_Field_get(self._c_field, <NITF_DATA*>&val, <field.nitf_ConvType>NITF_CONV_UINT, 8, &error):
            raise NitfError(error)
        return val

    def get_int(self):
        cdef nitf_Error error
        cdef int64_t val
        if not field.nitf_Field_get(self._c_field, <NITF_DATA*>&val, <field.nitf_ConvType>NITF_CONV_INT, 8, &error):
            raise NitfError(error)
        return val

    def get_real(self):
        cdef nitf_Error error
        cdef double val
        if not field.nitf_Field_get(self._c_field, <NITF_DATA*>&val, <field.nitf_ConvType>NITF_CONV_REAL, 8, &error):
            raise NitfError(error)
        return val

    def get_string(self):
        cdef nitf_Error error
        cdef Py_ssize_t val_length = self._c_field.length+1
        cdef char* val = <char*>PyMem_Malloc(val_length)

        if self._c_field.length <= 0:
            return ""
        try:
            if not field.nitf_Field_get(self._c_field, <NITF_DATA*>val, <field.nitf_ConvType>NITF_CONV_STRING, val_length, &error):
                raise NitfError(error)
            rval = <str>val[:val_length-1]
            return rval
        finally:
            PyMem_Free(val)

    def get_raw(self):
        return bytes(self)

    def set_uint32(self, value):
        cdef nitf_Error error
        if not field.nitf_Field_setUint32(self._c_field, value, &error):
            raise NitfError(error)

    def set_uint64(self, value):
        cdef nitf_Error error
        if not field.nitf_Field_setUint64(self._c_field, value, &error):
            raise NitfError(error)

    def set_int32(self, value):
        cdef nitf_Error error
        if not field.nitf_Field_setInt32(self._c_field, value, &error):
            raise NitfError(error)

    def set_int64(self, value):
        cdef nitf_Error error
        if not field.nitf_Field_setInt64(self._c_field, value, &error):
            raise NitfError(error)

    def set_string(self, value):
        cdef nitf_Error error
        if type(value) is str:
            value = value.encode(errors='replace')
        if not field.nitf_Field_setString(self._c_field, value, &error):
            raise NitfError(error)

    def set_real(self, value, conversion='f', include_plus=True):
        cdef nitf_Error error
        if not field.nitf_Field_setReal(self._c_field, conversion, include_plus, <double>value, &error):
            raise NitfError(error)

    def set_raw_data(self, buffer):
        cdef nitf_Error error
        if not field.nitf_Field_setRawData(self._c_field, <char*>buffer, len(buffer), &error):
            raise NitfError(error)

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef nitf_Error error
        if self._c_field is NULL:
            raise MemoryError()
        self._shape[0] = self._c_field.length
        if self._buf_value is NULL:
            self._buf_value = PyMem_Malloc(self._shape[0])
            self._ref_count = 1
        else:
            self._ref_count += 1
        buffer.buf = self._buf_value
        buffer.format = 'B'
        buffer.internal = NULL
        buffer.itemsize = 1
        buffer.len = self._shape[0]
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 1
        buffer.shape = self._shape
        buffer.strides = self._strides
        buffer.suboffsets = NULL

        if not field.nitf_Field_get(self._c_field, self._buf_value, <field.nitf_ConvType>NITF_CONV_RAW, buffer.len, &error):
            PyMem_Free(self._buf_value)
            self._buf_value = NULL
            raise NitfError(error)

    def __releasebuffer__(self, Py_buffer* buffer):
        assert(buffer.buf == self._buf_value)
        self._ref_count -= 1
        if self._ref_count == 0:
            PyMem_Free(self._buf_value)
            self._buf_value = NULL


cpdef enum FieldType:
    NITF_BCS_A
    NITF_BCS_N
    NITF_BINARY


cpdef enum ConvType:
    NITF_CONV_UINT
    NITF_CONV_INT
    NITF_CONV_REAL
    NITF_CONV_STRING
    NITF_CONV_RAW
