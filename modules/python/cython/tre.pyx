#cython: language_level=3, c_string_type=unicode, c_string_encoding=ascii

cimport field
from tre cimport *
cimport types
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer
from error cimport nitf_Error

import cython
import binascii
from collections import OrderedDict
from deprecated import deprecated
from .error import NitfError
from . import field, types


def split_key(k):
    tmp = k.split('[')
    nums = []
    val = tmp[0]
    for idx in range(1, len(tmp)):
        nums.append(int(tmp[idx][:-1]))
    return val, nums


cdef class Extensions:
    cdef nitf_Extensions* _c_extensions

    def __cinit__(self, capsule=None):
        self._c_extensions = NULL

        if capsule is not None:
            if not PyCapsule_IsValid(capsule, "Extensions"):
                raise TypeError("Invalid C pointer type")
            self._c_extensions = <nitf_Extensions*>PyCapsule_GetPointer(capsule, "Extensions")

    @staticmethod
    cdef from_ptr(nitf_Extensions* ptr):
        obj = Extensions()
        obj._c_extensions = ptr
        return obj

    def append(self, TRE tre not None):
        cdef nitf_Error error
        if not nitf_Extensions_appendTRE(self._c_extensions, tre._c_tre, &error):
            raise NitfError(error)

    def extend(self, it):
        for tre in it:
            self.append(tre)

    def todict(self, encoding):
        rval = OrderedDict()
        for tre in iter(self):
            rval.setdefault(tre.tag, []).append(tre.todict(encoding))
        return rval
        
    def __contains__(self, str item):
        return <bint>nitf_Extensions_exists(self._c_extensions, item)

    def __getitem__(self, str item not None):
        cdef nitf_List* tres
        tres = nitf_Extensions_getTREsByName(self._c_extensions, item)
        if tres is NULL:
            return []
        return types.List(TRE, PyCapsule_New(tres, "List", NULL))

    def __delitem__(self, str key not None):
        nitf_Extensions_removeTREsByName(self._c_extensions, key)

    def __getattr__(self, str item not None):
        return self[item]

    def __delattr__(self, str item not None):
        del self[item]

    def __iter__(self):
        cdef nitf_Error error
        cdef nitf_ExtensionsIterator it
        cdef nitf_ExtensionsIterator end
        cdef nitf_TRE* tre_ptr

        it = nitf_Extensions_begin(self._c_extensions)
        end = nitf_Extensions_end(self._c_extensions)

        while nitf_ExtensionsIterator_notEqualTo(&it, &end):
            tre_ptr = nitf_ExtensionsIterator_get(&it)
            if tre_ptr is not NULL:
                yield TRE.from_ptr(tre_ptr)
            nitf_ExtensionsIterator_increment(&it)

    @deprecated("Old SWIG API")
    def removeTREsByName(self, tag):
        del self[tag]


cdef class TRE:
    cdef nitf_TRE* _c_tre

    @cython.nonecheck(False)
    def __cinit__(self, str tag=None, str id=None, capsule=None):
        cdef nitf_Error error

        if capsule is not None:  # do not create a new tag, use an existing pointer
            self._c_tre = <nitf_TRE*>PyCapsule_GetPointer(capsule, "TRE")
            return
        self._c_tre = NULL
        if tag is not None:
            if id is None:
                self._c_tre = nitf_TRE_construct(tag, <char*>NULL, &error)
            else:
                self._c_tre = nitf_TRE_construct(tag, id, &error)
            if self._c_tre is NULL:
                raise NitfError(error)

    @staticmethod
    def from_capsule(c):
        assert(PyCapsule_IsValid(c, "TRE"))
        return TRE.from_ptr(<nitf_TRE*>PyCapsule_GetPointer(c, "TRE"))

    @staticmethod
    cdef from_ptr(nitf_TRE* ptr):
        obj = TRE()
        obj._c_tre = ptr
        return obj

    def _capsule(self):
        return PyCapsule_New(self._c_tre, "TRE", NULL)

    def clone(self):
        cdef nitf_TRE* ntre
        cdef nitf_Error error

        ntre = nitf_TRE_clone(self._c_tre, &error)
        if ntre == NULL:
            raise NitfError(error)
        return TRE.from_ptr(ntre)

    @property
    def tag(self):
        return <str>self._c_tre.tag

    def __len__(self):
        cdef nitf_Error error

        sz = nitf_TRE_getCurrentSize(self._c_tre, &error)
        if sz < 0:
            raise NitfError(error)
        return sz

    def __str__(self):
        rval = f'-----\n{self.tag}({len(self)})\n\t'
        rval += '\n\t'.join([f'{k} = {v}' for k,v in self])
        rval += '\n'
        return rval

    def __convert_item(self, val, encoding):
        if isinstance(val, field.Field):
            rval = val.get_pyvalue()
            if encoding is not None and isinstance(rval, bytes):
                try:
                    rval = rval.decode(encoding)
                except UnicodeDecodeError:
                    # invalid character...try quoting it
                    rval = binascii.b2a_qp(rval).decode(encoding)
        elif isinstance(val, list):
            rval = [self.__convert_item(litem, encoding) for litem in val]
        else:
            try:
                rval = val.todict(encoding)
            except AttributeError:
                rval = str(val)  # fall back to the printable string
        return rval

    def todict(self, encoding=None):
        rval = OrderedDict()
        loop_names = []
        for key, val in iter(self):
            key, indices = split_key(key)
            if len(indices) < len(loop_names):
                loop_names = loop_names[:len(indices)]
            tmp = rval
            for j, idx in enumerate(indices):
                if j > len(loop_names) - 1:
                    loop_names.append(list(tmp.keys())[-1])
                if loop_names[j] not in tmp or type(tmp[loop_names[j]]) != list:
                    tmp[loop_names[j]] = []
                tmp = tmp[loop_names[j]]
                if len(tmp) < idx + 1:
                    tmp.append(OrderedDict())
                tmp = tmp[idx]
            tmp[key] = self.__convert_item(val, encoding)
        return rval

    def __contains__(self, str item):
        return <bint>nitf_TRE_exists(self._c_tre, item)

    def __getitem__(self, str item not None):
        cdef field.nitf_Field* fld
        fld = nitf_TRE_getField(self._c_tre, item)
        if fld is NULL:
            raise NitfError('Field %s not found' % item)
        return field.Field(PyCapsule_New(fld, "Field", NULL))

    def __getattr__(self, item):
        return self[item]

    def __setitem__(self, str key not None, value):
        cdef nitf_Error error
        cdef char* tmp

        fld = self[key]
        if fld.type == field.FieldType.NITF_BINARY and len(value) < len(fld):
            # we need to pad the binary
            amt = len(fld) - len(value)
            value = value + amt * b' '
        tmp = <char*>value
        if not nitf_TRE_setField(self._c_tre, key, <NITF_DATA*>tmp, len(value), &error):
            raise NitfError(error)

    def __setattr__(self, key, value):
        self[key] = value

    @property
    def raw_data(self):
        return self['raw_data']

    def set_raw_data(self, value):
        cdef nitf_Error error
        cdef char* tmp

        tmp = <char*>value
        if not nitf_TRE_setField(self._c_tre, 'raw_data', <NITF_DATA*>tmp, len(value), &error):
            raise NitfError(error)

    def __iter__(self):
        cdef nitf_Error error
        cdef nitf_TREEnumerator* it
        cdef nitf_Pair* pair
        cdef field.nitf_Field* fld

        it = nitf_TRE_begin(self._c_tre, &error)
        if it is NULL:
            raise NitfError(error)
        while it.hasNext(&it):
            pair = it.next(it, &error)
            if pair is NULL:
                raise NitfError(error)
            fld = <field.nitf_Field*>pair.data
            if fld is not NULL:
                yield pair.key, field.Field(PyCapsule_New(fld, "Field", NULL))

    @deprecated("Old SWIG API")
    def getTag(self):
        return self.tag

    @deprecated("Old SWIG API")
    def getCurrentSize(self):
        return len(self)

    @deprecated("Old SWIG API")
    def getField(self, name):
        return self[name]

    @deprecated("Old SWIG API")
    def setField(self, name, value):
        self[name] = value

