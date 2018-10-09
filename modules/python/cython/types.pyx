#cython: language_level=3, c_string_type=unicode, c_string_encoding=ascii

cimport field
cimport types
from error cimport nitf_Error
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer
from .error import NitfError
from . import field


cdef class ListIter:
    cdef types.nitf_ListIterator _c_iter
    cdef types.nitf_List* _c_list
    cdef object _list

    def __cinit__(self):
        self._list = None
        self._c_list = NULL

    @staticmethod
    cdef from_ptr(lst, types.nitf_List* c_list):
        obj = ListIter()
        obj._list = lst
        obj._c_list = c_list
        obj._c_iter = types.nitf_List_begin(obj._c_list)
        return obj

    def __next__(self):
        cdef types.nitf_ListIterator end
        cdef NITF_DATA* data
        end = types.nitf_List_end(self._c_list)
        if types.nitf_ListIterator_equals(&self._c_iter, &end):
            raise StopIteration()
        data = types.nitf_ListIterator_get(&self._c_iter)
        types.nitf_ListIterator_increment(&self._c_iter)
        return NitfData.from_ptr(data).convert(self._list.container_type)


cdef class List:
    cdef types.nitf_List* _c_list
    cpdef type _container_type

    @property
    def container_type(self):
        return self._container_type

    def __cinit__(self, container_type, capsule=None):
        self._c_list = NULL
        self._container_type = container_type

        if capsule is not None:
            if not PyCapsule_IsValid(capsule, "List"):
                raise TypeError("Invalid C pointer type")
            self._c_list = <types.nitf_List*>PyCapsule_GetPointer(capsule, "List")

    cdef from_ptr(self, types.nitf_List* ptr):
        self._c_list = ptr
        return self

    def __len__(self):
        return types.nitf_List_size(self._c_list)

    def __getitem__(self, item):
        cdef types.nitf_Error error
        cdef NITF_DATA* rval
        if item > (len(self)-1) or item < 0:
            raise IndexError()
        rval = types.nitf_List_get(self._c_list, <int>item, &error)
        if rval is NULL:
            raise NitfError(error)
        return NitfData.from_ptr(rval).convert(self._container_type)

    def __iter__(self):
        return ListIter.from_ptr(self, self._c_list)

    def append(self, obj):
        cdef types.nitf_Error error
        cdef NITF_DATA* data

        c = obj.to_capsule()
        assert(PyCapsule_IsValid(c, "Field"))
        data = <NITF_DATA*>PyCapsule_GetPointer(c, "Field")
        if not nitf_List_pushBack(self._c_list, data, &error):
            raise NitfError(error)


cdef class NitfData:
    cdef NITF_DATA* _c_data

    def __cinit__(self):
        self._c_data = NULL

    @staticmethod
    cdef from_ptr(NITF_DATA* data):
        obj = NitfData()
        obj._c_data = data
        return obj

    cpdef convert(self, container_type):
        rval = None
        if container_type is str:
            rval = <char*>self._c_data
        elif container_type is bytes:
            rval = <char*>self._c_data
            rval = rval.encode(errors='replace')
        elif container_type is field.Field:
            rval = field.Field(PyCapsule_New(self._c_data, "Field", NULL))
        elif hasattr(container_type, 'from_capsule'):
            rval = container_type.from_capsule(PyCapsule_New(self._c_data, container_type.__name__, NULL))
        else:
            rval = <unsigned long long>self._c_data
        return rval

    def __str__(self):
        return self.convert(str)

