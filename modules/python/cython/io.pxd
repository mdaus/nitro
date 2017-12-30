from error cimport nitf_Error
from image_source cimport nitf_ImageSource
from record cimport nitf_Record
from types cimport NITF_BOOL


cdef extern from "nitf/System.h":
    ctypedef struct nitf_IOHandle:
        pass

    ctypedef struct nrt_HashTable:
        pass

    ctypedef enum nitf_AccessFlags:
        pass

    ctypedef enum nitf_CreationFlags:
        pass

    nitf_IOHandle nitf_IOHandle_create(const char *fname, nitf_AccessFlags access, nitf_CreationFlags creation, nitf_Error* error)

cdef inline is_iohandle_valid(nitf_IOHandle handle):
    return handle != 0xffffffff


cdef extern from "nitf/ImageWriter.h":
    ctypedef struct nitf_ImageWriter:
        pass

    NITF_BOOL nitf_ImageWriter_attachSource(nitf_ImageWriter* writer, nitf_ImageSource* imageSource, nitf_Error* error)


cdef extern from "nitf/Writer.h":
    ctypedef struct nitf_Writer:
        pass
    nitf_Writer* nitf_Writer_construct(nitf_Error* error)
    void nitf_Writer_destruct(nitf_Writer** writer)
    NITF_BOOL nitf_Writer_prepare(nitf_Writer* writer, nitf_Record* record, nitf_IOHandle ioHandle, nitf_Error* error)
    NITF_BOOL nitf_Writer_write(nitf_Writer* writer, nitf_Error* error)
    void nitf_IOHandle_close(nitf_IOHandle handle)
    nitf_ImageWriter* nitf_Writer_newImageWriter(nitf_Writer* writer, int index, nrt_HashTable* options, nitf_Error* error);


cdef extern from "nitf/Reader.h":
    ctypedef struct nitf_Reader:
        pass
    nitf_Reader* nitf_Reader_construct(nitf_Error* error)
    void nitf_Reader_destruct(nitf_Reader** writer)
    nitf_Record* nitf_Reader_read(nitf_Reader* reader, nitf_IOHandle inputHandle, nitf_Error* error)

