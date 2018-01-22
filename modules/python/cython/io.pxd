from error cimport nitf_Error
from image_source cimport nitf_ImageSource
from segment_source cimport nitf_SegmentSource
from record cimport nitf_Record
from types cimport *


cdef extern from "nitf/System.h":
    ctypedef struct nitf_IOHandle:
        pass

    ctypedef struct nrt_HashTable:
        pass

    ctypedef enum nitf_AccessFlags:
        pass

    ctypedef enum nitf_CreationFlags:
        pass

    ctypedef struct nitf_IOInterface:
        pass

    nitf_IOHandle nitf_IOHandle_create(const char *fname, nitf_AccessFlags access, nitf_CreationFlags creation, nitf_Error* error)

cdef inline is_iohandle_valid(nitf_IOHandle handle):
    return handle != 0xffffffff


cdef extern from "nitf/ImageWriter.h":
    ctypedef struct nitf_ImageWriter:
        pass

    NITF_BOOL nitf_ImageWriter_attachSource(nitf_ImageWriter* writer, nitf_ImageSource* imageSource, nitf_Error* error)


cdef extern from "nitf/SegmentWriter.h":
    ctypedef struct nitf_SegmentWriter:
        pass

    NITF_BOOL nitf_SegmentWriter_attachSource(nitf_SegmentWriter* writer, nitf_SegmentSource* segmentSource, nitf_Error* error)


cdef extern from "nitf/SubWindow.h":
    ctypedef struct nitf_SubWindow:
        nitf_Uint32 startRow
        nitf_Uint32 startCol
        nitf_Uint32 numRows
        nitf_Uint32 numCols
        nitf_Uint32 *bandList
        nitf_Uint32 numBands

    nitf_SubWindow* nitf_SubWindow_construct(nitf_Error* error)
    void nitf_SubWindow_destruct(nitf_SubWindow** subWindow)


cdef extern from "nitf/ImageReader.h":
    ctypedef struct nitf_ImageReader:
        nitf_ImageIO *imageDeblocker

    NITF_BOOL nitf_ImageReader_read(nitf_ImageReader* imageReader, nitf_SubWindow* subWindow, nitf_Uint8** user, int* padded, nitf_Error* error)


cdef extern from "nitf/Writer.h":
    ctypedef struct nitf_Writer:
        pass
    nitf_Writer* nitf_Writer_construct(nitf_Error* error)
    void nitf_Writer_destruct(nitf_Writer** writer)
    NITF_BOOL nitf_Writer_prepare(nitf_Writer* writer, nitf_Record* record, nitf_IOHandle ioHandle, nitf_Error* error)
    NITF_BOOL nitf_Writer_write(nitf_Writer* writer, nitf_Error* error)
    void nitf_IOHandle_close(nitf_IOHandle handle)
    nitf_ImageWriter* nitf_Writer_newImageWriter(nitf_Writer* writer, int index, nrt_HashTable* options, nitf_Error* error);
    nitf_SegmentWriter* nitf_Writer_newDEWriter(nitf_Writer *writer, int index, nitf_Error* error);


cdef extern from "nitf/Reader.h":
    ctypedef struct nitf_Reader:
        nitf_List *warningList
        nitf_IOInterface* input
        nitf_Record *record
        NITF_BOOL ownInput

    nitf_Reader* nitf_Reader_construct(nitf_Error* error)
    void nitf_Reader_destruct(nitf_Reader** writer)
    nitf_Record* nitf_Reader_read(nitf_Reader* reader, nitf_IOHandle inputHandle, nitf_Error* error)
    nitf_ImageReader* nitf_Reader_newImageReader(nitf_Reader* reader, int imageSegmentNumber, nrt_HashTable* options, nitf_Error* error)


cdef extern from "nitf/ImageIO.h":
    ctypedef struct nitf_ImageIO:
        pass

    nitf_Uint32 nitf_ImageIO_pixelSize(nitf_ImageIO * nitf)