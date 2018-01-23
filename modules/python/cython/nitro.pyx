#cython: language_level=3, c_string_type=unicode, c_string_encoding=ascii

cimport cython
cimport dataextension_segment
cimport field
cimport header
cimport image_segment
cimport image_source
cimport io
cimport record
cimport segment_source
cimport tre
cimport numpy as np
np.import_array()
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer
from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc
from error cimport nitf_Error
from types cimport *

import dataextension_segment
from . import field, header
import numpy as np
import os
import tre
import types
from deprecated import deprecated
from .error import NitfError


cdef class AccessFlags:
    NITF_ACCESS_READONLY=os.O_RDONLY
    NITF_ACCESS_WRITEONLY=os.O_WRONLY
    NITF_ACCESS_READWRITE=os.O_RDWR

cdef class CreationFlags:
    NITF_CREATE=os.O_CREAT
    NITF_OPEN_EXISTING=0x000
    NITF_TRUNCATE=os.O_TRUNC


cdef class Record:
    cdef record.nitf_Record* _c_record
    def __cinit__(self, version=NITF_VER_21):
        cdef nitf_Error error
        if version != NITF_VER_UNKNOWN:
            self._c_record = record.nitf_Record_construct(version, &error)
            if self._c_record is NULL:
                raise NitfError(error)

    def __dealloc__(self):
        if self._c_record is not NULL:
            record.nitf_Record_destruct(&self._c_record)

    @staticmethod
    cdef from_ptr(record.nitf_Record* ptr):
        obj = Record()
        obj._c_record = ptr
        return obj

    @property
    def version(self):
        return Version(record.nitf_Record_getVersion(self._c_record))

    @property
    def header(self):
        cdef header.nitf_FileHeader* hdr
        hdr = self._c_record.header
        h = header.FileHeader(PyCapsule_New(hdr, "FileHeader", NULL))
        return h

    @property
    def num_images(self):
        cdef nitf_Error error
        rval = <int>record.nitf_Record_getNumImages(self._c_record, &error)
        if rval < 0 or rval > 999:
            raise NitfError(error)
        return rval

    @deprecated("Old SWIG API")
    def getImages(self):
        return self.images

    @deprecated("Old SWIG API")
    def getImage(self, index):
        return self.get_image(index)

    @property
    def images(self):
        imgs = []
        for i in range(self.num_images):
            imgs.append(self.get_image(i))
        return imgs

    def get_image(self, index):
        cdef nitf_Error error
        cdef image_segment.nitf_ImageSegment* img
        img = <image_segment.nitf_ImageSegment*>nitf_List_get(self._c_record.images, index, &error)
        if img is NULL:
            raise NitfError(error)
        return ImageSegment.from_ptr(img)

    @property
    def num_graphics(self):
        cdef nitf_Error error
        rval = <int>record.nitf_Record_getNumGraphics(self._c_record, &error)
        if rval < 0 or rval > 999:
            raise NitfError(error)
        return rval

    @property
    def num_texts(self):
        cdef nitf_Error error
        rval = <int>record.nitf_Record_getNumTexts(self._c_record, &error)
        if rval < 0 or rval > 999:
            raise NitfError(error)
        return rval

    @property
    def num_data_extensions(self):
        cdef nitf_Error error
        rval = <int>record.nitf_Record_getNumDataExtensions(self._c_record, &error)
        if rval < 0 or rval > 999:
            raise NitfError(error)
        return rval

    @property
    def num_labels(self):
        cdef nitf_Error error
        rval = <int>record.nitf_Record_getNumLabels(self._c_record, &error)
        if rval < 0 or rval > 999:
            raise NitfError(error)
        return rval

    @property
    def num_reserved_extensions(self):
        cdef nitf_Error error
        rval = <int>record.nitf_Record_getNumReservedExtensions(self._c_record, &error)
        if rval < 0 or rval > 999:
            raise NitfError(error)
        return rval

    @deprecated("Old SWIG API")
    def newImageSegment(self):
        return self.new_image_segment()

    def new_image_segment(self):
        return ImageSegment.from_record(self._c_record)

    def new_data_extension_segment(self):
        return dataextension_segment.DESegment.from_record(PyCapsule_New(self._c_record, "Record", NULL))


cdef class ImageSegment:
    cdef image_segment.nitf_ImageSegment* _c_image

    @staticmethod
    cdef from_record(record.nitf_Record* c_rec):
        cdef nitf_Error error
        obj = ImageSegment()
        obj._c_image = record.nitf_Record_newImageSegment(c_rec, &error)
        if obj._c_image is NULL:
            raise NitfError(error)
        return obj  # allow chaining

    @staticmethod
    cdef from_ptr(image_segment.nitf_ImageSegment* ptr):
        obj = ImageSegment()
        obj._c_image = ptr
        return obj

    @property
    def subheader(self):
        cdef header.nitf_ImageSubheader* hdr
        hdr = self._c_image.subheader
        h = header.ImageSubheader(PyCapsule_New(hdr, "ImageSubheader", NULL))
        return h

    @deprecated("Old SWIG API")
    def addBands(self, num=1):
        if num < 0:
            num = 0
        self.create_bands(num)

    def create_bands(self, num):
        self.subheader.create_bands(num)


cdef class ImageSource:
    cdef image_source.nitf_ImageSource* _c_source
    cpdef _band_sources

    def __init__(self):
        self._band_sources = []

    def __cinit__(self):
        cdef nitf_Error error
        self._c_source = image_source.nitf_ImageSource_construct(&error)
        if self._c_source is NULL:
            raise NitfError(error)

    @deprecated("Old SWIG API")
    def addBand(self, band):
        return self.add_band(band)

    def add_band(self, BandSource bandsource):
        cdef nitf_Error error
        cdef image_source.nitf_BandSource* bsptr
        bsptr = <image_source.nitf_BandSource*>bandsource._c_source
        if not image_source.nitf_ImageSource_addBand(self._c_source, <image_source.nitf_BandSource*>bsptr, &error):
            raise NitfError(error)
        # Need to keep the bandsource object so that it isn't garbage collected if the caller doesn't keep their own copy
        self._band_sources.append(bandsource)


cdef class DataSource:
    cdef image_source.nitf_DataSource* _c_source
    def __cinit__(self):
        self._c_source = NULL


cdef class BandSource(DataSource):
    pass


cdef class MemorySource(BandSource):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, np.ndarray data not None, size=None, start=0, nbpp=0, pixskip=0):
        cdef nitf_Error error
        if size is None:
            size = data.data.nbytes
        self._c_source = <image_source.nitf_DataSource*>image_source.nitf_MemorySource_construct(<void*>&data.data[0], size, start, nbpp, pixskip, &error)
        if self._c_source is NULL:
            raise NitfError(error)


cdef class MemoryBandSource(MemorySource):
    pass


cdef class SegmentSource(DataSource):
    pass


cdef class SegmentMemorySource(SegmentSource):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, np.ndarray data not None, size=None, start=0, byte_skip=0):
        cdef nitf_Error error
        if size is None:
            size = data.nbytes
        self._c_source = <image_source.nitf_DataSource*>segment_source.nitf_SegmentMemorySource_construct(
                <const char*>&data.data[0], size, start, byte_skip, False, &error)
        if self._c_source is NULL:
            raise NitfError(error)


cdef class IOHandle:
    cdef io.nitf_IOHandle _c_io

    # deprecated, for compatibility
    ACCESS_READONLY=AccessFlags.NITF_ACCESS_READONLY
    ACCESS_WRITEONLY=AccessFlags.NITF_ACCESS_WRITEONLY
    ACCESS_READWRITE=AccessFlags.NITF_ACCESS_READWRITE
    CREATE=CreationFlags.NITF_CREATE
    OPEN_EXISTING=CreationFlags.NITF_OPEN_EXISTING
    TRUNCATE=CreationFlags.NITF_TRUNCATE

    def __cinit__(self, str fname, access=AccessFlags.NITF_ACCESS_READWRITE, creation=CreationFlags.NITF_CREATE):
        cdef nitf_Error error
        self._c_io = io.nitf_IOHandle_create(fname, access, creation, &error)
        if not io.is_iohandle_valid(self._c_io):
            raise NitfError(error)

    def close(self):
        io.nitf_IOHandle_close(self._c_io)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


cdef class SegmentFileSource(SegmentSource):
    def __cinit__(self, IOHandle handle, start=0, byte_skip=0):
        cdef nitf_Error error
        self._c_source = <image_source.nitf_DataSource*>segment_source.nitf_SegmentFileSource_construct(
            handle._c_io, start, byte_skip, &error)
        if self._c_source is NULL:
            raise NitfError(error)


cdef class Writer:
    cdef io.nitf_Writer* _c_writer
    cpdef Record _rcrd
    cpdef IOHandle _iohandle

    def __cinit__(self, Record rcrd=None, IOHandle iohandle=None):
        cdef nitf_Error error

        self._c_writer = io.nitf_Writer_construct(&error)
        if self._c_writer is NULL:
            raise NitfError(error)

        self._rcrd = rcrd
        self._iohandle = iohandle

    def __enter__(self):
        if self._rcrd is None or self._iohandle is None:
            raise ValueError("Must construct with a record and iohandle to use Writer as a context manager")
        self.prepare(self._rcrd, self._iohandle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is None and self._rcrd is not None and self._iohandle is not None:
            # automatically write() when used as a context manager
            self.write()

    def __dealloc__(self):
        if self._c_writer is not NULL:
            io.nitf_Writer_destruct(&self._c_writer)

    def prepare(self, Record recrd, IOHandle iohandle):
        cdef nitf_Error error
        cdef record.nitf_Record* rec=recrd._c_record
        cdef io.nitf_IOHandle hndl=iohandle._c_io

        if not io.nitf_Writer_prepare(self._c_writer, rec, hndl, &error):
            raise NitfError(error)
        return True

    def write(self):
        cdef nitf_Error error

        if not io.nitf_Writer_write(self._c_writer, &error):
            raise NitfError(error)

    def new_image_writer(self, index):
        cdef nitf_Error error
        cdef io.nitf_ImageWriter* iw

        iw = io.nitf_Writer_newImageWriter(self._c_writer, index, NULL, &error)
        if iw is NULL:
            raise NitfError(error)
        return ImageWriter.from_ptr(iw)

    def new_data_extension_writer(self, index):
        cdef nitf_Error error
        cdef io.nitf_SegmentWriter* sw

        sw = io.nitf_Writer_newDEWriter(self._c_writer, index, &error)
        if sw is NULL:
            raise NitfError(error)
        return SegmentWriter.from_ptr(sw)

    @deprecated("Old SWIG API")
    def newImageWriter(self, index):
        return self.new_image_writer(index)

    @deprecated("Old SWIG API")
    def newDEWriter(self, index):
        return self.new_data_extension_writer(index)


cdef class Reader:
    cdef io.nitf_Reader* _c_reader
    cpdef Record _record
    cpdef IOHandle _iohandle

    def __cinit__(self, iohandle=None):
        cdef nitf_Error error
        self._c_reader = io.nitf_Reader_construct(&error)
        if self._c_reader is NULL:
            raise NitfError(error)
        self._iohandle = iohandle

    def __dealloc__(self):
        if self._c_reader is not NULL:
            io.nitf_Reader_destruct(&self._c_reader)

    def __enter__(self):
        if self._iohandle is None:
            raise ValueError("Must construct with an iohandle to use Reader as a context manager")
        record = self.read(self._iohandle)
        return self, record

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def read(self, IOHandle iohandle):
        cdef nitf_Error error
        cdef io.nitf_IOHandle hndl = iohandle._c_io
        cdef record.nitf_Record* rec

        rec = io.nitf_Reader_read(self._c_reader, hndl, &error)
        if rec is NULL:
            raise NitfError(error)
        self._c_reader.ownInput = 0  # TODO: removing this causes a double free()..make sure this is correct behavior
        self._record = Record.from_ptr(rec)
        return self._record

    @deprecated("Old SWIG API")
    def newImageReader(self, num, options=None):
        return self.new_image_reader(num, options)

    @deprecated("Old SWIG API")
    def newDEReader(self, num, options=None):
        return self.new_data_extension_reader(num, options)

    def new_image_reader(self, num, options=None):
        cdef nitf_Error error
        cdef io.nitf_ImageReader* reader
        if options is not None:
            raise NotImplementedError("Options are not currently implemented")
        nbpp = int(self._record.images[num].subheader.numBitsPerPixel)
        reader = io.nitf_Reader_newImageReader(self._c_reader, num, <io.nrt_HashTable*>NULL, &error)
        if reader is NULL:
            raise NitfError(error)
        return ImageReader.from_ptr(reader, nbpp)

    def new_data_extension_reader(self, num):
        cdef nitf_Error error
        cdef io.nitf_SegmentReader* reader
        reader = io.nitf_Reader_newDEReader(self._c_reader, num, &error)
        if reader is NULL:
            raise NitfError(error)
        return SegmentReader.from_ptr(reader)


cdef class ImageWriter:
    cdef io.nitf_ImageWriter* _c_writer

    def __cinit__(self):
        self._c_writer = NULL

    @staticmethod
    cdef from_ptr(io.nitf_ImageWriter* ptr):
        obj = ImageWriter()
        obj._c_writer = ptr
        return obj

    def attach_source(self, ImageSource imagesource):
        cdef nitf_Error error
        cdef image_source.nitf_ImageSource* src = imagesource._c_source

        if not io.nitf_ImageWriter_attachSource(self._c_writer, src, &error):
            raise NitfError(error)

    @deprecated("Old SWIG API")
    def attachSource(self, ImageSource imagesource):
        self.attach_source(imagesource)


cdef class ImageReader:
    cdef io.nitf_ImageReader* _c_reader
    cdef int _nbpp

    def __cinit__(self, int nbpp):
        self._c_reader = NULL
        self._nbpp = nbpp

    @staticmethod
    cdef from_ptr(io.nitf_ImageReader* ptr, int nbpp):
        obj = ImageReader(nbpp)
        obj._c_reader = ptr
        return obj

    cpdef read(self, SubWindow window, downsampler=None, dtype=None):
        cdef nitf_Error error
        cdef nitf_Uint8** buf
        cdef int padded

        if downsampler is not None:
            raise NotImplementedError("Downsampling now implemented")

        rowSkip, colSkip = 1, 1  # TODO: get skip factor from downsampler
        # subimageSize = int((window.numRows // rowSkip)\
        #              * (window.numCols // colSkip)\
        #              * io.nitf_ImageIO_pixelSize(self._c_reader.imageDeblocker))
        subimageSize = (window.numRows // rowSkip) * (window.numCols // colSkip)
        if dtype is None:
            if self._nbpp == 8:
                dtype = np.uint8
            elif self._nbpp == 16:
                dtype = np.uint16
            elif self._nbpp == 32:
                dtype = np.uint32
            elif self._nbpp == 64:
                dtype = np.uint64
        if dtype is None:
            raise ValueError("Unable to determine dtype from nbpp, specify a dtype")
        result = np.ndarray((window.numBands, subimageSize), dtype=dtype, order='C')

        try:
            buf = <nitf_Uint8**>PyMem_Malloc(window.numBands * sizeof(nitf_Uint8*))
            if buf is NULL:
                raise MemoryError("Unable to allocate buffer for image read")
            buf[0] = <nitf_Uint8*>np.PyArray_DATA(result)

            for i in range(1, window.numBands):
                buf[i] = &(buf[0][sizeof(nitf_Uint8) * subimageSize * i])

            if not io.nitf_ImageReader_read(self._c_reader, window._c_window, buf, &padded, &error):
                raise NitfError(error)
        finally:
            PyMem_Free(buf)

        return result


cdef class SubWindow:
    cdef io.nitf_SubWindow* _c_window

    def __cinit__(self, image_subheader=None):
        cdef nitf_Error error
        self._c_window = io.nitf_SubWindow_construct(&error)
        if self._c_window is NULL:
            raise NitfError(error)
        self._c_window.startRow = 0
        self._c_window.startCol = 0
        self._c_window.bandList = <nitf_Uint32*>NULL

        if image_subheader:
            self._c_window.numRows = int(image_subheader['numRows'])
            self._c_window.numCols = int(image_subheader['numCols'])
            bands = int(image_subheader['numImageBands']) + \
                int(image_subheader['numMultispectralImageBands'])
            self._c_window.numBands = len(bands)
            self._c_window.bandList = <nitf_Uint32*>PyMem_Malloc(self._c_window.numBands)
            for i in range(bands):
                self._c_window.bandList[i] = <int>i
        else:
            self._c_window.numRows = 0
            self._c_window.numCols = 0

    def __dealloc__(self):
        if self._c_window != NULL:
            if self._c_window.bandList != NULL:
                PyMem_Free(self._c_window.bandList)
                io.nitf_SubWindow_destruct(&self._c_window)

    @property
    def startRow(self):
        return self._c_window.startRow

    @startRow.setter
    def startRow(self, nitf_Uint32 value):
        self._c_window.startRow = value

    @property
    def startCol(self):
        return self._c_window.startCol

    @startCol.setter
    def startCol(self, nitf_Uint32 value):
        self._c_window.startCol = value

    @property
    def numRows(self):
        return self._c_window.numRows

    @numRows.setter
    def numRows(self, nitf_Uint32 value):
        self._c_window.numRows = value

    @property
    def numCols(self):
        return self._c_window.numCols

    @numCols.setter
    def numCols(self, nitf_Uint32 value):
        self._c_window.numCols = value

    @property
    def numBands(self):
        return self._c_window.numBands

    @numBands.setter
    def numBands(self, nitf_Uint32 value):
        cdef nitf_Uint32* tmp
        if value == 0:
            self._c_window.numBands = value
            PyMem_Free(self._c_window.bandList)
            self._c_window.bandList = NULL
        # realloc works like malloc if the bandList is NULL
        tmp = <nitf_Uint32*>PyMem_Realloc(self._c_window.bandList, value)
        if tmp is NULL:
            raise MemoryError("Unable to reallocate bandList memory")
        self._c_window.bandList = tmp
        self._c_window.numBands = value

    @property
    def bandList(self):
        tmp = []
        for i in range(self._c_window.numBands):
            tmp.append(self._c_window.bandList[i])
        return tmp

    @bandList.setter
    def bandList(self, list value):
        if len(value) != self._c_window.numBands:
            self.numBands = len(value)
        for i, v in enumerate(value):
            self._c_window.bandList[i] = v

    def __str__(self):
        return f'{self.startRow}, {self.startCol}, {self.numRows}, {self.numCols}, {len(self.bandList)}'


cdef class SegmentReader:
    cdef io.nitf_SegmentReader* _c_reader

    def __cinit__(self):
        self._c_reader = NULL

    @staticmethod
    cdef from_ptr(io.nitf_SegmentReader* ptr):
        obj = SegmentReader()
        obj._c_reader = ptr
        return obj

    def read(self, np.npy_intp count=0):
        cdef nitf_Error error

        if count == 0:
            count = len(self)
        buf = np.PyArray_SimpleNew(1, &count, np.NPY_BYTE)
        if not io.nitf_SegmentReader_read(self._c_reader, np.PyArray_DATA(buf), count, &error):
            raise NitfError(error)
        return buf

    def __len__(self):
        cdef nitf_Error error
        l = io.nitf_SegmentReader_getSize(self._c_reader, &error)
        if l < 0:
            raise NitfError(error)
        return l


cdef class SegmentWriter:
    cdef io.nitf_SegmentWriter* _c_writer

    def __cinit__(self):
        self._c_writer = NULL

    @staticmethod
    cdef from_ptr(io.nitf_SegmentWriter* ptr):
        obj = SegmentWriter()
        obj._c_writer = ptr
        return obj

    def attach_source(self, SegmentSource segmentsource):
        cdef nitf_Error error
        cdef segment_source.nitf_SegmentSource* src = <segment_source.nitf_SegmentSource*>segmentsource._c_source

        if not io.nitf_SegmentWriter_attachSource(self._c_writer, src, &error):
            raise NitfError(error)

    @deprecated("Old SWIG API")
    def attachSource(self, SegmentSource segmentsource):
        self.attach_source(segmentsource)


cpdef enum Version:
    NITF_VER_20=100
    NITF_VER_21
    NITF_VER_UNKNOWN

