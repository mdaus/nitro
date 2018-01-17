#cython: language_level=3, c_string_type=unicode, c_string_encoding=ascii

cimport cython
cimport dataextension_segment
cimport extensions
cimport field
cimport header
cimport image_segment
cimport image_source
cimport io
cimport record
cimport tre
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc
from error cimport nitf_Error
from types cimport *

import contextlib
import numpy as np
import os
from deprecated import deprecated


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    """Useful context manager for temporarily changing numpy print options"""
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class NitfError(BaseException):
    def __init__(self, nitf_error):
        if type(nitf_error['message']) in [bytes, bytearray]:
            super().__init__(nitf_error['message'].decode(errors='replace'))
        else:
            super().__init__(nitf_error['message'])
        self.nitf_error = nitf_error

    def __str__(self):
        return "NitfError at {file}:{line} :: {func}: {message}".format(**self.nitf_error)

cdef class AccessFlags:
    NITF_ACCESS_READONLY=os.O_RDONLY
    NITF_ACCESS_WRITEONLY=os.O_WRONLY
    NITF_ACCESS_READWRITE=os.O_RDWR

cdef class CreationFlags:
    NITF_CREATE=os.O_CREAT
    NITF_OPEN_EXISTING=0x000
    NITF_TRUNCATE=os.O_TRUNC

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
        elif container_type is Field:
            rval = Field.from_ptr(<field.nitf_Field*>self._c_data)
        else:
            rval = <unsigned long long>self._c_data
        return rval

    def __str__(self):
        return self.convert(str)


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
        h = FileHeader.from_ptr(hdr)
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
        h = ImageSubheader.from_ptr(hdr)
        return h

    @deprecated("Old SWIG API")
    def addBands(self, num=1):
        if num < 0:
            num = 0
        self.create_bands(num)

    def create_bands(self, num):
        self.subheader.create_bands(num)


cdef class DESegment:
    cdef dataextension_segment.nitf_DESegment* _c_des

    @staticmethod
    cdef from_record(record.nitf_Record* c_rec):
        cdef nitf_Error error
        obj = DESegment()
        obj._c_des = record.nitf_Record_newDataExtensionSegment(c_rec, &error)
        if obj._c_des is NULL:
            raise NitfError(error)
        return obj  # allow chaining

    @staticmethod
    cdef from_ptr(dataextension_segment.nitf_DESegment* ptr):
        obj = DESegment()
        obj._c_des = ptr
        return obj

    @property
    def subheader(self):
        cdef header.nitf_DESubheader* hdr
        hdr = self._c_des.subheader
        h = DESubheader.from_ptr(hdr)
        return h


cdef class BaseFieldHeader:
    deprecated_items = {}
    equiv_items = {}

    def get_items(self):
        return {}

    def __contains__(self, item):
        return item in self.deprecated_items or item in self.equiv_items or item in self.get_items()

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value

    def __getitem__(self, item):
        item = self.deprecated_items.get(item, item)
        item = self.equiv_items.get(item, item)
        tmp = self.get_items()
        try:
            return tmp[item]
        except KeyError:
            raise AttributeError(item)

    def __setitem__(self, key, value):
        """Attempt to deduce the type. Good chance this might be wrong so be explicit if you need to."""
        key = self.deprecated_items.get(key, key)
        key = self.equiv_items.get(key, key)
        try:
            tmp = self.get_items()[key]
            if type(value) is int:
                if value < 0:
                    if -value >= (2 ** 32) / 2:
                        tmp.set_int64(value)
                    else:
                        tmp.set_int32(value)
                else:
                    if value >= (2 ** 32):
                        tmp.set_uint64(value)
                    else:
                        tmp.set_uint32(value)
            elif type(value) is float:
                tmp.set_real(value)
            else:
                tmp.set_string(str(value))
        except KeyError:
            raise AttributeError(key)

    def __str__(self):
        rval = ''
        for key, val in self.get_items().items():
            if isinstance(val, Field):
                rval += f"{key}({len(val)}) = '{val}'\n"
            else:
                rval += f"{key} = '{val}'\n"
        return rval

    def __repr__(self):
        return f"<{self.__class__.__name__} {<unsigned long long>self._c_header:#0x}>"


cdef class FileHeader(BaseFieldHeader):
    cdef header.nitf_FileHeader* _c_header

    @staticmethod
    cdef from_ptr(header.nitf_FileHeader* ptr):
        obj = FileHeader()
        obj._c_header = ptr
        return obj

    def get_items(self):
        tmp = {
            'fileHeader':Field.from_ptr(self._c_header.fileHeader),
            'fileVersion':Field.from_ptr(self._c_header.fileVersion),
            'complianceLevel':Field.from_ptr(self._c_header.complianceLevel),
            'systemType':Field.from_ptr(self._c_header.systemType),
            'originStationID':Field.from_ptr(self._c_header.originStationID),
            'fileDateTime':Field.from_ptr(self._c_header.fileDateTime),
            'fileTitle':Field.from_ptr(self._c_header.fileTitle),
            'classification':Field.from_ptr(self._c_header.classification),
            'messageCopyNum':Field.from_ptr(self._c_header.messageCopyNum),
            'messageNumCopies':Field.from_ptr(self._c_header.messageNumCopies),
            'encrypted':Field.from_ptr(self._c_header.encrypted),
            'backgroundColor':Field.from_ptr(self._c_header.backgroundColor),
            'originatorName':Field.from_ptr(self._c_header.originatorName),
            'originatorPhone':Field.from_ptr(self._c_header.originatorPhone),
            'fileLength':Field.from_ptr(self._c_header.fileLength),
            'headerLength':Field.from_ptr(self._c_header.headerLength),
            }
        return tmp


cdef class BandInfo:
    cdef header.nitf_BandInfo* _c_binfo

    @staticmethod
    cdef from_ptr(header.nitf_BandInfo* ptr):
        obj = BandInfo()
        obj._c_binfo = ptr
        return obj

    deprecated_items = {}
    equiv_items = {'IREPBAND': 'representation',
                   'ISUBCAT': 'subcategory',
                   'IFC': 'imageFilterCondition',
                   'IMFLT': 'imageFilterCode',
                   'NLUTS': 'numLUTs'}

    def get_items(self):
        tmp = {
            'representation':Field.from_ptr(self._c_binfo.representation),
            'subcategory':Field.from_ptr(self._c_binfo.subcategory),
            'imageFilterCondition':Field.from_ptr(self._c_binfo.imageFilterCondition),
            'imageFilterCode':Field.from_ptr(self._c_binfo.imageFilterCode),
            'numLUTs':Field.from_ptr(self._c_binfo.numLUTs),
            }
        return tmp

    def __contains__(self, item):
        return item in self.deprecated_items or item in self.equiv_items or item in self.get_items()

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value

    def __getitem__(self, item):
        item = self.deprecated_items.get(item, item)
        item = self.equiv_items.get(item, item)
        tmp = self.get_items()
        try:
            return tmp[item]
        except KeyError:
            raise AttributeError(item)

    def __setitem__(self, key, value):
        """Attempt to deduce the type. Good chance this might be wrong so be explicit if you need to."""
        key = self.deprecated_items.get(key, key)
        key = self.equiv_items.get(key, key)
        try:
            tmp = self.get_items()[key]
            if type(value) is int:
                if value < 0:
                    if -value >= (2 ** 32) / 2:
                        tmp.set_int64(value)
                    else:
                        tmp.set_int32(value)
                else:
                    if value >= (2 ** 32):
                        tmp.set_uint64(value)
                    else:
                        tmp.set_uint32(value)
            elif type(value) is float:
                tmp.set_real(value)
            else:
                tmp.set_string(str(value))
        except KeyError:
            raise AttributeError(key)


cdef class ImageSubheader(BaseFieldHeader):
    cdef header.nitf_ImageSubheader* _c_header

    @staticmethod
    cdef from_ptr(header.nitf_ImageSubheader* ptr):
        obj = ImageSubheader()
        obj._c_header = ptr
        return obj

    def insert_image_comment(self, comment, position=-1):
        cdef int rval
        cdef nitf_Error error
        cdef char* tmp

        tmp = comment

        rval = header.nitf_ImageSubheader_insertImageComment(self._c_header, str(comment), position, &error)
        if rval < 0:
            raise NitfError(error)
        return rval

    def create_bands(self, num_bands):
        cdef nitf_Error error
        if not header.nitf_ImageSubheader_createBands(self._c_header, num_bands, &error):
            raise NitfError(error)

    @property
    def band_count(self):
        cdef nitf_Error error
        cdef nitf_Uint32 cnt

        cnt = header.nitf_ImageSubheader_getBandCount(self._c_header, &error)
        if cnt == <nitf_Uint32>-1:
            raise NitfError(error)
        return cnt

    @deprecated("Old SWIG API")
    def getBandCount(self):
        return self.band_count

    @property
    def band_info(self):
        cnt = self.band_count
        for i in range(cnt):
            yield BandInfo.from_ptr(self._c_header.bandInfo[i])

    def get_band_info(self, unsigned int idx):
        if idx >= self.band_count:
            raise IndexError("Invalid band number")
        return BandInfo.from_ptr(self._c_header.bandInfo[idx])

    @deprecated("Old SWIG API")
    def getBandInfo(self, idx):
        return self.get_band_info(idx)

    @deprecated("Old SWIG API")
    def getXHD(self):
        return self.extendedSection

    @deprecated("Old SWIG API")
    def getUDHD(self):
        return self.userDefinedSection

    deprecated_items = {'numrows': 'numRows', 'numcols': 'numCols'}

    def get_items(self):
        tmp = {
            'filePartType':Field.from_ptr(self._c_header.filePartType),
            'imageId':Field.from_ptr(self._c_header.imageId),
            'imageDateAndTime':Field.from_ptr(self._c_header.imageDateAndTime),
            'targetId':Field.from_ptr(self._c_header.targetId),
            'imageTitle':Field.from_ptr(self._c_header.imageTitle),
            'imageSecurityClass':Field.from_ptr(self._c_header.imageSecurityClass),
            'encrypted':Field.from_ptr(self._c_header.encrypted),
            'imageSource':Field.from_ptr(self._c_header.imageSource),
            'numRows':Field.from_ptr(self._c_header.numRows),
            'numCols':Field.from_ptr(self._c_header.numCols),
            'pixelValueType':Field.from_ptr(self._c_header.pixelValueType),
            'imageRepresentation':Field.from_ptr(self._c_header.imageRepresentation),
            'imageCategory':Field.from_ptr(self._c_header.imageCategory),
            'actualBitsPerPixel':Field.from_ptr(self._c_header.actualBitsPerPixel),
            'pixelJustification':Field.from_ptr(self._c_header.pixelJustification),
            'imageCoordinateSystem':Field.from_ptr(self._c_header.imageCoordinateSystem),
            'cornerCoordinates':Field.from_ptr(self._c_header.cornerCoordinates),
            'numImageComments':Field.from_ptr(self._c_header.numImageComments),
            'imageComments':List(Field).from_ptr(self._c_header.imageComments),
            'imageCompression':Field.from_ptr(self._c_header.imageCompression),
            'compressionRate':Field.from_ptr(self._c_header.compressionRate),
            'numImageBands':Field.from_ptr(self._c_header.numImageBands),
            'numMultispectralImageBands':Field.from_ptr(self._c_header.numMultispectralImageBands),
            'imageSyncCode':Field.from_ptr(self._c_header.imageSyncCode),
            'imageMode':Field.from_ptr(self._c_header.imageMode),
            'numBlocksPerRow':Field.from_ptr(self._c_header.numBlocksPerRow),
            'numBlocksPerCol':Field.from_ptr(self._c_header.numBlocksPerCol),
            'numPixelsPerHorizBlock':Field.from_ptr(self._c_header.numPixelsPerHorizBlock),
            'numPixelsPerVertBlock':Field.from_ptr(self._c_header.numPixelsPerVertBlock),
            'numBitsPerPixel':Field.from_ptr(self._c_header.numBitsPerPixel),
            'imageDisplayLevel':Field.from_ptr(self._c_header.imageDisplayLevel),
            'imageAttachmentLevel':Field.from_ptr(self._c_header.imageAttachmentLevel),
            'imageLocation':Field.from_ptr(self._c_header.imageLocation),
            'imageMagnification':Field.from_ptr(self._c_header.imageMagnification),
            'userDefinedImageDataLength':Field.from_ptr(self._c_header.userDefinedImageDataLength),
            'userDefinedOverflow':Field.from_ptr(self._c_header.userDefinedOverflow),
            'extendedHeaderLength':Field.from_ptr(self._c_header.extendedHeaderLength),
            'extendedHeaderOverflow':Field.from_ptr(self._c_header.extendedHeaderOverflow),
            'userDefinedSection':Extensions.from_ptr(self._c_header.userDefinedSection),
            'extendedSection':Extensions.from_ptr(self._c_header.extendedSection),
            }
        return tmp


cdef class DESubheader(BaseFieldHeader):
    cdef header.nitf_DESubheader* _c_header

    @staticmethod
    cdef from_ptr(header.nitf_DESubheader* ptr):
        obj = DESubheader()
        obj._c_header = ptr
        return obj

    @deprecated("Old SWIG API")
    def getUDHD(self):
        return self.userDefinedSection

    deprecated_items = {'subheaderFields': 'subheader_fields'}

    def get_items(self):
        tmp = {
            'filePartType':Field.from_ptr(self._c_header.filePartType),
            'typeId':Field.from_ptr(self._c_header.imageId),
            'version':Field.from_ptr(self._c_header.version),
            'overflowedHeaderType':Field.from_ptr(self._c_header.overflowedHeaderType),
            'dataItemOverflowed':Field.from_ptr(self._c_header.dataItemOverflowed),
            'subheaderFieldsLength':Field.from_ptr(self._c_header.subheaderFieldsLength),
            'subheaderFields':List(TRE).from_ptr(self._c_headersubheaderFields),
            'userDefinedSection':Extensions.from_ptr(self._c_header.userDefinedSection),
        }
        return tmp

    @property
    def subheader_fields(self):
        rval = []
        for i in range(self._c_header.subheaderFieldsLength):
            rval.append(TRE.from_ptr(self._c_header.subheaderFields[i]))
        return rval


cdef class Extensions:
    cdef header.nitf_Extensions* _c_extensions

    def __cinit__(self):
        self._c_extensions = NULL

    @staticmethod
    cdef from_ptr(header.nitf_Extensions* ptr):
        obj = Extensions()
        obj._c_extensions = ptr
        return obj

    def append(self, TRE tre not None):
        cdef nitf_Error error
        if not extensions.nitf_Extensions_appendTRE(self._c_extensions, tre._c_tre, &error):
            raise NitfError(error)

    def extend(self, it):
        for tre in it:
            self.append(tre)

    def __contains__(self, str item):
        return <bint>extensions.nitf_Extensions_exists(self._c_extensions, item)

    def __getitem__(self, str item not None):
        cdef nitf_List* tres
        tres = extensions.nitf_Extensions_getTREsByName(self._c_extensions, item)
        if tres is NULL:
            return []
        return List(TRE).from_ptr(tres)

    def __delitem__(self, str key not None):
        extensions.nitf_Extensions_removeTREsByName(self._c_extensions, key)

    def __getattr__(self, str item not None):
        return self[item]

    def __delattr__(self, str item not None):
        del self[item]

    def __iter__(self):
        cdef nitf_Error error
        cdef extensions.nitf_ExtensionsIterator it
        cdef extensions.nitf_ExtensionsIterator end
        cdef tre.nitf_TRE* tre_ptr

        it = extensions.nitf_Extensions_begin(self._c_extensions)
        end = extensions.nitf_Extensions_end(self._c_extensions)

        while extensions.nitf_ExtensionsIterator_notEqualTo(&it, &end):
            tre_ptr = extensions.nitf_ExtensionsIterator_get(&it)
            if tre_ptr is not NULL:
                yield TRE.from_ptr(tre_ptr)
            extensions.nitf_ExtensionsIterator_increment(&it)

    @deprecated("Old SWIG API")
    def removeTREsByName(self, tag):
        del self[tag]


cdef class Field:
    cpdef field.nitf_Field* _c_field
    cdef Py_ssize_t _shape[1]
    cdef Py_ssize_t _strides[1]
    cdef void* _buf_value
    cdef int _ref_count

    def __cinit__(self):
        self._c_field = NULL
        self._buf_value = NULL
        self._strides[0] = 1
        self._shape[0] = 0
        self._ref_count = 0

    @staticmethod
    cdef from_ptr(field.nitf_Field* ptr):
        obj = Field()
        obj._c_field = ptr
        return obj

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
        if not field.nitf_Field_get(self._c_field, <NITF_DATA*>&val, NITF_CONV_UINT, 8, &error):
            raise NitfError(error)
        return val

    def get_int(self):
        cdef nitf_Error error
        cdef int64_t val
        if not field.nitf_Field_get(self._c_field, <NITF_DATA*>&val, NITF_CONV_INT, 8, &error):
            raise NitfError(error)
        return val

    def get_real(self):
        cdef nitf_Error error
        cdef double val
        if not field.nitf_Field_get(self._c_field, <NITF_DATA*>&val, NITF_CONV_REAL, 8, &error):
            raise NitfError(error)
        return val

    def get_string(self):
        cdef nitf_Error error
        cdef Py_ssize_t val_length = self._c_field.length+1
        cdef char* val = <char*>PyMem_Malloc(val_length)

        if self._c_field.length <= 0:
            return ""
        try:
            if not field.nitf_Field_get(self._c_field, <NITF_DATA*>val, NITF_CONV_STRING, val_length, &error):
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

        if not field.nitf_Field_get(self._c_field, self._buf_value, NITF_CONV_RAW, buffer.len, &error):
            PyMem_Free(self._buf_value)
            self._buf_value = NULL
            raise NitfError(error)

    def __releasebuffer__(self, Py_buffer* buffer):
        assert(buffer.buf == self._buf_value)
        self._ref_count -= 1
        if self._ref_count == 0:
            PyMem_Free(self._buf_value)
            self._buf_value = NULL


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

    def __dealloc__(self):
        # TODO: enabling this is causing a crash do to double free()..make sure this is still getting completely deallocated
        # if self._c_source is not NULL:
        #     image_source.nitf_ImageSource_destruct(&self._c_source)
        # self._band_sources = []
        pass

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

    def __dealloc__(self):
        if self._c_source is not NULL:
            import traceback
            # traceback.print_stack(limit=5)
            # image_source.nitf_DataSource_destruct(&self._c_source)


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


cdef class Writer:
    cdef io.nitf_Writer* _c_writer
    def __cinit__(self):
        cdef nitf_Error error
        self._c_writer = io.nitf_Writer_construct(&error)
        if self._c_writer is NULL:
            raise NitfError(error)

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

    @deprecated("Old SWIG API")
    def newImageWriter(self, index):
        return self.new_image_writer(index)


cdef class Reader:
    cdef io.nitf_Reader* _c_reader
    cpdef Record _record

    def __cinit__(self):
        cdef nitf_Error error
        self._c_reader = io.nitf_Reader_construct(&error)
        if self._c_reader is NULL:
            raise NitfError(error)

    def __dealloc__(self):
        if self._c_reader is not NULL:
            io.nitf_Reader_destruct(&self._c_reader)

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


cdef class TRE:
    cdef tre.nitf_TRE* _c_tre

    @cython.nonecheck(False)
    def __cinit__(self, str tag=None, str id=None):
        cdef nitf_Error error

        self._c_tre = NULL
        if tag is not None:
            if id is None:
                self._c_tre = tre.nitf_TRE_construct(tag, <char*>NULL, &error)
            else:
                self._c_tre = tre.nitf_TRE_construct(tag, id, &error)
            if self._c_tre is NULL:
                raise NitfError(error)

    @staticmethod
    cdef from_ptr(tre.nitf_TRE* ptr):
        obj = TRE()
        obj._c_tre = ptr
        return obj

    @property
    def tag(self):
        return <str>self._c_tre.tag

    def __len__(self):
        cdef nitf_Error error

        sz = tre.nitf_TRE_getCurrentSize(self._c_tre, &error)
        if sz < 0:
            raise NitfError(error)
        return sz

    def __str__(self):
        rval = f'-----\n{self.tag}({len(self)})\n\t'
        rval += '\n\t'.join([f'{k} = {v}' for k,v in self])
        rval += '\n'
        return rval

    def __contains__(self, str item):
        return <bint>tre.nitf_TRE_exists(self._c_tre, item)

    def __getitem__(self, str item not None):
        tre.nitf_TRE_getField(self._c_tre, item)

    def __getattr__(self, item):
        return self[item]

    def __setitem__(self, str key not None, value):
        cdef nitf_Error error
        cdef char* tmp

        tmp = <char*>value
        if not tre.nitf_TRE_setField(self._c_tre, key, <NITF_DATA*>tmp, len(value), &error):
            raise NitfError(error)

    def __setattr__(self, key, value):
        self[key] = value

    def __iter__(self):
        cdef nitf_Error error
        cdef tre.nitf_TREEnumerator* it
        cdef nitf_Pair* pair
        cdef field.nitf_Field* fld

        it = tre.nitf_TRE_begin(self._c_tre, &error)
        if it is NULL:
            raise NitfError(error)
        while it.hasNext(&it):
            pair = it.next(it, &error)
            if pair is NULL:
                raise NitfError(error)
            fld = <field.nitf_Field*>pair.data
            if fld is not NULL:
                yield pair.key, Field.from_ptr(fld)

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


cdef class ListIter:
    cdef nitf_ListIterator _c_iter
    cdef nitf_List* _c_list
    cdef object _list

    def __cinit__(self):
        self._list = None
        self._c_list = NULL

    @staticmethod
    cdef from_ptr(lst, nitf_List* c_list):
        obj = ListIter()
        obj._list = lst
        obj._c_list = c_list
        obj._c_iter = nitf_List_begin(obj._c_list)
        return obj

    def __next__(self):
        cdef nitf_ListIterator end
        cdef NITF_DATA* data
        end = nitf_List_end(self._c_list)
        if nitf_ListIterator_equals(&self._c_iter, &end):
            raise StopIteration()
        data = nitf_ListIterator_get(&self._c_iter)
        nitf_ListIterator_increment(&self._c_iter)
        return NitfData.from_ptr(data).convert(self._list.container_type)

cdef class List:
    cdef nitf_List* _c_list
    cpdef type _container_type

    @property
    def container_type(self):
        return self._container_type

    def __cinit__(self, container_type):
        self._c_list = NULL
        self._container_type = container_type

    cdef from_ptr(self, nitf_List* ptr):
        self._c_list = ptr
        return self

    def __len__(self):
        return nitf_List_size(self._c_list)

    def __getitem__(self, item):
        cdef nitf_Error error
        cdef NITF_DATA* rval
        if item > (len(self)-1) or item < 0:
            raise IndexError()
        rval = nitf_List_get(self._c_list, <int>item, &error)
        if rval is NULL:
            raise NitfError(error)
        return NitfData.from_ptr(rval).convert(self._container_type)

    def __iter__(self):
        return ListIter.from_ptr(self, self._c_list)


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


cpdef enum Version:
    NITF_VER_20=100
    NITF_VER_21
    NITF_VER_UNKNOWN

cpdef enum FieldType:
    NITF_BCS_A
    NITF_BCS_N
    NITF_BINARY

cpdef enum nitf_ConvType:
    NITF_CONV_UINT
    NITF_CONV_INT
    NITF_CONV_REAL
    NITF_CONV_STRING
    NITF_CONV_RAW

cpdef enum ErrorCode:
    NITF_NO_ERR=0
    NITF_ERR_MEMORY
    NITF_ERR_OPENING_FILE
    NITF_ERR_READING_FROM_FILE
    NITF_ERR_SEEKING_IN_FILE
    NITF_ERR_WRITING_TO_FILE
    NITF_ERR_STAT_FILE
    NITF_ERR_LOADING_DLL
    NITF_ERR_UNLOADING_DLL
    NITF_ERR_RETRIEVING_DLL_HOOK
    NITF_ERR_UNINITIALIZED_DLL_READ
    NITF_ERR_INVALID_PARAMETER
    NITF_ERR_INVALID_OBJECT
    NITF_ERR_INVALID_FILE
    NITF_ERR_COMPRESSION
    NITF_ERR_DECOMPRESSION
    NITF_ERR_PARSING_FILE
    NITF_ERR_INT_STACK_OVERFLOW
    NITF_ERR_UNK

