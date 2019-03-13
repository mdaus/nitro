#cython: language_level=3, c_string_type=unicode, c_string_encoding=ascii

cimport field
cimport header
cimport tre
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer
from error cimport nitf_Error
from types cimport *

from . import field
from . import tre
from . import types
from collections import Mapping
from deprecated import deprecated
from .error import NitfError


def __convert_item(val, encoding):
    if isinstance(val, types.List):
        val = list(val)
    if isinstance(val, field.Field):
        rval = val.get_pyvalue()
        if encoding is not None and isinstance(rval, bytes):
            rval = rval.decode(encoding)
    elif isinstance(val, list):
        rval = [__convert_item(litem, encoding) for litem in val]
    else:
        try:
            rval = val.todict(encoding)
        except AttributeError:
            rval = str(val)  # fall back to the printable string
    return rval


cdef class BandInfo:
    cdef header.nitf_BandInfo* _c_binfo

    def __cinit__(self, c):
        assert(PyCapsule_IsValid(c, "BandInfo"))
        self._c_binfo = <header.nitf_BandInfo*>PyCapsule_GetPointer(c, "BandInfo")

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
            'representation':field.Field(PyCapsule_New(self._c_binfo.representation, "Field", NULL)),
            'subcategory':field.Field(PyCapsule_New(self._c_binfo.subcategory, "Field", NULL)),
            'imageFilterCondition':field.Field(PyCapsule_New(self._c_binfo.imageFilterCondition, "Field", NULL)),
            'imageFilterCode':field.Field(PyCapsule_New(self._c_binfo.imageFilterCode, "Field", NULL)),
            'numLUTs':field.Field(PyCapsule_New(self._c_binfo.numLUTs, "Field", NULL)),
        }
        return tmp

    def todict(self, encoding=None):
        rval = {}
        for key, val in self.get_items().items():
            tmp = __convert_item(val, encoding)
            rval[key] = tmp
        return rval

    def __contains__(self, item):
        if item in super():
            return True
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
            if isinstance(val, field.Field):
                rval += f"{key}({len(val)}) = '{val}'\n"
            else:
                rval += f"{key} = '{val}'\n"
        return rval

    def todict(self, encoding=None):
        rval = {}
        for key, val in self.get_items().items():
            tmp = __convert_item(val, encoding)
            if not isinstance(tmp, Mapping) or len(tmp) > 0:
                rval[key] = tmp
        return rval

    def __repr__(self):
        return f"<{self.__class__.__name__} {self._capsule()}>"


cdef class FileSecurity(BaseFieldHeader):
    cdef header.nitf_FileSecurity* _c_security

    def __cinit__(self, c):
        assert(PyCapsule_IsValid(c, "FileSecurity"))
        self._c_security = <header.nitf_FileSecurity*>PyCapsule_GetPointer(c, "FileSecurity")

    def _capsule(self):
        return PyCapsule_New(self._c_security, "FileSecurity", NULL)

    @staticmethod
    cdef from_ptr(header.nitf_FileSecurity* ptr):
        obj = FileSecurity()
        obj._c_security = ptr
        return obj

    def get_items(self):
        tmp = {
            'classificationSystem': field.Field(PyCapsule_New(self._c_security.classificationSystem, "Field", NULL)),
            'codewords': field.Field(PyCapsule_New(self._c_security.codewords, "Field", NULL)),
            'controlAndHandling': field.Field(PyCapsule_New(self._c_security.controlAndHandling, "Field", NULL)),
            'releasingInstructions': field.Field(PyCapsule_New(self._c_security.releasingInstructions, "Field", NULL)),
            'declassificationType': field.Field(PyCapsule_New(self._c_security.declassificationType, "Field", NULL)),
            'declassificationDate': field.Field(PyCapsule_New(self._c_security.declassificationDate, "Field", NULL)),
            'declassificationExemption': field.Field(PyCapsule_New(self._c_security.declassificationExemption, "Field", NULL)),
            'downgrade': field.Field(PyCapsule_New(self._c_security.downgrade, "Field", NULL)),
            'downgradeDateTime': field.Field(PyCapsule_New(self._c_security.downgradeDateTime, "Field", NULL)),
            'classificationText': field.Field(PyCapsule_New(self._c_security.classificationText, "Field", NULL)),
            'classificationAuthorityType': field.Field(PyCapsule_New(self._c_security.classificationAuthorityType, "Field", NULL)),
            'classificationAuthority': field.Field(PyCapsule_New(self._c_security.classificationAuthority, "Field", NULL)),
            'classificationReason': field.Field(PyCapsule_New(self._c_security.classificationReason, "Field", NULL)),
            'securitySourceDate': field.Field(PyCapsule_New(self._c_security.securitySourceDate, "Field", NULL)),
            'securityControlNumber': field.Field(PyCapsule_New(self._c_security.securityControlNumber, "Field", NULL)),
            }
        return tmp


cdef class FileHeader(BaseFieldHeader):
    cdef header.nitf_FileHeader* _c_header

    def __cinit__(self, c):
        assert(PyCapsule_IsValid(c, "FileHeader"))
        self._c_header = <header.nitf_FileHeader*>PyCapsule_GetPointer(c, "FileHeader")

    @staticmethod
    cdef from_ptr(header.nitf_FileHeader* ptr):
        obj = FileHeader()
        obj._c_header = ptr
        return obj

    def _capsule(self):
        return PyCapsule_New(self._c_header, "FileHeader", NULL)

    def get_items(self):
        tmp = {
            'fileHeader':field.Field(PyCapsule_New(self._c_header.fileHeader, "Field", NULL)),
            'fileVersion':field.Field(PyCapsule_New(self._c_header.fileVersion, "Field", NULL)),
            'complianceLevel':field.Field(PyCapsule_New(self._c_header.complianceLevel, "Field", NULL)),
            'systemType':field.Field(PyCapsule_New(self._c_header.systemType, "Field", NULL)),
            'originStationID':field.Field(PyCapsule_New(self._c_header.originStationID, "Field", NULL)),
            'fileDateTime':field.Field(PyCapsule_New(self._c_header.fileDateTime, "Field", NULL)),
            'fileTitle':field.Field(PyCapsule_New(self._c_header.fileTitle, "Field", NULL)),
            'classification':field.Field(PyCapsule_New(self._c_header.classification, "Field", NULL)),
            'securityGroup':FileSecurity(PyCapsule_New(self._c_header.securityGroup, "FileSecurity", NULL)),
            'messageCopyNum':field.Field(PyCapsule_New(self._c_header.messageCopyNum, "Field", NULL)),
            'messageNumCopies':field.Field(PyCapsule_New(self._c_header.messageNumCopies, "Field", NULL)),
            'encrypted':field.Field(PyCapsule_New(self._c_header.encrypted, "Field", NULL)),
            'backgroundColor':field.Field(PyCapsule_New(self._c_header.backgroundColor, "Field", NULL)),
            'originatorName':field.Field(PyCapsule_New(self._c_header.originatorName, "Field", NULL)),
            'originatorPhone':field.Field(PyCapsule_New(self._c_header.originatorPhone, "Field", NULL)),
            'fileLength':field.Field(PyCapsule_New(self._c_header.fileLength, "Field", NULL)),
            'headerLength':field.Field(PyCapsule_New(self._c_header.headerLength, "Field", NULL)),
            'userDefinedSection':tre.Extensions(PyCapsule_New(self._c_header.userDefinedSection, "Extensions", NULL)),
            'extendedSection':tre.Extensions(PyCapsule_New(self._c_header.extendedSection, "Extensions", NULL)),
            }
        return tmp


cdef class ImageSubheader(BaseFieldHeader):
    cdef header.nitf_ImageSubheader* _c_header

    def __cinit__(self, c=None):
        if c is not None:
            assert(PyCapsule_IsValid(c, "ImageSubheader"))
            self._c_header = <header.nitf_ImageSubheader*>PyCapsule_GetPointer(c, "ImageSubheader")

    def _capsule(self):
        return PyCapsule_New(self._c_header, "ImageSubheader", NULL)

    @staticmethod
    cdef from_ptr(header.nitf_ImageSubheader* ptr):
        obj = ImageSubheader()
        obj._c_header = ptr
        return obj

    def __copy__(self):
        cdef nitf_ImageSubheader* pObj
        cdef nitf_Error error
        pObj = nitf_ImageSubheader_clone(self._c_header, &error)
        if pObj is NULL:
            raise NitfError(error)
        return ImageSubheader.from_ptr(pObj)

    def insert_image_comment(self, comment, position=-1):
        cdef int rval
        cdef nitf_Error error
        cdef char* tmp

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
            yield BandInfo(PyCapsule_New(self._c_header.bandInfo[i], "BandInfo", NULL))

    def get_band_info(self, unsigned int idx):
        if idx >= self.band_count:
            raise IndexError("Invalid band number")
        return BandInfo(PyCapsule_New(self._c_header.bandInfo[idx], "BandInfo", NULL))

    @deprecated("Old SWIG API")
    def getBandInfo(self, idx):
        return self.get_band_info(idx)

    @deprecated("Old SWIG API")
    def getXHD(self):
        return self.extendedSection

    @deprecated("Old SWIG API")
    def getUDHD(self):
        return self.userDefinedSection

    def todict(self, encoding=None):
        rval = super().todict(encoding)
        bi = []
        for b in self.band_info:
            bi.append(b.todict(encoding))
        rval["bandInfo"] = bi
        return rval

    deprecated_items = {'numrows': 'numRows', 'numcols': 'numCols'}

    equiv_items = {'classification': 'imageSecurityClass'}  # so there's an equiv name to the file header version

    def get_items(self):
        tmp = {
            'filePartType':field.Field(PyCapsule_New(self._c_header.filePartType, "Field", NULL)),
            'imageId':field.Field(PyCapsule_New(self._c_header.imageId, "Field", NULL)),
            'imageDateAndTime':field.Field(PyCapsule_New(self._c_header.imageDateAndTime, "Field", NULL)),
            'targetId':field.Field(PyCapsule_New(self._c_header.targetId, "Field", NULL)),
            'imageTitle':field.Field(PyCapsule_New(self._c_header.imageTitle, "Field", NULL)),
            'imageSecurityClass':field.Field(PyCapsule_New(self._c_header.imageSecurityClass, "Field", NULL)),
            'securityGroup':FileSecurity(PyCapsule_New(self._c_header.securityGroup, "FileSecurity", NULL)),
            'encrypted':field.Field(PyCapsule_New(self._c_header.encrypted, "Field", NULL)),
            'imageSource':field.Field(PyCapsule_New(self._c_header.imageSource, "Field", NULL)),
            'numRows':field.Field(PyCapsule_New(self._c_header.numRows, "Field", NULL)),
            'numCols':field.Field(PyCapsule_New(self._c_header.numCols, "Field", NULL)),
            'pixelValueType':field.Field(PyCapsule_New(self._c_header.pixelValueType, "Field", NULL)),
            'imageRepresentation':field.Field(PyCapsule_New(self._c_header.imageRepresentation, "Field", NULL)),
            'imageCategory':field.Field(PyCapsule_New(self._c_header.imageCategory, "Field", NULL)),
            'actualBitsPerPixel':field.Field(PyCapsule_New(self._c_header.actualBitsPerPixel, "Field", NULL)),
            'pixelJustification':field.Field(PyCapsule_New(self._c_header.pixelJustification, "Field", NULL)),
            'imageCoordinateSystem':field.Field(PyCapsule_New(self._c_header.imageCoordinateSystem, "Field", NULL)),
            'cornerCoordinates':field.Field(PyCapsule_New(self._c_header.cornerCoordinates, "Field", NULL)),
            'numImageComments':field.Field(PyCapsule_New(self._c_header.numImageComments, "Field", NULL)),
            'imageComments':types.List(field.Field, PyCapsule_New(self._c_header.imageComments, "List", NULL)),
            'imageCompression':field.Field(PyCapsule_New(self._c_header.imageCompression, "Field", NULL)),
            'compressionRate':field.Field(PyCapsule_New(self._c_header.compressionRate, "Field", NULL)),
            'numImageBands':field.Field(PyCapsule_New(self._c_header.numImageBands, "Field", NULL)),
            'numMultispectralImageBands':field.Field(PyCapsule_New(self._c_header.numMultispectralImageBands, "Field", NULL)),
            'imageSyncCode':field.Field(PyCapsule_New(self._c_header.imageSyncCode, "Field", NULL)),
            'imageMode':field.Field(PyCapsule_New(self._c_header.imageMode, "Field", NULL)),
            'numBlocksPerRow':field.Field(PyCapsule_New(self._c_header.numBlocksPerRow, "Field", NULL)),
            'numBlocksPerCol':field.Field(PyCapsule_New(self._c_header.numBlocksPerCol, "Field", NULL)),
            'numPixelsPerHorizBlock':field.Field(PyCapsule_New(self._c_header.numPixelsPerHorizBlock, "Field", NULL)),
            'numPixelsPerVertBlock':field.Field(PyCapsule_New(self._c_header.numPixelsPerVertBlock, "Field", NULL)),
            'numBitsPerPixel':field.Field(PyCapsule_New(self._c_header.numBitsPerPixel, "Field", NULL)),
            'imageDisplayLevel':field.Field(PyCapsule_New(self._c_header.imageDisplayLevel, "Field", NULL)),
            'imageAttachmentLevel':field.Field(PyCapsule_New(self._c_header.imageAttachmentLevel, "Field", NULL)),
            'imageLocation':field.Field(PyCapsule_New(self._c_header.imageLocation, "Field", NULL)),
            'imageMagnification':field.Field(PyCapsule_New(self._c_header.imageMagnification, "Field", NULL)),
            'userDefinedImageDataLength':field.Field(PyCapsule_New(self._c_header.userDefinedImageDataLength, "Field", NULL)),
            'userDefinedOverflow':field.Field(PyCapsule_New(self._c_header.userDefinedOverflow, "Field", NULL)),
            'extendedHeaderLength':field.Field(PyCapsule_New(self._c_header.extendedHeaderLength, "Field", NULL)),
            'extendedHeaderOverflow':field.Field(PyCapsule_New(self._c_header.extendedHeaderOverflow, "Field", NULL)),
            'userDefinedSection':tre.Extensions(PyCapsule_New(self._c_header.userDefinedSection, "Extensions", NULL)),
            'extendedSection':tre.Extensions(PyCapsule_New(self._c_header.extendedSection, "Extensions", NULL)),
            }
        return tmp


cdef class DESubheader(BaseFieldHeader):
    cdef header.nitf_DESubheader* _c_header

    def __cinit__(self, c):
        assert(PyCapsule_IsValid(c, "DESubheader"))
        self._c_header = <header.nitf_DESubheader*>PyCapsule_GetPointer(c, "DESubheader")

    def _capsule(self):
        return PyCapsule_New(self._c_header, "DESubheader", NULL)

    @staticmethod
    cdef from_ptr(header.nitf_DESubheader* ptr):
        obj = DESubheader()
        obj._c_header = ptr
        return obj

    @deprecated("Old SWIG API")
    def getUDHD(self):
        return self.userDefinedSection

    equiv_items = {'NITF_DE': 'filePartType',
                   'NITF_DESTAG': 'typeId',
                   'NITF_DESVER': 'version',
                   'NITF_DESCLAS': 'securityClass',
                   'NITF_DESOFLW': 'overflowedHeaderType',
                   'NITF_DESITEM': 'dataItemOverflowed',
                   'NITF_DESSHL': 'subheaderFieldsLength',
                  }

    def get_items(self):
        tmp = {
            'filePartType':field.Field(PyCapsule_New(self._c_header.filePartType, "Field", NULL)),
            'typeId':field.Field(PyCapsule_New(self._c_header.typeID, "Field", NULL)),
            'version':field.Field(PyCapsule_New(self._c_header.version, "Field", NULL)),
            'securityGroup':FileSecurity(PyCapsule_New(self._c_header.securityGroup, "FileSecurity", NULL)),
            'overflowedHeaderType':field.Field(PyCapsule_New(self._c_header.overflowedHeaderType, "Field", NULL)),
            'dataItemOverflowed':field.Field(PyCapsule_New(self._c_header.dataItemOverflowed, "Field", NULL)),
            'subheaderFieldsLength':field.Field(PyCapsule_New(self._c_header.subheaderFieldsLength, "Field", NULL)),
            'subheaderFields':tre.TRE(capsule=PyCapsule_New(self._c_header.subheaderFields, "TRE", NULL)) if self._c_header.subheaderFields is not NULL else None,
            'userDefinedSection':tre.Extensions(PyCapsule_New(self._c_header.userDefinedSection, "Extensions", NULL)),
        }
        return tmp

    def __setitem__(self, key, value):
        if key == 'subheaderFields':  # special handling since it can be NULL and PyCapsule doesn't like that
            if self._c_header.subheaderFields is not NULL:
                tre.nitf_TRE_destruct(&self._c_header.subheaderFields)
            self._c_header.subheaderFields = <tre.nitf_TRE*>PyCapsule_GetPointer(value._capsule(), "TRE")
        else:
            super().__setitem__(key, value)


cdef class TextSubheader(BaseFieldHeader):
    cdef header.nitf_TextSubheader* _c_header

    def __cinit__(self, c):
        assert(PyCapsule_IsValid(c, "TextSubheader"))
        self._c_header = <header.nitf_TextSubheader*>PyCapsule_GetPointer(c, "TextSubheader")

    def _capsule(self):
        return PyCapsule_New(self._c_header, "TextSubheader", NULL)

    @staticmethod
    cdef from_ptr(header.nitf_TextSubheader* ptr):
        obj = TextSubheader()
        obj._c_header = ptr
        return obj

    @deprecated("Old SWIG API")
    def getUDHD(self):
        return self.userDefinedSection

    equiv_items = {'TE': 'filePartType',
                   'TEXTID': 'textID',
                   'TXTALVL': 'attachmentLevel',
                   'TXTDT': 'dateTime',
                   'TXTITL': 'title',
                   'classification': 'securityClass',
                   'TSCLAS': 'securityClass',
                   'ENCRYP': 'encrypted',
                   'TXTFMT': 'format',
                   'TXSHDL': 'extendedHeaderLength',
                   'TXSOFL': 'extendedHeaderOverflow',
                   'TXSHD': 'extendedSection',
                   }

    def get_items(self):
        tmp = {
            'filePartType':field.Field(PyCapsule_New(self._c_header.filePartType, "Field", NULL)),
            'textID':field.Field(PyCapsule_New(self._c_header.textID, "Field", NULL)),
            'attachmentLevel':field.Field(PyCapsule_New(self._c_header.attachmentLevel, "Field", NULL)),
            'dateTime':field.Field(PyCapsule_New(self._c_header.dateTime, "Field", NULL)),
            'title':field.Field(PyCapsule_New(self._c_header.title, "Field", NULL)),
            'securityClass':field.Field(PyCapsule_New(self._c_header.securityClass, "Field", NULL)),
            'securityGroup':FileSecurity(PyCapsule_New(self._c_header.securityGroup, "FileSecurity", NULL)),
            'encrypted':field.Field(PyCapsule_New(self._c_header.encrypted, "Field", NULL)),
            'format':field.Field(PyCapsule_New(self._c_header.format, "Field", NULL)),
            'extendedHeaderLength':field.Field(PyCapsule_New(self._c_header.extendedHeaderLength, "Field", NULL)),
            'extendedHeaderOverflow':field.Field(PyCapsule_New(self._c_header.extendedHeaderOverflow, "Field", NULL)),
            'extendedSection':tre.Extensions(PyCapsule_New(self._c_header.extendedSection, "Extensions", NULL)),
        }
        return tmp
