from types cimport *
from error cimport nitf_Error

cdef extern from "nitf/Field.h":
    ctypedef enum nitf_FieldType:
        pass

    ctypedef enum nitf_ConvType:
        pass

    ctypedef struct nitf_Field:
        nitf_FieldType type
        char* raw
        size_t length
        NITF_BOOL resizable

    NITF_BOOL nitf_Field_setRawData(nitf_Field * field, NITF_DATA * data, size_t dataLength, nitf_Error * error);
    NITF_BOOL nitf_Field_setUint32(nitf_Field * field, nitf_Uint32 number, nitf_Error * error);
    NITF_BOOL nitf_Field_setUint64(nitf_Field * field, nitf_Uint64 number, nitf_Error * error);
    NITF_BOOL nitf_Field_setInt32(nitf_Field * field, nitf_Int32 number, nitf_Error * error);
    NITF_BOOL nitf_Field_setInt64(nitf_Field * field, nitf_Int64 number, nitf_Error * error);
    # void nitf_Field_trimString(char *str);
    NITF_BOOL nitf_Field_setString(nitf_Field * field, const char *str, nitf_Error * error);
    # NITF_BOOL nitf_Field_setDateTime(nitf_Field * field, const nitf_DateTime *dateTime, const char *format, nitf_Error * error);
    NITF_BOOL nitf_Field_setReal(nitf_Field * field, const char *type, NITF_BOOL plus, double value, nitf_Error *error);
    NITF_BOOL nitf_Field_get(nitf_Field * field, NITF_DATA * outValue, nitf_ConvType convType, size_t length, nitf_Error * error);
