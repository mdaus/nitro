#cython: language_level=3, c_string_type=unicode, c_string_encoding=ascii

from error cimport nitf_Error


class NitfError(BaseException):
    def __init__(self, nitf_error):
        if type(nitf_error['message']) in [bytes, bytearray]:
            super().__init__(nitf_error['message'].decode(errors='replace'))
        else:
            super().__init__(nitf_error['message'])
        self.nitf_error = nitf_error

    def __str__(self):
        return "NitfError at {file}:{line} :: {func}: {message}".format(**self.nitf_error)


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

