from error cimport nitf_Error
from types cimport *
from dataextension_segment cimport nitf_DESegment
from image_segment cimport nitf_ImageSegment
from header cimport nitf_FileHeader

cdef extern from "nitf/Record.h":
    ctypedef struct nitf_Record:
        nitf_FileHeader *header
        nitf_List *images
        nitf_List *graphics;
        nitf_List *labels;
        nitf_List *texts;
        nitf_List *dataExtensions;
        nitf_List *reservedExtensions;

    ctypedef enum nitf_Version:
        pass

    nitf_Record *nitf_Record_construct(nitf_Version version, nitf_Error * error);
    # nitf_Record * nitf_Record_clone(const nitf_Record * source, nitf_Error * error);
    void nitf_Record_destruct(nitf_Record ** record);
    nitf_Version nitf_Record_getVersion(const nitf_Record * record);
    nitf_Uint32 nitf_Record_getNumImages(const nitf_Record* record, nitf_Error* error);
    nitf_ImageSegment* nitf_Record_newImageSegment(nitf_Record * record, nitf_Error * error);
    nitf_Uint32 nitf_Record_getNumGraphics(const nitf_Record* record, nitf_Error* error);
    # nitf_GraphicSegment* nitf_Record_newGraphicSegment(nitf_Record * record, nitf_Error * error);
    nitf_Uint32 nitf_Record_getNumTexts(const nitf_Record* record, nitf_Error* error);
    # nitf_TextSegment* nitf_Record_newTextSegment(nitf_Record * record, nitf_Error * error);
    nitf_Uint32 nitf_Record_getNumDataExtensions(const nitf_Record* record, nitf_Error* error);
    nitf_DESegment* nitf_Record_newDataExtensionSegment(nitf_Record * record, nitf_Error * error);
    # NITF_BOOL nitf_Record_removeImageSegment(nitf_Record * record, nitf_Uint32 segmentNumber, nitf_Error * error);
    # NITF_BOOL nitf_Record_removeGraphicSegment(nitf_Record * record, nitf_Uint32 segmentNumber, nitf_Error * error);
    nitf_Uint32 nitf_Record_getNumLabels(const nitf_Record* record, nitf_Error* error);
    # NITF_BOOL nitf_Record_removeLabelSegment(nitf_Record * record, nitf_Uint32 segmentNumber, nitf_Error * error);
    # NITF_BOOL nitf_Record_removeTextSegment(nitf_Record * record, nitf_Uint32 segmentNumber, nitf_Error * error);
    # NITF_BOOL nitf_Record_removeDataExtensionSegment(nitf_Record * record, nitf_Uint32 segmentNumber, nitf_Error * error);
    nitf_Uint32 nitf_Record_getNumReservedExtensions(const nitf_Record* record, nitf_Error* error);
    # NITF_BOOL nitf_Record_removeReservedExtensionSegment(nitf_Record * record, nitf_Uint32 segmentNumber, nitf_Error * error);
    # NITF_BOOL nitf_Record_moveImageSegment(nitf_Record * record, nitf_Uint32 oldIndex, nitf_Uint32 newIndex, nitf_Error * error);
    # NITF_BOOL nitf_Record_moveGraphicSegment(nitf_Record * record, nitf_Uint32 oldIndex, nitf_Uint32 newIndex, nitf_Error * error);
    # NITF_BOOL nitf_Record_moveLabelSegment(nitf_Record * record, nitf_Uint32 oldIndex, nitf_Uint32 newIndex, nitf_Error * error);
    # NITF_BOOL nitf_Record_moveTextSegment(nitf_Record * record, nitf_Uint32 oldIndex, nitf_Uint32 newIndex, nitf_Error * error);
    # NITF_BOOL nitf_Record_moveDataExtensionSegment(nitf_Record * record, nitf_Uint32 oldIndex, nitf_Uint32 newIndex, nitf_Error * error);
    # NITF_BOOL nitf_Record_moveReservedExtensionSegment(nitf_Record * record, nitf_Uint32 oldIndex, nitf_Uint32 newIndex, nitf_Error * error);
    # NITF_BOOL nitf_Record_mergeTREs(nitf_Record * record, nitf_Error * error);
    # NITF_BOOL nitf_Record_unmergeTREs(nitf_Record * record, nitf_Error * error);

