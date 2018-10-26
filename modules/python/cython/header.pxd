from field cimport nitf_Field
from types cimport *
from error cimport nitf_Error
from tre cimport nitf_TRE, nitf_Extensions


cdef extern from "nitf/BandInfo.h":
    ctypedef struct nitf_BandInfo:
        nitf_Field *representation
        nitf_Field *subcategory
        nitf_Field *imageFilterCondition
        nitf_Field *imageFilterCode
        nitf_Field *numLUTs


cdef extern from "nitf/FileHeader.h":
    ctypedef struct nitf_FileHeader:
        nitf_Field *fileHeader
        nitf_Field *fileVersion
        nitf_Field *complianceLevel
        nitf_Field *systemType
        nitf_Field *originStationID
        nitf_Field *fileDateTime
        nitf_Field *fileTitle
        nitf_Field *classification
        nitf_FileSecurity *securityGroup;
        nitf_Field *messageCopyNum
        nitf_Field *messageNumCopies
        nitf_Field *encrypted
        nitf_Field *backgroundColor
        nitf_Field *originatorName
        nitf_Field *originatorPhone
        nitf_Field *fileLength
        nitf_Field *headerLength
        nitf_Field *numImages
        nitf_Field *numGraphics
        nitf_Field *numLabels
        nitf_Field *numTexts
        nitf_Field *numDataExtensions
        nitf_Field *numReservedExtensions
        # nitf_ComponentInfo **imageInfo;
        # nitf_ComponentInfo **graphicInfo;
        # nitf_ComponentInfo **labelInfo;
        # nitf_ComponentInfo **textInfo;
        # nitf_ComponentInfo **dataExtensionInfo;
        # nitf_ComponentInfo **reservedExtensionInfo;
        nitf_Field *userDefinedHeaderLength
        nitf_Field *userDefinedOverflow
        nitf_Field *extendedHeaderLength
        nitf_Field *extendedHeaderOverflow
        nitf_Extensions *userDefinedSection
        nitf_Extensions *extendedSection

cdef extern from "nitf/FileSecurity.h":
    ctypedef struct nitf_FileSecurity:
        nitf_Field *classificationSystem
        nitf_Field *codewords
        nitf_Field *controlAndHandling
        nitf_Field *releasingInstructions
        nitf_Field *declassificationType
        nitf_Field *declassificationDate
        nitf_Field *declassificationExemption
        nitf_Field *downgrade
        nitf_Field *downgradeDateTime
        nitf_Field *classificationText
        nitf_Field *classificationAuthorityType
        nitf_Field *classificationAuthority
        nitf_Field *classificationReason
        nitf_Field *securitySourceDate
        nitf_Field *securityControlNumber


cdef extern from "nitf/ImageSubheader.h":
    ctypedef struct nitf_ImageSubheader:
        nitf_Field *filePartType
        nitf_Field *imageId
        nitf_Field *imageDateAndTime
        nitf_Field *targetId
        nitf_Field *imageTitle
        nitf_Field *imageSecurityClass
        nitf_FileSecurity *securityGroup
        nitf_Field *encrypted
        nitf_Field *imageSource
        nitf_Field *numRows
        nitf_Field *numCols
        nitf_Field *pixelValueType
        nitf_Field *imageRepresentation
        nitf_Field *imageCategory
        nitf_Field *actualBitsPerPixel
        nitf_Field *pixelJustification
        nitf_Field *imageCoordinateSystem
        nitf_Field *cornerCoordinates
        nitf_Field *numImageComments
        nitf_List* imageComments
        nitf_Field *imageCompression
        nitf_Field *compressionRate
        nitf_Field *numImageBands
        nitf_Field *numMultispectralImageBands
        nitf_BandInfo **bandInfo
        nitf_Field *imageSyncCode
        nitf_Field *imageMode
        nitf_Field *numBlocksPerRow
        nitf_Field *numBlocksPerCol
        nitf_Field *numPixelsPerHorizBlock
        nitf_Field *numPixelsPerVertBlock
        nitf_Field *numBitsPerPixel
        nitf_Field *imageDisplayLevel
        nitf_Field *imageAttachmentLevel
        nitf_Field *imageLocation
        nitf_Field *imageMagnification
        nitf_Field *userDefinedImageDataLength
        nitf_Field *userDefinedOverflow
        nitf_Field *extendedHeaderLength
        nitf_Field *extendedHeaderOverflow
        nitf_Extensions *userDefinedSection
        nitf_Extensions *extendedSection

    int nitf_ImageSubheader_insertImageComment(nitf_ImageSubheader* subhdr, const char* comment, int position, nitf_Error *error)
    NITF_BOOL nitf_ImageSubheader_createBands(nitf_ImageSubheader * subhdr, nitf_Uint32 numBands, nitf_Error * error)
    nitf_Uint32 nitf_ImageSubheader_getBandCount(nitf_ImageSubheader* subhdr, nitf_Error* error)

cdef extern from "nitf/DESubheader.h":
    ctypedef struct nitf_DESubheader:
        nitf_Field *filePartType;
        nitf_Field *typeID;
        nitf_Field *version;
        nitf_Field *securityClass;
        nitf_FileSecurity *securityGroup;
        nitf_Field *overflowedHeaderType;
        nitf_Field *dataItemOverflowed;
        nitf_Field *subheaderFieldsLength;
        nitf_TRE *subheaderFields;
        nitf_Uint64 dataLength;
        nitf_Extensions *userDefinedSection

cdef extern from "nitf/TextSubheader.h":
    ctypedef struct nitf_TextSubheader:
        nitf_Field *filePartType;
        nitf_Field *textID;
        nitf_Field *attachmentLevel;
        nitf_Field *dateTime;
        nitf_Field *title;
        nitf_Field *securityClass;
        nitf_FileSecurity *securityGroup;
        nitf_Field *encrypted;
        nitf_Field *format;
        nitf_Field *extendedHeaderLength;
        nitf_Field *extendedHeaderOverflow;
        nitf_Extensions *extendedSection;
