#ifndef __ZIP_ZIP_ENTRY_H__
#define __ZIP_ZIP_ENTRY_H__


#include "zip/Types.h"

namespace zip
{
    /*!
     *  \class ZipEntry
     *  \brief Each entry in a ZipFile is a ZipEntry
     *
     *  Class stores the information about individual elements
     *  in a PKZIP zip file
     */
    class ZipEntry
    {
	enum CompressionMethod { COMP_STORED = 0, COMP_DEFLATED = 8 };

        unsigned short mVersionMadeBy;
        unsigned short mVersionToExtract;
        unsigned short mGeneralPurposeBitFlag;
	unsigned short mCompressionMethod;
        unsigned short mLastModifiedTime;
        unsigned short mLastModifiedDate;
        unsigned int   mCRC32;
        unsigned short mInternalAttrs;
        unsigned int mExternalAttrs;

	sys::ubyte* mCompressedData;
	sys::Size_T mCompressedSize;
	sys::Size_T mUncompressedSize;
	std::string mFileName;
	std::string mFileComment;

	static void inflate(sys::ubyte* out, sys::Size_T outLen,
			    sys::ubyte* in, sys::Size_T inLen);

    public:

	ZipEntry(sys::ubyte* compressedData,
		 sys::Size_T compressedSize,
		 sys::Size_T uncompressedSize,
		 std::string fileName,
		 std::string fileComment,
                 unsigned short versionMadeBy,
                 unsigned short versionToExtract,
                 unsigned short generalPurposeBitFlag,
                 unsigned short compressionMethod,
                 unsigned short lastModifiedTime,
                 unsigned short lastModifiedDate,
                 unsigned int   crc32,
                 unsigned short internalAttrs,
                 unsigned int externalAttrs) :
            mCompressedData(compressedData),
            mCompressedSize(compressedSize),
            mUncompressedSize(uncompressedSize),
            mFileName(fileName),
            mFileComment(fileComment),
            mVersionMadeBy(versionMadeBy),
            mVersionToExtract(versionToExtract),
            mGeneralPurposeBitFlag(generalPurposeBitFlag),
            mCompressionMethod(compressionMethod),
            mLastModifiedTime(lastModifiedTime),
            mLastModifiedDate(lastModifiedDate),
            mCRC32(crc32),
            mInternalAttrs(internalAttrs),
            mExternalAttrs(externalAttrs)
	{
	}
	
	~ZipEntry() {}

	sys::ubyte* decompress();
	void decompress(sys::ubyte* out, sys::Size_T outLen);

        unsigned short getVersionMadeBy() const
        {
            return mVersionMadeBy;
        }
        const char* getVersionMadeByString() const;

        unsigned short getVersionToExtract() const
        {
            return mVersionToExtract;
        }
        unsigned short getGeneralPurposeBitFlag() const
        {
            return mGeneralPurposeBitFlag;
        }
	unsigned short getCompressionMethod() const
        {
            return mCompressionMethod;
        }
        unsigned short getLastModifiedTime() const
        {
            return mLastModifiedTime;
        }

        unsigned short getLastModifiedDate() const
        {
            return mLastModifiedDate;
        }
        unsigned int getCRC32() const
        {
            return mCRC32;
        }
        unsigned short getInternalAttrs() const
        {
            return mInternalAttrs;
        }
        unsigned int getExternalAttrs() const
        {
            return mExternalAttrs;
        }

	std::string getFileName() const 
        { 
            return mFileName; 
        }
	std::string getFileComment() const 
        { 
            return mFileComment; 
        }
	sys::Size_T getUncompressedSize() const 
        { 
            return mUncompressedSize; 
        }
	sys::Size_T getCompressedSize() const 
        { 
            return mCompressedSize; 
        }
    };


}

/*!
 *  Output stream overload
 */
std::ostream& operator<<(std::ostream& os, const zip::ZipEntry& ze);


#endif
