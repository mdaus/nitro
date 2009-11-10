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
	sys::ubyte* mCompressedData;
	sys::Size_T mCompressedSize;
	sys::Size_T mUncompressedSize;
	std::string mFileName;
	std::string mFileComment;
	unsigned short mCompressionMethod;

	static void inflate(sys::ubyte* out, sys::Size_T outLen,
			    sys::ubyte* in, sys::Size_T inLen);

    public:

	ZipEntry(sys::ubyte* compressedData,
		 sys::Size_T compressedSize,
		 sys::Size_T uncompressedSize,
		 std::string fileName,
		 std::string fileComment,
		 unsigned short compressionMethod) :
	    mCompressedData(compressedData),
	    mCompressedSize(compressedSize),
	    mUncompressedSize(uncompressedSize),
	    mFileName(fileName),
	    mFileComment(fileComment),
	    mCompressionMethod(compressionMethod) 
	{
	}
	
	~ZipEntry() {}

	sys::ubyte* decompress();
	void decompress(sys::ubyte* out, sys::Size_T outLen);

	std::string getFileName() const { return mFileName; }
	std::string getFileComment() const { return mFileComment; }
	sys::Size_T getUncompressedSize() const { return mUncompressedSize; }
	sys::Size_T getCompressedSize() const { return mCompressedSize; }
    };


}

#endif
