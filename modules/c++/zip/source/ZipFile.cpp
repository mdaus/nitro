#include "zip/ZipFile.h"

using namespace zip;

#define Z_READ_SHORT_INC(BUF, OFF) readShort(&BUF[OFF]); OFF += 2
#define Z_READ_INT_INC(BUF, OFF) readInt(&BUF[OFF]); OFF += 4

ZipFile::~ZipFile()
{
    for (unsigned int i = 0; i < mEntries.size(); ++i)
    {
	// Delete ZipEntry
	delete mEntries[i];
    }

    if (mCompressed)
	delete [] mCompressed;
}


unsigned int ZipFile::readInt(sys::ubyte* buf)
{
    unsigned int le = *((unsigned int*)buf);
    // Kind of hackish, but we need it like yesterday
    
    if (mSwapBytes)
	le = sys::byteSwap<unsigned int>(le);
    return le;
}

unsigned short ZipFile::readShort(sys::ubyte* buf)
{
    unsigned short le;
    if (mSwapBytes)
	le = sys::byteSwap<unsigned short>(le);
    return le;
}


ZipFile::Iterator ZipFile::lookup(std::string fileName) const
{
    ZipFile::Iterator p;
    for (p = mEntries.begin(); p != mEntries.end(); ++p)
    {

	if ( (*p)->getFileName() == fileName)
	    break;
    }
    
    return p;
    
}

void ZipFile::readCentralDir()
{
    
    if (mCompressedLength < EOCD_LEN)
	throw except::IOException(
	    Ctxt("stream source too small to be a zip stream")
	    );
    
    sys::ubyte* start;
    sys::ubyte* eocd;
    
    if (mCompressedLength > MAX_EOCD_SEARCH)
	start = mCompressed + mCompressedLength - MAX_EOCD_SEARCH;

    else
	start = mCompressed;
    
    sys::ubyte* p = mCompressed + mCompressedLength - 4;
    
    while( p >= start )
    {
	if (*p == 0x50)
	{
	    if (readInt(p) == CD_SIGNATURE)
	    {
		eocd = p;
		break;
	    }
	}
	p--;
    }
    if ( p < start)
    {
	throw except::IOException(Ctxt("EOCD not found"));
    }
    
    // else still rockin'
    readCentralDirValues(eocd, (mCompressed + mCompressedLength) - eocd);
    
    p = mCompressed + mCentralDirOffset;
    sys::SSize_T len = (mCompressed + mCompressedLength) - p;
    
    for (unsigned int i = 0; i < mEntries.size(); ++i)
    {
	mEntries[i] = newCentralDirEntry(&p, len);
    }
}

ZipEntry* ZipFile::newCentralDirEntry(sys::ubyte** buf, sys::SSize_T len)
{
    if (len < ENTRY_LEN)
	throw except::IOException(Ctxt("CDE entry not large enough"));
    
    sys::SSize_T off = 0;

    sys::ubyte* p = *buf;

    unsigned int entrySig = Z_READ_INT_INC(p, off);

    if (entrySig != ENTRY_SIGNATURE)
	throw except::IOException(
	    Ctxt("Did not find entry signature")
	    );
    unsigned short versionMadeBy     = Z_READ_SHORT_INC(p, off);
    unsigned short versionToExtract  = Z_READ_SHORT_INC(p, off);
    unsigned short gpBitFlag         = Z_READ_SHORT_INC(p, off);
    unsigned short compressionMethod = Z_READ_SHORT_INC(p, off);
    unsigned short lastModTime       = Z_READ_SHORT_INC(p, off);
    unsigned short lastModDate       = Z_READ_SHORT_INC(p, off);
    unsigned int crc32               = Z_READ_INT_INC(p, off);
    unsigned int compressedSize      = Z_READ_INT_INC(p, off);
    unsigned int uncompressedSize    = Z_READ_INT_INC(p, off);
    unsigned short fileNameLength    = Z_READ_SHORT_INC(p, off);
    unsigned short extraFieldLength  = Z_READ_SHORT_INC(p, off);
    unsigned short fileCommentLength = Z_READ_SHORT_INC(p, off);
    unsigned short diskNumberStart   = Z_READ_SHORT_INC(p, off);
    unsigned short internalAttrs     = Z_READ_SHORT_INC(p, off);
    unsigned short externalAttrs     = Z_READ_INT_INC(p, off);
    unsigned int localHeaderRelOffset = readInt(&p[off]);
    p += ENTRY_LEN;
    
    std::string fileName;
    if (fileNameLength != 0)
	fileName = std::string((const char*)p, fileNameLength);

    p += fileNameLength;
    
    // I dont know what extra field is, but skip it for now	    
    if (extraFieldLength)
	p += extraFieldLength;
    
    std::string fileComment;
    if (fileCommentLength)
	fileComment = std::string((const char*)p, fileCommentLength);
    
    p += fileCommentLength;
    
    *buf = p;
    
    p = mCompressed + localHeaderRelOffset;
    
    extraFieldLength = readShort(&p[0x1c]);
    
    sys::Size_T dataOffset = localHeaderRelOffset + LFH_SIZE
	+ fileNameLength + extraFieldLength;
    

    return new ZipEntry(mCompressed + dataOffset,
			compressedSize,
			uncompressedSize,
			fileName,
			fileComment,
			compressionMethod);
    
}

void ZipFile::readCentralDirValues(sys::ubyte* buf, sys::SSize_T len)
{
    
    if (len < EOCD_LEN)
	throw except::IOException(Ctxt("len < EOCD_LEN"));
    
    unsigned short off = 4;
    
    unsigned short diskNum = Z_READ_SHORT_INC(buf, off);
    if (diskNum != 0)
	throw except::IOException(Ctxt("disk number must be 0"));
    
    unsigned short diskWithCentralDir = Z_READ_SHORT_INC(buf, off);

    if (diskWithCentralDir != 0)
	throw except::IOException(
	    Ctxt("central dir disk number must be 0")
	    );
    
    unsigned short entryCount = Z_READ_SHORT_INC(buf, off);
    unsigned short totalEntries = Z_READ_SHORT_INC(buf, off);

    if (totalEntries != entryCount)
	throw except::IOException(Ctxt("Total entries must match entries"));
    
    mEntries.resize( entryCount );

    mCentralDirSize = Z_READ_INT_INC(buf, off);
    mCentralDirOffset = Z_READ_INT_INC(buf, off);

    unsigned short commentLength = Z_READ_SHORT_INC(buf, off);
    
    if ( EOCD_LEN + commentLength > len)
	throw except::IOException(Ctxt("Comment line too long"));
    
    mComment = std::string((const char*)(buf + EOCD_LEN), commentLength);
    
}

// void ZipFile::copyString(const sys::ubyte* buf, sys::SSize_T len)
// {
//   std::string s(buf, len);
//   /* 	    char* cstr = new char[len + 1]; */
//   /* 	    cstr[len] = 0; */
//   /* 	    memcpy(cstr, buf + EOCD_LEN, len); */
//   /* 	    std::string s = cstr; */
//   /* 	    delete [] cstr; */
//   return s;
// }
std::ostream& operator<<(std::ostream& os, const zip::ZipFile& zf)
{

    for (zip::ZipFile::Iterator p = zf.begin(); p != zf.end(); ++p)
    {
	os << "File: " << (*p)->getFileName() << std::endl;
	os << "\tstored size: " << (*p)->getCompressedSize() << std::endl;
	os << "\tactual size: " << (*p)->getUncompressedSize() << std::endl;
    }
    return os;
}
