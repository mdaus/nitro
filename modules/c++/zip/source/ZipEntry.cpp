#include "zip/ZipEntry.h"

using namespace zip;

void ZipEntry::inflate(sys::ubyte* out, sys::Size_T outLen,
		       sys::ubyte* in, sys::Size_T inLen)
{
    z_stream zstream;
    unsigned long crc;
    memset(&zstream, 0, sizeof(zstream));
    zstream.zalloc = Z_NULL;
    zstream.zfree = Z_NULL;
    zstream.opaque = Z_NULL;
    zstream.next_in = in;
    zstream.avail_in = inLen;
    zstream.next_out = (Bytef*)out;
    zstream.avail_out = outLen;
    zstream.data_type = Z_UNKNOWN;
    
    int zerr = inflateInit2(&zstream, -MAX_WBITS);
    if (zerr != Z_OK)
    {
	throw except::IOException(
	    Ctxt(
		FmtX("inflateInit2 failed [%d]",
		     zerr)
		)
	    );
    }
    
    // decompress
    zerr = ::inflate(&zstream, Z_FINISH);
    
    if (zerr != Z_STREAM_END)
    {
	throw except::IOException(
	    Ctxt(
		FmtX("inflate failed [%d]: wanted: %d, got: %lu",
		     zerr,
		     Z_STREAM_END,
		     zstream.total_out)
		)
	    );
    }
    inflateEnd(&zstream);
    
}


sys::ubyte* ZipEntry::decompress()
{
    sys::ubyte* uncompressed = new sys::ubyte[mUncompressedSize];
    decompress(uncompressed, mUncompressedSize);
    return uncompressed;
}


void ZipEntry::decompress(sys::ubyte* out, sys::Size_T outLen)
{
    
    if (mCompressionMethod == COMP_STORED)
    {
	memcpy(out, mCompressedData, outLen);
    }
    else
    {
	inflate(out, outLen, mCompressedData, mCompressedSize);
    }
    
}

