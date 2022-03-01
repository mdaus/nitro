/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
 * (C) Copyright 2022, Maxar Technologies, Inc.
 *
 * NITRO is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, If not,
 * see <http://www.gnu.org/licenses/>.
 *
 */

#include "nitf/J2KTileWriter.hpp"

#include <stdexcept>
#include <limits>
#include <sstream>

#include <gsl/gsl.h>
#include <except/Exception.h>
#include <sys/Conf.h>
#include <mem/ScopedArray.h>
#include <io/ByteStream.h>

#include "j2k/j2k_TileWriter.h"

#undef min
#undef max

namespace
{
    size_t writeImpl(void* buffer, size_t numBytes, void* data)
    {
        try
        {
            ::io::SeekableOutputStream* compressedOutputStream =
                static_cast<::io::SeekableOutputStream*>(data);

            compressedOutputStream->write(buffer, numBytes);
        }
        catch (const except::Exception& ex)
        {
            // Openjpeg expects (OPJ_SIZE_T)-1 as the result of a failed
            // call to a user provided write.
            return std::numeric_limits<size_t>::max();
        }
        return numBytes;
    }

    sys::Off_T skipImpl(sys::Off_T bytesToSkip, void* data)
    {
        try
        {
            ::io::SeekableOutputStream* compressedOutputStream =
                static_cast<::io::SeekableOutputStream*>(data);

            compressedOutputStream->seek(bytesToSkip, ::io::Seekable::CURRENT);
        }
        catch (const except::Exception& ex)
        {
            // Openjpeg expects -1 as the result of a failed call to a user
            // provided skip
            return -1;
        }

        return bytesToSkip;
    }

    bool seekImpl(int64_t numBytes, void* data)
    {
        try
        {
            ::io::SeekableOutputStream* compressedOutputStream =
                static_cast<::io::SeekableOutputStream*>(data);

            compressedOutputStream->seek(numBytes, ::io::Seekable::START);
        }
        catch (const except::Exception& ex)
        {
            // Openjpeg expects 0 (OPJ_FALSE) as the result of a failed call
            // to a user provided seek
            return false;
        }

        return true;
    }
}

j2k::impl::TileWriter::TileWriter(
    std::shared_ptr< ::io::SeekableOutputStream> outputStream,
    std::shared_ptr<const CompressionParameters> compressionParams) :
    mCompressionParams(compressionParams),
    mStream(j2k::StreamType::OUTPUT),
    mImage(mCompressionParams->getRawImageDims()),
    mEncoder(mImage, *mCompressionParams),
    mIsCompressing(false)
{
    setOutputStream(outputStream);
    j2k_stream_set_write_function(mStream.getNative(), writeImpl);
    j2k_stream_set_seek_function(mStream.getNative(), seekImpl);
    j2k_stream_set_skip_function(mStream.getNative(), skipImpl);
}

j2k::impl::TileWriter::~TileWriter()
{
    try
    {
        end();
    }
    catch (...)
    {
    }
}

void j2k::impl::TileWriter::start()
{
    if (!mIsCompressing)
    {
        const int startCompressSuccess = opj_start_compress(mEncoder.getNative(), mImage.getNative(), mStream.getNative());
        if (!startCompressSuccess)
        {
            if (mEncoder.errorOccurred())
            {
                const std::string opjErrorMsg = mEncoder.getErrorMessage();
                mEncoder.clearError();

                throw except::Exception(
                    Ctxt("Error starting compression "
                        "with openjpeg error: " + opjErrorMsg));
            }

            throw except::Exception(Ctxt("Error starting compression."));
        }

        mIsCompressing = true;
    }
}

void j2k::impl::TileWriter::end()
{
    if (mIsCompressing)
    {
        const int endCompressSuccess = opj_end_compress(mEncoder.getNative(), mStream.getNative());
        if (!endCompressSuccess)
        {
            if (mEncoder.errorOccurred())
            {
                const std::string opjErrorMsg = mEncoder.getErrorMessage();
                mEncoder.clearError();

                throw except::Exception(
                    Ctxt("Error ending compression "
                        "with openjpeg error: " + opjErrorMsg));
            }

            throw except::Exception(Ctxt("Error ending compression."));
        }

        mIsCompressing = false;
    }
}

void j2k::impl::TileWriter::flush()
{
    if (!mIsCompressing)
    {
        throw except::Exception(
            Ctxt("Cannot flush data to output stream: compression "
                "has not been started."));
    }

    const int flushSuccess = opj_flush(mEncoder.getNative(), mStream.getNative());
    if (!flushSuccess)
    {
        if (mEncoder.errorOccurred())
        {
            const std::string opjErrorMsg = mEncoder.getErrorMessage();
            mEncoder.clearError();

            throw except::Exception(Ctxt("Failed to flush J2K codestream data with openjpeg error: " + opjErrorMsg));
        }

        throw except::Exception(Ctxt("Failed to flush J2K codestream data."));
    }
}

void j2k::impl::TileWriter::writeTile(const sys::ubyte* tileData, size_t tileIndex)
{
    start();

    const types::RowCol<size_t> tileDims(mCompressionParams->getTileDims());

    // Resize of the dimensions of this tile if it is a partial tile
    types::RowCol<size_t> resizedTileDims(tileDims);
    resizeTile(resizedTileDims, tileIndex);

    // Create a smaller buffer for our partial tile
    mem::ScopedArray<sys::ubyte> partialTileBuffer;
    if (resizedTileDims.col < tileDims.col ||
        resizedTileDims.row < tileDims.row)
    {
        partialTileBuffer.reset(new sys::ubyte[resizedTileDims.area()]);
        for (size_t row = 0; row < resizedTileDims.row; ++row)
        {
            const sys::ubyte* srcTileRowStart = tileData + row * tileDims.col;

            std::copy(srcTileRowStart,
                srcTileRowStart + resizedTileDims.col,
                partialTileBuffer.get() + row * resizedTileDims.col);
        }
    }

    sys::ubyte* imageData = partialTileBuffer.get() ?
        partialTileBuffer.get() : const_cast<sys::ubyte*>(tileData);

    // Compress the tile - if an I/O error occurs in our write handler,
    // the OPJEncoder error handler will get called.
    const int writeSuccess = opj_write_tile(mEncoder.getNative(),
        tileIndex,
        imageData,
        resizedTileDims.area(),
        mStream.getNative());
    if (!writeSuccess)
    {
        std::ostringstream os;
        os << "Failed to compress tile " << tileIndex
            << " (rows: " << resizedTileDims.row << ", cols: "
            << resizedTileDims.col << ")";
        if (mEncoder.errorOccurred())
        {
            const std::string opjErrorMsg = mEncoder.getErrorMessage();
            mEncoder.clearError();

            os << " with openjpeg error: " << opjErrorMsg;
            throw except::Exception(Ctxt(os.str()));
        }

        throw except::Exception(Ctxt(os.str()));
    }
}

void j2k::impl::TileWriter::setOutputStream(std::shared_ptr<::io::SeekableOutputStream> outputStream)
{
    mOutputStream = outputStream;
    opj_stream_set_user_data(mStream.getNative(),
        mOutputStream.get(),
        NULL);

}

void j2k::impl::TileWriter::resizeTile(types::RowCol<size_t>& tile, size_t tileIndex)
{
    const types::RowCol<size_t> tileDims =
        mCompressionParams->getTileDims();

    const types::RowCol<size_t> rawImageDims =
        mCompressionParams->getRawImageDims();

    const size_t numColsOfTiles = mCompressionParams->getNumColsOfTiles();
    const size_t numRowsOfTiles = mCompressionParams->getNumRowsOfTiles();

    const size_t tileRow = tileIndex / numColsOfTiles;
    if (tileRow == numRowsOfTiles - 1 &&
        rawImageDims.row % tileDims.row != 0)
    {
        tile.row = rawImageDims.row % tileDims.row;
    }

    const size_t tileCol = tileIndex - (tileRow * numColsOfTiles);
    if (tileCol == numColsOfTiles - 1 &&
        rawImageDims.col % tileDims.col != 0)
    {
        tile.col = rawImageDims.col % tileDims.col;
    }
}
