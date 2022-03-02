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

#include "nitf/J2KCompressor.hpp"

#include <string.h>

#include <numeric>
#include <stdexcept>
#include <limits>
#include <sstream>
#include <vector>
#include <algorithm>
#include <std/span>
#include <iterator>
#include <std/memory>

#include <gsl/gsl.h>
#include <except/Exception.h>
#include <sys/Conf.h>
#include <io/ByteStream.h>
#include <math/Round.h>
#include <nitf/ImageBlocker.hpp>
#include <mt/WorkSharingBalancedRunnable1D.h>
#include <io/BufferViewStream.h>
#include <io/NullStreams.h>

#include "nitf/J2KTileWriter.hpp"

#undef min
#undef max

namespace
{
    using BufferViewStream = io::BufferViewStream<std::byte> ;

    class CodestreamOp final
    {
        types::RowCol<size_t> getRowColIndices(size_t tileIndex) const noexcept
        {
            return types::RowCol<size_t>(
                tileIndex / mCompressionParams.getNumColsOfTiles(),
                tileIndex % mCompressionParams.getNumColsOfTiles());
        }

        const size_t mStartTile;
        std::shared_ptr<BufferViewStream>* const mTileStreams;
        const std::byte* const mUncompressedImage;
        j2k::CompressionParameters mCompressionParams;

        mutable std::unique_ptr<j2k::details::TileWriter> mWriter;
        std::vector<std::byte> mImageBlock;
        std::byte* mpImageBlock = nullptr;

    public:
        CodestreamOp(
            size_t startTile,
            std::shared_ptr<BufferViewStream>* tileStreams,
            const std::byte* uncompressedImage,
            const j2k::CompressionParameters& compressionParams) :
            mStartTile(startTile),
            mTileStreams(tileStreams),
            mUncompressedImage(uncompressedImage),
            mCompressionParams(compressionParams)
        {
            mImageBlock.resize(mCompressionParams.getTileDims().area());
            mpImageBlock = mImageBlock.data();
        }
        CodestreamOp(const CodestreamOp&) = delete;
        CodestreamOp& operator=(const CodestreamOp&) = delete;
        CodestreamOp(CodestreamOp&&) = default;
        CodestreamOp& operator=(CodestreamOp&&) = delete;

        void operator()(size_t localTileIndex) const
        {
            const auto tileDims = mCompressionParams.getTileDims();
            const auto imageBlock = mpImageBlock;
            const auto globalTileIndex = localTileIndex + mStartTile;
            const auto fullDims = mCompressionParams.getRawImageDims();

            // Need local indices to offset into the uncompressed image properly
            const auto localTileIndices = getRowColIndices(localTileIndex);

            const types::RowCol<size_t> localStart(localTileIndices.row * tileDims.row, localTileIndices.col * tileDims.col);

            const auto uncompressedImage = mUncompressedImage + localStart.row * fullDims.col + localStart.col;

            // Need global indices to determine if we're on the edge of the global image or not
            const auto globalTileIndices = getRowColIndices(globalTileIndex);
            const types::RowCol<size_t> globalStart(globalTileIndices.row * tileDims.row, globalTileIndices.col * tileDims.col);
            const types::RowCol<size_t> globalEnd(std::min(globalStart.row + tileDims.row, fullDims.row), std::min(globalStart.col + tileDims.col, fullDims.col));

            // Block it
            nitf::ImageBlocker::block(uncompressedImage,
                sizeof(std::byte), fullDims.col,
                tileDims.row, tileDims.col,
                globalEnd.row - globalStart.row, globalEnd.col - globalStart.col,
                imageBlock);

            auto tileStream = mTileStreams[localTileIndex];
            if (!mWriter)
            {
                mWriter = std::make_unique<j2k::details::TileWriter>(*tileStream, mCompressionParams);

                // Write out the header
                // OpenJPEG makes us write the header, but we only want to keep it if we're tile 0
                mWriter->start();
                mWriter->flush();

                if (globalTileIndex != 0)
                {
                    tileStream->seek(0, io::Seekable::START);
                }
            }
            else
            {
                mWriter->setOutputStream(*tileStream);
            }

            // Write out the tile
            mWriter->writeTile(imageBlock, globalTileIndex);
            mWriter->flush();
        }

        void finalize(bool keepFooter)
        {
            if (!mWriter)
            {
                return;
            }
            
            // Writer::end() is required to clean up OpenJPEG objects
            // This step also writes out the footer, which we may or may not
            // actually want to keep
            if (keepFooter)
            {
                // tileIdx is guaranteed to be in-bounds if keepFooter is true
                const size_t lastTile = mCompressionParams.getNumTiles() - 1;
                const size_t tileIdx = lastTile - mStartTile;
                mWriter->setOutputStream(*mTileStreams[tileIdx]);

                // Write out the footer
                mWriter->end();
            }
            else
            {
                io::SeekableNullOutputStream outStream;
                mWriter->setOutputStream(outStream);

                // Write out the footer
                mWriter->end();
            }
        }        
    };
}

/*!
 * In rare cases, the "compressed" image will actually be slightly larger
 * than the uncompressed image.  This is a presumably worst case number
 * here - it's probably much larger than it needs to be.
 */
constexpr long double POOR_COMPRESSION_SCALE_FACTOR = 2.0;

j2k::Compressor::Compressor(const CompressionParameters& compressionParams) noexcept :
    mCompressionParams(compressionParams)
{
}

size_t j2k::Compressor::getMaxBytesRequiredToCompress() const noexcept
{
    return getMaxBytesRequiredToCompress(mCompressionParams.getNumTiles());
}

size_t j2k::Compressor::getMaxBytesRequiredToCompress(size_t numTiles) const noexcept
{
    const auto bytesPerTile = mCompressionParams.getTileDims().area();
    const auto maxBytes_ = gsl::narrow_cast<long double>(bytesPerTile * numTiles) * POOR_COMPRESSION_SCALE_FACTOR;
    const auto maxBytes = gsl::narrow_cast<size_t>(std::ceil(maxBytes_));
    return maxBytes;
}

void j2k::Compressor::compress(
    const std::byte* rawImageData,
    size_t numThreads,
    std::vector<std::byte>& compressedData,
    std::vector<size_t>& bytesPerTile) const
{
    compressedData.resize(getMaxBytesRequiredToCompress());
    std::span<std::byte> compressedDataView(compressedData.data(), compressedData.size());

    compressedDataView = compress(rawImageData, numThreads, compressedDataView, bytesPerTile);
    compressedData.resize(compressedDataView.size());
}

std::span<std::byte> j2k::Compressor::compress(const std::byte* rawImageData,
    size_t numThreads,
    std::span<std::byte> compressedData,
    std::vector<size_t>& bytesPerTile) const
{
    return compressTileSubrange(rawImageData,
        types::Range(0, mCompressionParams.getNumTiles()),
        numThreads,
        compressedData,
        bytesPerTile);
}

void j2k::Compressor::compressTile(
    const std::byte* rawImageData,
    size_t tileIndex,
    std::vector<std::byte>& compressedTile) const
{
    compressedTile.resize(getMaxBytesRequiredToCompress(1));
    std::vector<size_t> bytesPerTile;
    std::span<std::byte> compressedView(compressedTile.data(), compressedTile.size());
    compressedView = compressTileSubrange(rawImageData, types::Range(tileIndex, 1), 1,
        compressedView, bytesPerTile);
    compressedTile.resize(compressedView.size());
}

std::span<std::byte> j2k::Compressor::compressTile(
    const std::byte* rawImageData,
    size_t tileIndex,
    std::span<std::byte> compressedTile) const
{
    std::vector<size_t> bytesPerTile;
    return compressTileSubrange(rawImageData, types::Range(tileIndex, 1), 1,
        compressedTile, bytesPerTile);
}

std::span<std::byte> j2k::Compressor::compressRowSubrange(
    const std::byte* rawImageData,
    size_t globalStartRow,
    size_t numLocalRows,
    size_t numThreads,
    std::span<std::byte> compressedData,
    types::Range& tileRange,
    std::vector<size_t>& bytesPerTile) const
{
    // Sanity checks
    const size_t numRowsInTile = mCompressionParams.getTileDims().row;
    if (globalStartRow % numRowsInTile != 0)
    {
        std::ostringstream ostr;
        ostr << "Global start row = " << globalStartRow << " must be a multiple of number of rows in tile = " << numRowsInTile;
        throw except::Exception(Ctxt(ostr.str()));
    }

    if ((numLocalRows % numRowsInTile != 0) &&
        (globalStartRow + numLocalRows != mCompressionParams.getRawImageDims().row))
    {
        std::ostringstream ostr;
        ostr << "Number of local rows = " << numLocalRows << " must be a multiple of number of rows in tile = " << numRowsInTile;
        throw except::Exception(Ctxt(ostr.str()));
    }

    const auto startRowOfTiles = globalStartRow / numRowsInTile;
    const auto numRowsOfTiles = math::ceilingDivide(numLocalRows, numRowsInTile);

    const auto numColsOfTiles = mCompressionParams.getNumColsOfTiles();
    tileRange.mStartElement = startRowOfTiles * numColsOfTiles;
    tileRange.mNumElements = numRowsOfTiles * numColsOfTiles;

    return compressTileSubrange(rawImageData,
        tileRange,
        numThreads,
        compressedData,
        bytesPerTile);
}

std::span<std::byte> j2k::Compressor::compressTileSubrange(
    const std::byte* rawImageData,
    const types::Range& tileRange,
    size_t numThreads,
    std::span<std::byte> compressedData,
    std::vector<size_t>& bytesPerTile) const
{
    // We write initially directly into 'compressedData', reserving the max
    // expected # of bytes/tile
    const auto numTiles = tileRange.mNumElements;
    const auto maxNumBytesPerTile = getMaxBytesRequiredToCompress(1);
    const auto numBytesNeeded = maxNumBytesPerTile * numTiles;
    if (compressedData.size() < numBytesNeeded)
    {
        std::ostringstream ostr;
        ostr << "Require " << numBytesNeeded << " bytes for compression of "
            << numTiles << " tiles but only received " << compressedData.size() << " bytes";
        throw except::Exception(Ctxt(ostr.str()));
    }

    auto compressedPtr = compressedData.data();
    std::vector<std::shared_ptr<BufferViewStream>> tileStreams(numTiles);
    for (size_t tile = 0; tile < numTiles; ++tile, compressedPtr += maxNumBytesPerTile)
    {
        auto bufferStream = std::make_shared<BufferViewStream>(mem::BufferView<std::byte>(compressedPtr, maxNumBytesPerTile));
        tileStreams[tile] = bufferStream;
    }

    std::vector<CodestreamOp> ops;
    ops.reserve(numThreads);
    for (size_t ii = 0; ii < numThreads; ++ii)
    {
        ops.emplace_back(tileRange.mStartElement, tileStreams.data(), rawImageData, mCompressionParams);
    }

    // Compress the image
    mt::runWorkSharingBalanced1D(numTiles, numThreads, ops);

    // End compression for each thread
    // If the last tile is in 'tileRange', we need to ensure that we write the
    // footer exactly once
    bool keepFooter = (tileRange.endElement() == mCompressionParams.getNumTiles());
    for (auto& op : ops)
    {
        op.finalize(keepFooter);
        keepFooter = false;
    }

    // At this point the tiles are all compressed into 'compressedData' but
    // they're not contiguous.  Shift memory around to get a contiguous buffer.
    size_t numBytesWritten = 0;
    bytesPerTile.resize(numTiles);

    auto dest = compressedData.data();
    for (size_t tileNum = 0; tileNum < numTiles; ++tileNum)
    {
        BufferViewStream& tileStream = *tileStreams[tileNum];

        // This shouldn't be possible because if a tile was too big, it would
        // have thrown when compressing
        const auto numBytesThisTile = gsl::narrow<size_t>(tileStream.tell());
        if (numBytesWritten + numBytesThisTile > compressedData.size())
        {
            std::ostringstream os;
            os << "Cannot write " << numBytesThisTile << " bytes for tile " << tileNum << " at byte offset " << numBytesWritten
                << " - exceeds maximum compressed image size (" << compressedData.size() << " bytes)!";
            throw except::Exception(Ctxt(os.str()));
        }

        const auto src = tileStream.get();

        // Copy the tile to the output buffer
        // Since we're reusing the same buffer for the contiguous output,
        // memory addresses may overlap so we need to use memmove()
        if (src != dest)
        {
            ::memmove(dest, src, numBytesThisTile);
        }

        numBytesWritten += numBytesThisTile;
        bytesPerTile[tileNum] = numBytesThisTile;
        dest += numBytesThisTile;
    }

    return std::span<std::byte>(compressedData.data(), numBytesWritten);
}


