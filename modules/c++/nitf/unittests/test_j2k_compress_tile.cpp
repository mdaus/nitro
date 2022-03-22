/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2004 - 2017, MDA Information Systems LLC
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
 * License along with this program;
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <cstring>
#include <iostream>
#include <numeric>
#include <std/cstddef>

#include <sio/lite/FileReader.h>
#include <sio/lite/ReadUtils.h>
#include <types/Range.h>
#include <io/ReadUtils.h>
#include <io/TempFile.h>
#include <io/FileOutputStream.h>
#include <sys/OS.h>
#include <gsl/gsl.h>

#include <import/nrt.h>
#include <nitf/ImageBlocker.hpp>
#include <nitf/J2KCompressor.hpp>

#include "TestCase.h"

struct Image final
{
    types::RowCol<size_t> dims;
    std::vector<std::byte> pixels;
};

static void generateTestImage(Image& image)
{
    image.dims.row = 2048;
    image.dims.col = 1024;
    image.pixels.resize(image.dims.area());

    for (size_t ii = 0; ii < image.pixels.size(); ++ii)
    {
        // Let's write this out as some sort of pattern so
        // J2K compression has something to work with
        image.pixels[ii] = static_cast<std::byte>((ii / 100) % 128);
    }
}

static void compressEntireImage(const j2k::Compressor& compressor,
    const Image& inputImage,
    std::vector<std::byte>& outputImage,
    size_t numThreads)
{
    std::vector<size_t> bytesPerBlock;
    const std::span<const std::byte> pixels(inputImage.pixels.data(), inputImage.pixels.size());
    compressor.compress(pixels, outputImage, bytesPerBlock);
}

static void compressTileSubrange(const j2k::Compressor& compressor,
    const j2k::CompressionParameters& params,
    const Image& inputImage,
    size_t numSubsets,
    std::vector<std::byte>& outputImage)
{
    const auto tileDims = params.getTileDims();
    const types::RowCol<size_t> numTiles(params.getNumRowsOfTiles(), params.getNumColsOfTiles());

    const auto rowsOfTilesPerSubset = numTiles.row / numSubsets;

    std::vector<std::vector<std::byte>> compressedImages(numSubsets);
    std::vector<size_t> numBytesCompressed(numSubsets);

    for (size_t subset = 0; subset < numSubsets; ++subset)
    {
        const auto tileRow = rowsOfTilesPerSubset * subset;
        const auto numRowsOfTiles = (subset == numSubsets - 1) ? numTiles.row - tileRow : rowsOfTilesPerSubset;
        const auto subsetNumTiles = numRowsOfTiles * numTiles.col;

        auto& compressedImage = compressedImages[subset];
        compressedImage.resize(compressor.getMaxBytesRequiredToCompress(subsetNumTiles));

        const std::span<std::byte> compressedTiles(compressedImage.data(), compressedImage.size());
        std::vector<size_t> compressedBytesPerBlock;

        const auto offset = (tileRow * tileDims.row) * inputImage.dims.col;
        const std::span<const std::byte> uncompressed(inputImage.pixels.data() + offset, inputImage.pixels.size() - offset);

        compressor.compressTileSubrange(
            uncompressed,
            types::Range(tileRow * numTiles.col, subsetNumTiles),
            compressedTiles,
            compressedBytesPerBlock);
        numBytesCompressed[subset] = compressedTiles.size();
    }

    const auto numTotalBytesCompressed = std::accumulate(numBytesCompressed.begin(), numBytesCompressed.end(), static_cast<size_t>(0));

    outputImage.resize(numTotalBytesCompressed);
    auto outputPtr = outputImage.data();
    for (size_t ii = 0; ii < numSubsets; ++ii)
    {
        ::memcpy(outputPtr, compressedImages[ii].data(), numBytesCompressed[ii]);
        outputPtr += numBytesCompressed[ii];
    }
}

TEST_CASE(unittest_compress_tile)
{
    /* placeholder */
}

TEST_MAIN(
    (void)argc;
    (void)argv;
    TEST_CHECK(unittest_compress_tile);
    )

