/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
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

#include <vector>
#include <iostream>
#include <string>
#include <std/filesystem>

#include <gsl/gsl.h>

#include <import/nitf.hpp>
#include <nitf/UnitTests.hpp>

#include "TestCase.h"

const std::string output_file = "test_writer_3++.nitf";

using path = std::filesystem::path;

static path findInputFile()
{
    static const auto unittests = path("modules") / "c++" / "nitf" / "unittests";
    static const auto inputPath = nitf::Test::findInputFile(unittests, "sicd_50x50.nitf");
    return inputPath;
}

static std::string makeBandName(const std::string& rootFile, int imageNum, int bandNum)
{
    std::string::size_type pos = rootFile.find_last_of("/\\");
    std::ostringstream os;
    os << rootFile.substr(pos + 1) << "__" << imageNum << "_band_" << bandNum;
    std::string newFile = os.str();

    while ((pos = newFile.find(".")) != std::string::npos)
        newFile.replace(pos, 1, "_");
    newFile += ".man";
    return newFile;
}

static nitf::ImageSource setupBands(int nbands, int imageNum, const std::string& inRootFile)
{
    nitf::ImageSource iSource;
    for (int i = 0; i < nbands; i++)
    {
        std::string inFile = makeBandName(inRootFile, imageNum, i);
        nitf::FileSource fs(inFile, 0, 1, 0);
        iSource.addBand(fs);
    }
    return iSource;
}

static void doWrite(const nitf::Record& record_, nitf::Reader& reader, const std::string& inRootFile,
    nitf::Writer& writer)
{
    auto& record = const_cast<nitf::Record&>(record_); // TODO: remove when API is const-correct

    int numImages = record.getHeader().getNumImages();
    nitf::ListIterator end = record.getImages().end();
    nitf::ListIterator iter = record.getImages().begin();

    for (int i = 0; i < numImages && iter != end; ++i, ++iter)
    {
        nitf::ImageSegment imseg;
        imseg = *iter;
        const auto nbands = static_cast<int>(imseg.getSubheader().numImageBands());
        nitf::ImageWriter iWriter = writer.newImageWriter(i);
        nitf::ImageSource iSource = setupBands(nbands, i, inRootFile);
        iWriter.attachSource(iSource);
    }

    const auto num = static_cast<int>(record.getNumDataExtensions());
    for (int i = 0; i < num; i++)
    {
        nitf::SegmentReaderSource readerSource(reader.newDEReader(i));
        mem::SharedPtr< ::nitf::WriteHandler> segmentWriter(
            new nitf::SegmentWriter(readerSource));
        writer.setDEWriteHandler(i, segmentWriter);
    }

    writer.write();
}

static void manuallyWriteImageBands(const std::string& testName,
    nitf::ImageSegment& segment, const std::string& imageName, nitf::ImageReader& deserializer, int imageNumber)
{
    int padded;

    nitf::ImageSubheader subheader = segment.getSubheader();

    uint32_t nBits = subheader.getNumBitsPerPixel();
    uint32_t nBands = subheader.getNumImageBands();
    uint32_t xBands = subheader.getNumMultispectralImageBands();
    nBands += xBands;

    const auto nRows = subheader.numRows();
    const auto nColumns = gsl::narrow<uint32_t>(subheader.numCols());

    //one row at a time
    const auto subWindowSize = static_cast<size_t>(nColumns * NITF_NBPP_TO_BYTES(nBits));

    TEST_ASSERT_EQ(2, static_cast<int>(nBands));
    TEST_ASSERT_EQ(0, static_cast<int>(xBands));
    TEST_ASSERT_EQ(static_cast<size_t>(50), nRows);
    TEST_ASSERT_EQ(static_cast<uint32_t>(50), nColumns);
    TEST_ASSERT_EQ(nitf::PixelValueType::Floating , subheader.pixelValueType());
    TEST_ASSERT_EQ(static_cast<size_t>(32), subheader.numBitsPerPixel());
    TEST_ASSERT_EQ_STR("32", subheader.getActualBitsPerPixel().toString());
    TEST_ASSERT_EQ_STR("R", subheader.getPixelJustification().toString());
    TEST_ASSERT_EQ(nitf::BlockingMode::Pixel, subheader.imageBlockingMode());
    TEST_ASSERT_EQ(static_cast<size_t>(1), subheader.numBlocksPerRow());
    TEST_ASSERT_EQ(static_cast<size_t>(1), subheader.numBlocksPerCol());
    TEST_ASSERT_EQ(static_cast<size_t>(50), subheader.numPixelsPerHorizBlock());
    TEST_ASSERT_EQ(static_cast<size_t>(50), subheader.numPixelsPerVertBlock());
    TEST_ASSERT_EQ(nitf::ImageCompression::NC, subheader.imageCompression());
    TEST_ASSERT_EQ_STR("    ", subheader.getCompressionRate().toString());

    nitf::BufferList<std::byte> buffer(nBands);
    std::vector<uint32_t> bandList(nBands);

    for (uint32_t band = 0; band < nBands; band++)
        bandList[band] = band;

    nitf::SubWindow subWindow(1, nColumns);

    // necessary ?
    nitf::DownSampler* pixelSkip = new nitf::PixelSkip(1, 1);
    subWindow.setDownSampler(pixelSkip);
    setBands(subWindow, bandList);

    buffer.initialize(subWindowSize);

    std::vector<nitf::IOHandle> handles;
    //make the files
    for (int i = 0; i < static_cast<int>(nBands); i++)
    {
        std::string name = makeBandName(imageName, imageNumber, i);
        nitf::IOHandle toFile(name, NITF_ACCESS_WRITEONLY, NITF_CREATE);
        handles.push_back(toFile);
    }

    //read all row blocks and write to disk
    for (uint32_t i = 0; i < nRows; ++i)
    {
        subWindow.setStartRow(i);
        deserializer.read(subWindow, buffer.data(), &padded);
        for (uint32_t j = 0; j < nBands; j++)
        {
            handles[j].write(buffer[j], subWindowSize);
        }
    }

    //close output handles
    for (uint32_t i = 0; i < nBands; i++)
        handles[i].close();
}

static nitf::Record doRead(const std::string& testName,
    const std::string& inFile, nitf::Reader& reader)
{
    // Check that wew have a valid NITF
    const auto version = nitf::Reader::getNITFVersion(inFile);
    TEST_ASSERT(version != NITF_VER_UNKNOWN);

    nitf::IOHandle io(inFile);
    nitf::Record record = reader.read(io);

    /*  Set this to the end, so we'll know when we're done!  */
    nitf::ListIterator end = record.getImages().end();
    nitf::ListIterator iter = record.getImages().begin();
    for (int count = 0, numImages = record.getHeader().getNumImages();
        count < numImages && iter != end; ++count, ++iter)
    {
        nitf::ImageSegment imageSegment = *iter;
        nitf::ImageReader deserializer = reader.newImageReader(count);

        /*  Write the thing out  */
        manuallyWriteImageBands(testName, imageSegment, inFile, deserializer, count);
    }

    return record;
}

/*
    * This test tests the round-trip process of taking an input NITF
    * file and writing it to a new file. This includes writing the image
    * segments (headers, extensions, and image data). This is an example
    * of how users can write the image data to their NITF file
    */
static void test_writer_3__doWrite(nitf::Record record, nitf::Reader& reader, const std::string& inRootFile, const std::string& outFile)
{
    nitf::Writer writer;
    nitf::IOHandle output(outFile, NITF_ACCESS_WRITEONLY, NITF_CREATE);
    writer.prepare(output, record);

    doWrite(record, reader, inRootFile, writer);

    output.close();
}
TEST_CASE(test_writer_3_)
{
    const auto input_file = findInputFile().string();

    nitf::Reader reader;
    nitf::Record record = doRead(testName, input_file, reader);
    test_writer_3__doWrite(record, reader, input_file, output_file);
}

/*
    * This test tests the round-trip process of taking an input NITF
    * file and writing it to a new file. This includes writing the image
    * segments (headers, extensions, and image data).
    *
    * This example differs from test_writer_3 in that it tests the
    * BufferedWriter classes, and writes the entire file as a set of
    * configurable sized blocks.  The last block may be smaller than the others
    * if the data does not fill the block.
    *
    */
static void test_buffered_write__doWrite(const std::string& testName,
    nitf::Record record, nitf::Reader& reader,
    const std::string& inRootFile,
    const std::string& outFile,
    size_t bufferSize)
{
    nitf::BufferedWriter output(outFile, bufferSize);
    nitf::Writer writer;
    writer.prepareIO(output, record);

    doWrite(record, reader, inRootFile, writer);

    const auto blocksWritten = output.getNumBlocksWritten();
    const auto partialBlocksWritten = output.getNumPartialBlocksWritten();
    output.close();
    TEST_ASSERT_EQ(static_cast<uint64_t>(60), blocksWritten);
    TEST_ASSERT_EQ(static_cast<uint64_t>(53), partialBlocksWritten);
}
TEST_CASE(test_buffered_write_)
{
    const auto input_file = findInputFile().string();

    size_t blockSize = 8192;

    nitf::Reader reader;
    nitf::Record record = doRead(testName, input_file, reader);
    test_buffered_write__doWrite(testName,
        record, reader, input_file, output_file, blockSize);

}

TEST_MAIN(
    TEST_CHECK(test_writer_3_);
    TEST_CHECK(test_buffered_write_);
    )
