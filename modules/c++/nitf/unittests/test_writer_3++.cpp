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

#include <sys/Filesystem.h>

#include <import/nitf.hpp>
#include <gsl/gsl.h>

#include "TestCase.h"

namespace fs = std::filesystem;

/*
 * This test tests the round-trip process of taking an input NITF
 * file and writing it to a new file. This includes writing the image
 * segments (headers, extensions, and image data). This is an example
 * of how users can write the image data to their NITF file
 */

static std::string argv0;
static fs::path findInputFile()
{
    const fs::path inputFile = fs::path("modules") / "c++" / "nitf" / "unittests" / "sicd_50x50.nitf";
 
    fs::path root;
    if (argv0.empty())
    {
        // running in Visual Studio
        root = fs::current_path().parent_path().parent_path();
    }
    else
    {
        root = fs::absolute(argv0).parent_path().parent_path().parent_path().parent_path();
        root = root.parent_path().parent_path();
    }

    return root / inputFile;
}

static nitf::Record doRead(const std::string& inFile, nitf::Reader& reader);

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

static void doWrite(nitf::Record record, nitf::Reader& reader, const std::string& inRootFile, const std::string& outFile)
{
    nitf::Writer writer;
    nitf::IOHandle output(outFile, NITF_ACCESS_WRITEONLY, NITF_CREATE);
    writer.prepare(output, record);

    int numImages = record.getHeader().getNumImages();
    nitf::ListIterator end = record.getImages().end();
    nitf::ListIterator iter = record.getImages().begin();

    for (int i = 0; i < numImages && iter != end; ++i, ++iter)
    {
        nitf::ImageSegment imseg;
        imseg = *iter;
        int nbands = imseg.getSubheader().numImageBands();
        nitf::ImageWriter iWriter = writer.newImageWriter(i);
        nitf::ImageSource iSource = setupBands(nbands, i, inRootFile);
        iWriter.attachSource(iSource);
    }

    const auto num = gsl::narrow<int>(record.getNumDataExtensions());
    for (int i = 0; i < num; i++)
    {
        nitf::SegmentReaderSource readerSource(reader.newDEReader(i));
        std::shared_ptr< ::nitf::WriteHandler> segmentWriter(
            new nitf::SegmentWriter(readerSource));
        writer.setDEWriteHandler(i, segmentWriter);
    }

    writer.write();
    output.close();
}

static std::string testName;

TEST_CASE(test_writer_3)
{
    ::testName = testName;

    std::string input_file = findInputFile().string();
    const std::string output_file = "test_writer_3++.nitf";

    // Check that wew have a valid NITF
    const auto version = nitf::Reader::getNITFVersion(input_file);
    TEST_ASSERT(version != NITF_VER_UNKNOWN);

    nitf::Reader reader;
    nitf::Record record = doRead(input_file, reader);
    doWrite(record, reader, input_file, output_file);
}

static void manuallyWriteImageBands(nitf::ImageSegment & segment,
                             const std::string& imageName,
                             nitf::ImageReader& deserializer,
                             int imageNumber)
{
    int padded;

    nitf::ImageSubheader subheader = segment.getSubheader();

    uint32_t nBits = subheader.getNumBitsPerPixel();
    uint32_t nBands = subheader.getNumImageBands();
    uint32_t xBands = subheader.getNumMultispectralImageBands();
    nBands += xBands;

    const auto nRows = subheader.numRows();
    const auto nColumns = subheader.numCols();

    //one row at a time
    const auto subWindowSize = gsl::narrow<size_t>(nColumns * NITF_NBPP_TO_BYTES(nBits));

    TEST_ASSERT_EQ(2, nBands);
    TEST_ASSERT_EQ(0, xBands);
    TEST_ASSERT_EQ(50, nRows);
    TEST_ASSERT_EQ(50, nColumns);
    TEST_ASSERT_EQ("R  ", subheader.getPixelValueType().toString());
    TEST_ASSERT_EQ(32, subheader.numBitsPerPixel());
    TEST_ASSERT_EQ("32", subheader.getActualBitsPerPixel().toString());
    TEST_ASSERT_EQ("R", subheader.getPixelJustification().toString());
    TEST_ASSERT_EQ("P", subheader.getImageMode().toString());
    TEST_ASSERT_EQ(1, subheader.numBlocksPerRow());
    TEST_ASSERT_EQ(1, subheader.numBlocksPerCol());
    TEST_ASSERT_EQ(50, subheader.numPixelsPerHorizBlock());
    TEST_ASSERT_EQ(50, subheader.numPixelsPerVertBlock());
    TEST_ASSERT_EQ("NC", subheader.imageCompression());
    TEST_ASSERT_EQ("    ", subheader.getCompressionRate().toString());

    std::vector<uint8_t*> buffer(nBands);
    std::vector<std::unique_ptr<uint8_t[]>> buffer_(nBands);
    std::vector<uint32_t> bandList(nBands);

    for (uint32_t band = 0; band < nBands; band++)
        bandList[band] = band;

    nitf::SubWindow subWindow;
    subWindow.setStartCol(0);
    subWindow.setNumRows(1);
    subWindow.setNumCols(gsl::narrow<uint32_t>(nColumns));

    // necessary ?
    std::unique_ptr<nitf::DownSampler> pixelSkip(new nitf::PixelSkip(1, 1));
    subWindow.setDownSampler(pixelSkip.get());
    subWindow.setBandList(bandList.data());
    subWindow.setNumBands(nBands);

    for (uint32_t i = 0; i < nBands; i++)
    {
        buffer_[i].reset(new uint8_t[subWindowSize]);
        buffer[i] = buffer_[i].get();
    }

    std::vector<nitf::IOHandle> handles;
    //make the files
    for (int i = 0; i < gsl::narrow<int>(nBands); i++)
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

static nitf::Record doRead(const std::string& inFile, nitf::Reader& reader)
{
    nitf::IOHandle io(inFile);
    nitf::Record record = reader.read(io);

    /*  Set this to the end, so we'll know when we're done!  */
    nitf::ListIterator iter = record.getImages().begin();

    const uint32_t num = record.getNumImages();
    for (int i = 0; i < gsl::narrow<int>(num); i++)
    {
        nitf::ImageSegment imageSegment = *iter;
        iter++;
        nitf::ImageReader deserializer = reader.newImageReader(i);

        /*  Write the thing out  */
        manuallyWriteImageBands(imageSegment, inFile, deserializer, i);
    }

    return record;
}

TEST_MAIN(
    (void)argc;
    argv0 = argv[0];
    TEST_CHECK(test_writer_3);
)