/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2004 - 2020, MDA Information Systems LLC
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

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

#include <import/nitf.hpp>
#include <nitf/ImageSubheader.hpp>
#include <nitf/ImageWriter.hpp>
#include <nitf/Record.hpp>

#include "TestCase.h"

static void doChangeFileHeader(const std::string& inputPathname, const std::string& outputPathname)
{
    if (nitf::Reader::getNITFVersion(inputPathname) == NITF_VER_UNKNOWN)
    {
        throw std::invalid_argument("Invalid NITF: " + inputPathname);
    }

    nitf::Reader reader;
    nitf::IOHandle io(inputPathname);
    
    nitf::Record record = reader.read(io);
    nitf::FileHeader fileHeader = record.getHeader();

    auto fileTitle = fileHeader.getFileTitle();
    auto strFileTitle = fileTitle.toString();
    str::replaceAll(strFileTitle, " ", "*"); // field is fixed length
    fileTitle.set(strFileTitle);

    record.setHeader(fileHeader);

    nitf::Writer writer;
    nitf::IOHandle output(outputPathname, NITF_ACCESS_WRITEONLY, NITF_CREATE);
    writer.prepare(output, record);
    writer.setWriteHandlers(io, record);
    writer.write();
}

namespace
{
TEST_CASE(imageWriterThrowsOnFailedConstruction)
{
    nitf::ImageSubheader subheader;
    TEST_EXCEPTION(nitf::ImageWriter(subheader));
}

TEST_CASE(constructValidImageWriter)
{
    nitf::Record record;
    nitf::ImageSegment segment = record.newImageSegment();
    nitf::ImageSubheader subheader = segment.getSubheader();
    std::vector<nitf::BandInfo> bands = {nitf::BandInfo(), nitf::BandInfo()};
    subheader.setPixelInformation("INT", 8, 8, "R", "MONO", "VIS", bands);
    subheader.setBlocking(100, 200, 10, 10, "P");
    nitf::ImageWriter writer(subheader);
}

TEST_CASE(changeFileHeader)
{
    std::string inputPathname;
    std::string outputPathname;
    if (sys::OS().getEnvIfSet("NITF_UNIT_TEST_inputPathname_", inputPathname))
    {
        // If one is set, they both must be set
        TEST_ASSERT_TRUE(sys::OS().getEnvIfSet("NITF_UNIT_TEST_outputPathname_", outputPathname));
    }
    else
    {
        // need env. vars. set
        std::clog << "NITF_UNIT_TEST_inputPathname_ not set, assuming success.\n";
        TEST_ASSERT_TRUE(true);
        return;
    }

    doChangeFileHeader(inputPathname, outputPathname);

    nitf::Reader reader;
    nitf::IOHandle io(outputPathname);
    nitf::Record record = reader.read(io);
    nitf::FileHeader fileHeader = record.getHeader();

    const auto fileTitle = fileHeader.getFileTitle().toString();
    auto npos = fileTitle.find(" ");
    TEST_ASSERT_EQ(npos, std::string::npos);
    npos = fileTitle.find("*");
    TEST_ASSERT(npos != std::string::npos);
}
}

TEST_MAIN(
    TEST_CHECK(imageWriterThrowsOnFailedConstruction);
    TEST_CHECK(constructValidImageWriter);
    TEST_CHECK(changeFileHeader);
    )
