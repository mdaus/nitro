/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2004 - 2017, MDA Information Systems LLC
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
#include <except/Exception.h>
#include <nitf/BandInfo.hpp> // TODO: CppBandInfo.hpp
#include <nitf/CppImageSubheader.hpp>
#include "TestCase.h"

namespace
{
TEST_CASE(testSetPixelInformation)
{
    std::vector<nitf::BandInfo> bandInfos;
    nitf::LookupTable lut(3, 2);
    nitf::BandInfo firstInfo;
    nitf::BandInfo secondInfo;

    firstInfo.init("CP", "nth", "D", "ls");
    secondInfo.init("Rp", "djf", "C", "sz");

    bandInfos.push_back(firstInfo);
    bandInfos.push_back(secondInfo);

    nitf::CppImageSubheader subheader;
    subheader.setPixelInformation("INT",
            8,
            10,
            "R",
            "MONO",
            "SL",
            bandInfos);

    TEST_ASSERT_EQ(subheader.getPixelValueType(), "INT");
    TEST_ASSERT_EQ(subheader.getPixelJustification(), "R");
    TEST_ASSERT_EQ(subheader.getNumBitsPerPixel(), 8);
    TEST_ASSERT_EQ(subheader.getActualBitsPerPixel(), 10);
    TEST_ASSERT_EQ(subheader.getImageRepresentation(), "MONO");
    TEST_ASSERT_EQ(subheader.getImageCategory(), "SL");
    TEST_ASSERT_EQ(subheader.getBandInfo(1).getRepresentation().toString(),
            "Rp");
    TEST_ASSERT_EQ(subheader.getBandInfo(0).getRepresentation().toString(),
            "CP");

    TEST_ASSERT_EQ(subheader.getNumImageBands(), 2);
    TEST_ASSERT_EQ(subheader.getNumMultispectralImageBands(), 0);

    nitf::CppImageSubheader xSubheader;
    bandInfos.resize(10);
    xSubheader.setPixelInformation("R", 8, 10, "L", "RGB/LUT", "TI",
            bandInfos);
    TEST_ASSERT_EQ(xSubheader.getNumImageBands(), 0);
    TEST_ASSERT_EQ(xSubheader.getNumMultispectralImageBands(), 10);
}

TEST_CASE(testSetCorners)
{
    nitf::CppImageSubheader subheader;
    types::LatLonCorners corners;
    corners.getCorner(0).setLat(50);
    corners.getCorner(0).setLon(50);
    corners.getCorner(1).setLat(50);
    corners.getCorner(1).setLon(50.87);
    corners.getCorner(2).setLat(62);
    corners.getCorner(2).setLon(50.87);
    corners.getCorner(3).setLat(62);
    corners.getCorner(3).setLon(50);

    subheader.setCornersFromLatLons(NITF_CORNERS_DECIMAL, corners);
    TEST_ASSERT_EQ(subheader.getCornerCoordinates(),
            "+50.000+050.000"
            "+50.000+050.870"
            "+62.000+050.870"
            "+62.000+050.000");
    TEST_ASSERT_EQ(subheader.getImageCoordinateSystem(), "D");
    TEST_ASSERT_EQ(subheader.getCornersType(), NITF_CORNERS_DECIMAL);
    subheader.setCornersFromLatLons(NITF_CORNERS_GEO, corners);
    TEST_ASSERT_EQ(subheader.getCornerCoordinates(),
            "500000N0500000E"
            "500000N0505212E"
            "620000N0505212E"
            "620000N0500000E");
    TEST_ASSERT_EQ(subheader.getImageCoordinateSystem(), "G");
    TEST_ASSERT_EQ(subheader.getCornersType(), NITF_CORNERS_GEO);
    TEST_THROWS(subheader.setCornersFromLatLons(NITF_CORNERS_UTM, corners));
    types::LatLonCorners roundTrippedCorners = subheader.getCornersAsLatLons();
    TEST_ASSERT((roundTrippedCorners == corners));
}

TEST_CASE(testSetBlocking)
{
    nitf::CppImageSubheader subheader;
    subheader.setBlocking(3, 4, 1, 2, "B");
    TEST_ASSERT_EQ(subheader.getNumRows(), 3);
    TEST_ASSERT_EQ(subheader.getNumCols(), 4);
    TEST_ASSERT_EQ(subheader.getNumPixelsPerHorizBlock(), 2);
    TEST_ASSERT_EQ(subheader.getNumPixelsPerVertBlock(), 1);
    TEST_ASSERT_EQ(subheader.getImageMode(), "B");
    TEST_ASSERT_EQ(subheader.getNumBlocksPerRow(), 2);
    TEST_ASSERT_EQ(subheader.getNumBlocksPerCol(), 3);

    subheader.setBlocking(50, 100, 9000, 9000, "B");
    TEST_ASSERT_EQ(subheader.getNumRows(), 50);
    TEST_ASSERT_EQ(subheader.getNumCols(), 100);
    TEST_ASSERT_EQ(subheader.getNumPixelsPerHorizBlock(), 0);
    TEST_ASSERT_EQ(subheader.getNumPixelsPerVertBlock(), 0);
    TEST_ASSERT_EQ(subheader.getImageMode(), "B");
    TEST_ASSERT_EQ(subheader.getNumBlocksPerRow(), 1);
    TEST_ASSERT_EQ(subheader.getNumBlocksPerCol(), 1);

    subheader.setDimensions(75, 200);
    TEST_ASSERT_EQ(subheader.getImageMode(), "B");
}

TEST_CASE(testCreateBands)
{
    nitf::CppImageSubheader subheader;
    subheader.createBands(5);
    TEST_ASSERT_EQ(subheader.getBandCount(), 5);
    subheader.createBands(3);
    TEST_ASSERT_EQ(subheader.getBandCount(), 8);
}

TEST_CASE(testImageComments)
{
    nitf::CppImageSubheader subheader;
    subheader.insertImageComment("First comment", 0);
    subheader.insertImageComment("Second comment", 0);
    subheader.insertImageComment("Third comment", 5);
    const std::vector<nitf::BCSA<std::string, 80> >& comments =
            subheader.getImageComments();
    TEST_ASSERT_EQ(comments.size(), 3);
    TEST_ASSERT_EQ(comments.size(), subheader.getNumImageComments());
    TEST_ASSERT(std::string(comments[0]) == "Second comment");
    TEST_ASSERT(std::string(comments[1]) == "First comment");
    TEST_ASSERT(std::string(comments[2]) == "Third comment");

    for (size_t ii = 0; ii < 6; ++ii)
    {
        subheader.insertImageComment("comment", ii + 3);
    }
    TEST_THROWS(subheader.insertImageComment("Wrong", 0));
    subheader.removeImageComment(1);
    TEST_ASSERT(std::string(comments[1]) == "Third comment");
    TEST_ASSERT(subheader.getNumImageComments() == 8);
    TEST_THROWS(subheader.removeImageComment(8));
}

TEST_CASE(testDateAndTime)
{
    nitf::CppImageSubheader subheader;
    nitf::DateTime datetime(2016, 5, 13);
    subheader.setImageDateAndTime(datetime);
    TEST_ASSERT(subheader.getImageDateAndTime() == datetime);
}

TEST_CASE(testCheckSetters)
{
    nitf::CppImageSubheader subheader;
    subheader.setClassificationReason("D");
    TEST_ASSERT_EQ(subheader.getClassificationReason(), "D");
    TEST_THROWS(subheader.setClassificationReason("a"));
    TEST_THROWS(subheader.setClassificationReason("H"));

    subheader.setImageLocation("99999-0000");
    TEST_THROWS(subheader.setImageLocation("99999-00000"));
    TEST_THROWS(subheader.setImageLocation("99999-000"));
    TEST_THROWS(subheader.setImageLocation("9a99999999"));
    TEST_THROWS(subheader.setImageLocation("-9b9900000"));

    subheader.setImageLocation(-4, -10);
    TEST_ASSERT_EQ(subheader.getImageLocation(), "-0004-0010");

    subheader.setImageMagnification("1.0");
    TEST_ASSERT_EQ(subheader.getImageMagnification(), "1.0");
    subheader.setImageMagnification("/1");
    TEST_THROWS(subheader.setImageMagnification("2"));
    TEST_THROWS(subheader.setImageMagnification("/9999"));
}
}

int main(int /*argc*/, char** /*argv*/)
{
    try
    {
        TEST_CHECK(testSetPixelInformation);
        TEST_CHECK(testSetCorners);
        TEST_CHECK(testSetBlocking);
        TEST_CHECK(testCreateBands);
        TEST_CHECK(testImageComments);
        TEST_CHECK(testDateAndTime);
        TEST_CHECK(testCheckSetters);
        return 0;
    }
    catch(const except::Exception& ex)
    {
        std::cerr << ex.toString() << std::endl;
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
    }
    catch(...)
    {
        std::cerr << "Unknown exception\n";
    }
    return 1;
}

