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
    subheader.setPixelInformation("pixelType",
            8,
            10,
            "justification",
            "irep",
            "icat",
            bandInfos);

    // TODO: Do strings have length limits to worry about?
    TEST_ASSERT_EQ(subheader.getPixelValueType(), "pixelType");
    TEST_ASSERT_EQ(subheader.getPixelJustification(), "justification");
    TEST_ASSERT_EQ(subheader.getNumBitsPerPixel(), 8);
    TEST_ASSERT_EQ(subheader.getActualBitsPerPixel(), 10);
    TEST_ASSERT_EQ(subheader.getImageRepresentation(), "irep");
    TEST_ASSERT_EQ(subheader.getImageCategory(), "icat");
    TEST_ASSERT_EQ(subheader.getBandInfo(1).getRepresentation().toString(),
            "Rp");
    TEST_ASSERT_EQ(subheader.getBandInfo(0).getRepresentation().toString(),
            "CP");

    TEST_ASSERT_EQ(subheader.getNumImageBands(), 2);
    TEST_ASSERT_EQ(subheader.getNumMultispectralImageBands(), 0);

    nitf::CppImageSubheader xSubheader;
    bandInfos.resize(10);
    xSubheader.setPixelInformation("pixType", 8, 10, "just", "irep", "icat",
            bandInfos);
    TEST_ASSERT_EQ(xSubheader.getNumImageBands(), 0);
    TEST_ASSERT_EQ(xSubheader.getNumMultispectralImageBands(), 10);



}

int main(int /*argc*/, char** /*argv*/)
{
    try
    {
        TEST_CHECK(testSetPixelInformation);
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

