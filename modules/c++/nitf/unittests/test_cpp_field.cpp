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


#include <nitf/CppField.hpp>
#include <nitf/Field.hpp>
#include <nitf/Field.h>
#include <nitf/DateTime.hpp>
#include "TestCase.h"

TEST_CASE(testSettingBSCAField)
{
    nitf::AlphaNumericField<2> field;
    TEST_ASSERT_EQ(field.getType(), NITF_BCS_A);
    TEST_THROWS(field = std::string("foo"));
    field = std::string("ab");
    TEST_ASSERT_EQ(field.toString(), "ab");

    nitf::AlphaNumericField<5> largeField;
    largeField = "c";
    TEST_ASSERT_EQ(largeField.toString().size(), 5);
    TEST_ASSERT_EQ(largeField.toString(), "c    ");

    largeField = "";
    TEST_ASSERT_EQ(largeField.toString(), "     ");
}

TEST_CASE(testSettingBSCAFieldWithNumericData)
{
    nitf::AlphaNumericField<1> smallField;
    TEST_THROWS(smallField = static_cast<nitf::Int8>(15));
    smallField = static_cast<nitf::Int16>(3);
    TEST_ASSERT_EQ(smallField.toString(), "3");

    nitf::AlphaNumericField<20> largeField;
    largeField = nitf::Int64(3e9);
    TEST_ASSERT_EQ(largeField.toString().size(), 20);
    TEST_ASSERT_EQ(largeField.toString(),
            "3000000000          ");

    nitf::AlphaNumericField<5> mediumField;
    TEST_THROWS(mediumField = static_cast<double>(123456));
    mediumField = static_cast<double>(12345);
    TEST_ASSERT_EQ(mediumField.toString(), "12345");
    mediumField = static_cast<double>(1234);
    TEST_ASSERT_EQ(mediumField.toString(), "1234 ");
    mediumField = static_cast<double>(12);
    TEST_ASSERT_EQ(mediumField.toString(), "12.00");
    mediumField = static_cast<double>(12.3454143);
    TEST_ASSERT_EQ(mediumField.toString(), "12.35");
}

TEST_CASE(testBSCADateTime)
{
    nitf::AlphaNumericField<20> field;
    field = nitf::DateTime(2017, 1, 23);
    TEST_ASSERT_EQ(field.toString(),
            "20170123000000      ");
    if (field.asDateTime() != nitf::DateTime(2017, 1, 23))
    {
        TEST_ASSERT(false);
    }
    nitf::AlphaNumericField<3> smallField;
    TEST_THROWS(smallField = nitf::DateTime());
    TEST_THROWS(smallField.asDateTime());
}

TEST_CASE(testRoundTripData)
{
    // nitf::Field defined operator nitf::Int8, then throws if you
    // try to use it. Is operator desired?
    nitf::AlphaNumericField<10> field;
    nitf::Int8 data = 8;
    field = data;
    nitf::Int8 roundTripped = field;
    TEST_ASSERT_EQ(roundTripped, data);

    // nitf::Field doesn't worry about overflow
    // Should this?
    field = 40000;
    nitf::Int16 overflowingInt = field;
    TEST_ASSERT_EQ(overflowingInt, -25536);

    field = 32.45;
    double real = field;
    TEST_ASSERT_EQ(32.45, real);

    field = "test";
    const std::string operatorString = field;
    std::string roundTrippedString = "test" + std::string(6, ' ');
    TEST_ASSERT_EQ(operatorString, roundTrippedString);
}

int main(int /*argc*/, char** /*argv*/)
{
    try
    {
        TEST_CHECK(testSettingBSCAField);
        TEST_CHECK(testSettingBSCAFieldWithNumericData);
        TEST_CHECK(testBSCADateTime);
        TEST_CHECK(testRoundTripData);
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

