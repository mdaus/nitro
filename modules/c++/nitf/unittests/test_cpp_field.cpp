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
#include <nitf/DateTime.hpp>
#include "TestCase.h"

TEST_CASE(testSettingBSCAField)
{
    nitf::BCSA<std::string, 2> stringField;
    TEST_THROWS(stringField = "foo");
    stringField = "a ";
    TEST_ASSERT_EQ(stringField.toString(), "a ");

    nitf::BCSA<std::string, 5> largeStringField;
    largeStringField = "";
    TEST_ASSERT_EQ(largeStringField.toString(), "     ");

    nitf::BCSA<nitf::Int8, 1> smallIntField;
    TEST_THROWS(smallIntField = 50);
    smallIntField = 5;
    TEST_ASSERT_EQ(smallIntField.toString(), "5");

    nitf::BCSA<nitf::Int64, 20> largeIntField;
    largeIntField = nitf::Int64(3e9);
    TEST_ASSERT_EQ(largeIntField.toString().size(), 20);
    TEST_ASSERT_EQ(largeIntField.toString(),
            "3000000000          ");

    nitf::BCSA<nitf::Uint32, 10> largeUnsignedField;
    largeUnsignedField = nitf::Uint32(3e9);
    TEST_ASSERT_EQ(largeUnsignedField.toString(),
            "3000000000");

    nitf::BCSA<double, 5> doubleField;
    TEST_THROWS(doubleField = static_cast<double>(123456));
    doubleField = static_cast<double>(12345);
    TEST_ASSERT_EQ(doubleField.toString(), "12345");
    doubleField = static_cast<double>(1234);
    TEST_ASSERT_EQ(doubleField.toString().size(), 5);
    TEST_ASSERT_EQ(doubleField.toString(), "1234 ");
    doubleField = static_cast<double>(12);
    TEST_ASSERT_EQ(doubleField.toString(), "12.00");
    doubleField = static_cast<double>(12.3454143);
    TEST_ASSERT_EQ(doubleField.toString(), "12.35");

    nitf::BCSA<std::string, 5> emptyString;
    TEST_ASSERT_EQ(emptyString.toString(), "     ");
    nitf::BCSA<nitf::Uint32, 3> emptyNumber;
    TEST_ASSERT_EQ(emptyNumber.toString(), "   ");
}

TEST_CASE(testSettingBSCNField)
{
    nitf::BCSN<nitf::Uint8, 2> numField;
    numField = 1;
    TEST_ASSERT_EQ(numField.toString(), "01");

    nitf::BCSN<float, 5> floatField;
    floatField = 12;
    TEST_ASSERT_EQ(floatField.toString(), "12.00");
    floatField = 1.04;
    TEST_ASSERT_EQ(floatField.toString(), "1.040");
    TEST_ASSERT_ALMOST_EQ(floatField, 1.04);

    floatField = -13;
    TEST_ASSERT_EQ(floatField.toString(), "-13.0");

    nitf::BCSN<std::string, 10> stringNumericField;
    TEST_THROWS(stringNumericField = "a");

    nitf::BCSN<nitf::Int32, 5> negativeField(-5);
    TEST_ASSERT_EQ(negativeField.toString(), "-0005");

    nitf::BCSN<nitf::Int32, 3> emptyField;
    TEST_ASSERT_EQ(emptyField.toString(), "000");

}

TEST_CASE(testBSCADateTime)
{
    nitf::BCSA<nitf::DateTime, 14> datetimeField;
    datetimeField = nitf::DateTime(2017, 1, 23);
    TEST_ASSERT_EQ(datetimeField.toString(),
            "20170123000000");
    if (datetimeField.getValue() != nitf::DateTime(2017, 1, 23))
    {
        TEST_ASSERT(false);
    }

    bool constructorThrows = false;
    try
    {
        nitf::BCSA<nitf::DateTime, 3>();
    }
    catch(const except::Exception& /*ex*/)
    {
        constructorThrows = true;
    }
    TEST_ASSERT(constructorThrows);

    nitf::BCSA<nitf::DateTime, 8> shortDateField;
    shortDateField = nitf::DateTime(2017, 1, 23);
    TEST_ASSERT_EQ(shortDateField.toString(), "20170123");
}

TEST_CASE(testRoundTripData)
{
    // nitf::Field defined operator nitf::Int8, then throws if you
    // try to use it. Is operator desired?
    nitf::BCSA<nitf::Int8, 10> field;
    nitf::Int8 data = 8;
    field = data;
    nitf::Int8 roundTripped = field.getValue();
    TEST_ASSERT_EQ(roundTripped, data);

    // nitf::Field doesn't worry about overflow
    // Should this?
    nitf::BCSA<nitf::Int16, 10> shortField(40000);
    nitf::Int16 overflowingInt = shortField.getValue();
    TEST_ASSERT_EQ(overflowingInt, -25536);

    nitf::BCSA<double, 10> doubleField;
    doubleField = 32.45;
    double real = doubleField.getValue();
    TEST_ASSERT_EQ(32.45, real);

    nitf::BCSA<std::string, 10> stringField("test");
    TEST_ASSERT_EQ("test", stringField.getValue());
}

int main(int /*argc*/, char** /*argv*/)
{
    try
    {
        TEST_CHECK(testSettingBSCAField);
        TEST_CHECK(testSettingBSCNField);
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

