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
 * License along with this program;
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <import/nrt.h>
#include "Test.h"

TEST_CASE(testParseZeroDegrees)
{
    const char* geoString = "001234S";
    int degrees;
    int minutes;
    double seconds;
    nrt_Error error;
    double decimal;
    nrt_Utils_parseGeographicString(geoString, &degrees, &minutes, &seconds,
            &error);
    decimal = nrt_Utils_geographicToDecimal(degrees, minutes, seconds);
    TEST_ASSERT(fabs(decimal - -.2094444) < 1e-6);
}

TEST_CASE(testParseZeroMinutes)
{
    const char* geoString = "0000034W";
    int degrees;
    int minutes;
    double seconds;
    nrt_Error error;
    double decimal;
    nrt_Utils_parseGeographicString(geoString, &degrees, &minutes, &seconds,
            &error);
    decimal = nrt_Utils_geographicToDecimal(degrees, minutes, seconds);
    TEST_ASSERT(fabs(decimal - -.009444) < 1e-6);
}

TEST_CASE(testParseZeroMinutesEast)
{
    const char* geoString = "0000034E";
    int degrees;
    int minutes;
    double seconds;
    nrt_Error error;
    double decimal;
    nrt_Utils_parseGeographicString(geoString, &degrees, &minutes, &seconds,
            &error);
    decimal = nrt_Utils_geographicToDecimal(degrees, minutes, seconds);
    TEST_ASSERT(fabs(decimal - .009444) < 1e-6);

}

TEST_CASE(testDecimalToDmsNegativeMinutes)
{
    const double decimal = -0.2094444;
    int degrees;
    int minutes;
    double seconds;
    nrt_Utils_decimalToGeographic(decimal, &degrees, &minutes, &seconds);
    TEST_ASSERT(degrees == 0);
    TEST_ASSERT(minutes == -12);
    TEST_ASSERT(fabs(seconds - 34) < 1);
}

TEST_CASE(testDecimalToDmsPositiveMinutes)
{
    const double decimal = 0.2094444;
    int degrees;
    int minutes;
    double seconds;
    nrt_Utils_decimalToGeographic(decimal, &degrees, &minutes, &seconds);
    TEST_ASSERT(degrees == 0);
    TEST_ASSERT(minutes == 12);
    TEST_ASSERT(fabs(seconds - 34) < 1);
}

TEST_CASE(testDecimalToDmsNegativeSeconds)
{
    const double decimal = -0.009444;
    int degrees;
    int minutes;
    double seconds;
    nrt_Utils_decimalToGeographic(decimal, &degrees, &minutes, &seconds);
    TEST_ASSERT(degrees == 0);
    TEST_ASSERT(minutes == 0);
    TEST_ASSERT(fabs(seconds - -34) < 1);
}

TEST_CASE(testDecimalToDmsPositiveSeconds)
{
    const double decimal = 0.009444;
    int degrees;
    int minutes;
    double seconds;
    nrt_Utils_decimalToGeographic(decimal, &degrees, &minutes, &seconds);
    TEST_ASSERT(degrees == 0);
    TEST_ASSERT(minutes == 0);
    TEST_ASSERT(fabs(seconds - 34) < 1);
}

TEST_CASE(testParseDecimal)
{
    const char* decimalString = "+12.345";
    double decimal;
    nrt_Error error;
    nrt_Utils_parseDecimalString(decimalString, &decimal, &error);
    TEST_ASSERT(fabs(decimal - 12.345) < 1e-6);
}

TEST_CASE(testDmsToCharArrayPositiveDMS)
{
    char lonCharArray[9];
    char latCharArray[8];

    nrt_Utils_geographicLonToCharArray(1, -1, 13, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "001-113E");
    nrt_Utils_geographicLatToCharArray(1, -1, 13, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "01-113N");

    nrt_Utils_geographicLonToCharArray(1, 0, 73, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "0010113E");
    nrt_Utils_geographicLatToCharArray(1, 0, 73, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "010113N");

    nrt_Utils_geographicLonToCharArray(1, -1, 73, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "0010013E");
    nrt_Utils_geographicLatToCharArray(1, -1, 73, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "010013N");

    nrt_Utils_geographicLonToCharArray(1, 60, 73, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "0020113E");
    nrt_Utils_geographicLatToCharArray(1, 60, 73, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "020113N");
}

TEST_CASE(testDmsToCharArrayNegativeDMS)
{
    char lonCharArray[9];
    char latCharArray[8];

    nrt_Utils_geographicLonToCharArray(-1, -1, 13, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "001-113W");
    nrt_Utils_geographicLatToCharArray(-1, -1, 13, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "01-113S");

    nrt_Utils_geographicLonToCharArray(-1, 0, 73, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "0010113W");
    nrt_Utils_geographicLatToCharArray(-1, 0, 73, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "010113S");

    nrt_Utils_geographicLonToCharArray(-1, -1, 73, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "0010013W");
    nrt_Utils_geographicLatToCharArray(-1, -1, 73, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "010013S");

    nrt_Utils_geographicLonToCharArray(-1, 60, 73, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "0020113W");
    nrt_Utils_geographicLatToCharArray(-1, 60, 73, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "020113S");
}

TEST_CASE(testDmsToCharArrayNegativeDegrees)
{
    char lonCharArray[9];
    char latCharArray[8];

    nrt_Utils_geographicLonToCharArray(-1, 0, 0, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "0010000W");
    nrt_Utils_geographicLatToCharArray(-1, 0, 0, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "010000S");

    nrt_Utils_geographicLonToCharArray(-181, 0, 0, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "1810000W");
    nrt_Utils_geographicLatToCharArray(-181, 0, 0, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "1810000");

    nrt_Utils_geographicLonToCharArray(-361, 0, 0, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "3610000W");
    nrt_Utils_geographicLatToCharArray(-361, 0, 0, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "3610000");
}

TEST_CASE(testDmsToCharArrayPositiveDegrees)
{
    char lonCharArray[9];
    char latCharArray[8];

    nrt_Utils_geographicLonToCharArray(0, 1, 0, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "0000100E");
    nrt_Utils_geographicLatToCharArray(0, 1, 0, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "000100N");

    nrt_Utils_geographicLonToCharArray(0, 1, 13, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "0000113E");
    nrt_Utils_geographicLatToCharArray(0, 1, 13, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "000113N");

    nrt_Utils_geographicLonToCharArray(0, 61, 0, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "0006100E");
    nrt_Utils_geographicLatToCharArray(0, 61, 0, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "006100N");
}

TEST_CASE(testDmsToCharArrayNegativeMinutes)
{
    char lonCharArray[9];
    char latCharArray[8];

    nrt_Utils_geographicLonToCharArray(0, -1, 0, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "0000100W");
    nrt_Utils_geographicLatToCharArray(0, -1, 0, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "000100S");

    nrt_Utils_geographicLonToCharArray(0, -1, 13, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "0000113W");    
    nrt_Utils_geographicLatToCharArray(0, -1, 13, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "000113S");

    nrt_Utils_geographicLonToCharArray(0, -61, 0, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "0006100W");
    nrt_Utils_geographicLatToCharArray(0, -61, 0, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "006100S");
}

TEST_CASE(testDmsToCharArrayPositiveMinutes)
{
    char lonCharArray[9];
    char latCharArray[8];

    nrt_Utils_geographicLonToCharArray(0, 1, 0, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "0000100E");
    nrt_Utils_geographicLatToCharArray(0, 1, 0, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "000100N");

    nrt_Utils_geographicLonToCharArray(0, 1, 13, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "0000113E");
    nrt_Utils_geographicLatToCharArray(0, 1, 13, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "000113N");

    nrt_Utils_geographicLonToCharArray(0, 61, 0, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "0006100E");
    nrt_Utils_geographicLatToCharArray(0, 61, 0, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "006100N");
}

TEST_CASE(testDmsToCharArrayNegativeSeconds)
{
    char lonCharArray[9];
    char latCharArray[8];

    nrt_Utils_geographicLonToCharArray(0, 0, -13, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "0000013W");

    nrt_Utils_geographicLatToCharArray(0, 0, -13, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "000013S");

    nrt_Utils_geographicLonToCharArray(0, 1, -73, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "00001-72");

    nrt_Utils_geographicLatToCharArray(0, 1, -73, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "0001-72");

    nrt_Utils_geographicLonToCharArray(0, 0, -73, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "0000113W");

    nrt_Utils_geographicLatToCharArray(0, 0, -73, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "000113S");
}

TEST_CASE(testDmsToCharArrayPositiveSeconds)
{
    char lonCharArray[9];
    char latCharArray[8];

	nrt_Utils_geographicLonToCharArray(0, 0, 13, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "0000013E");

	nrt_Utils_geographicLatToCharArray(0, 0, 13, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "000013N");

    nrt_Utils_geographicLonToCharArray(0, 0, 73, lonCharArray);
    TEST_ASSERT_EQ_STR(lonCharArray, "0000113E");

    nrt_Utils_geographicLatToCharArray(0, 0, 73, latCharArray);
    TEST_ASSERT_EQ_STR(latCharArray, "000113N");
}

TEST_CASE(testDmsToCharArrayZero)
{
    char lonCharArray[9];
    nrt_Utils_geographicLonToCharArray(0, 0, 0, lonCharArray);
    TEST_ASSERT(strcmp(lonCharArray, "0000000E") == 0);
}

TEST_MAIN(
    (void)argc;
    (void)argv;
    CHECK(testParseZeroDegrees);
    CHECK(testParseZeroMinutes);
    CHECK(testParseZeroMinutesEast);
    CHECK(testDecimalToDmsNegativeMinutes);
    CHECK(testDecimalToDmsPositiveMinutes);
    CHECK(testDecimalToDmsNegativeSeconds);
    CHECK(testDecimalToDmsPositiveSeconds);
    CHECK(testParseDecimal);
    CHECK(testDmsToCharArrayNegativeDMS);
    CHECK(testDmsToCharArrayPositiveDMS);
    CHECK(testDmsToCharArrayNegativeDegrees);
    CHECK(testDmsToCharArrayPositiveDegrees);
    CHECK(testDmsToCharArrayNegativeMinutes);
    CHECK(testDmsToCharArrayPositiveMinutes);
    CHECK(testDmsToCharArrayNegativeSeconds);
    CHECK(testDmsToCharArrayPositiveSeconds);
    CHECK(testDmsToCharArrayZero);
    )

