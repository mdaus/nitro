/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2004 - 2016, MDA Information Systems LLC
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


#include <sys/Conf.h>
#include "nitf/Utils.hpp"

using namespace nitf;

bool Utils::isNumeric(std::string str)
{
    return (bool) nitf_Utils_isNumeric((char*) str.c_str());
}

bool Utils::isAlpha(std::string str)
{
    return (bool) nitf_Utils_isNumeric((char*) str.c_str());
}

void Utils::decimalToGeographic(double decimal, int* degrees, int* minutes,
                                double* seconds)
{
    nitf_Utils_decimalToGeographic(decimal, degrees, minutes, seconds);
}

double Utils::geographicToDecimal(int degrees, int minutes, double seconds)
{
    return nitf_Utils_geographicToDecimal(degrees, minutes, seconds);
}

char Utils::cornersTypeAsCoordRep(nitf::CornersType type)
{
    return nitf_Utils_cornersTypeAsCoordRep(type);
}

std::string Utils::decimalLatToGeoString(double decimal)
{
    char buffer[LAT_STRING_LENGTH + 1];
    buffer[LAT_STRING_LENGTH] = '\0';
    nrt_Utils_decimalLatToGeoCharArray(decimal, buffer);
    return std::string(buffer);
}

std::string Utils::decimalLonToGeoString(double decimal)
{
    char buffer[LON_STRING_LENGTH + 1];
    buffer[LON_STRING_LENGTH] = '\0';
    nitf_Utils_decimalLonToGeoCharArray(decimal, buffer);
    return std::string(buffer);
}

std::string Utils::decimalLatToString(double decimal)
{
    char buffer[LAT_STRING_LENGTH + 1];
    buffer[LAT_STRING_LENGTH] = 0;
    nitf_Utils_decimalLatToCharArray(decimal, buffer);
    return std::string(buffer);
}

std::string Utils::decimalLonToString(double decimal)
{
    char buffer[LON_STRING_LENGTH + 1];
    buffer[LON_STRING_LENGTH] = '\0';
    nitf_Utils_decimalLonToCharArray(decimal, buffer);
    return std::string(buffer);
}

double Utils::geographicStringToDecimal(const std::string& geo)
{
    int degrees;
    int minutes;
    double seconds;
    nitf_Error error;
    if (!nitf_Utils_parseGeographicString((char*) geo.c_str(), &degrees,
            &minutes, &seconds, &error))
    {
        throw except::Exception(Ctxt("Unable to parse " + geo));
    }
    return nitf_Utils_geographicToDecimal(degrees, minutes, seconds);
}

double Utils::parseDecimalString(const std::string& decimal)
{
    double result;
    nitf_Error error;
    if (!nitf_Utils_parseDecimalString((char*) decimal.c_str(), &result, &error))
    {
        throw except::Exception(Ctxt("Unable to parse " + decimal));
    }
    return result;
}

std::pair<std::string, std::string> Utils::splitIgeolo(
        const std::string& imageCoordinates, size_t index)
{
    if (index > 3)
    {
        throw except::Exception(Ctxt("Index must be in [0:3]. "
                "Got " + str::toString(index)));
    }
    if (imageCoordinates.size() != 60)
    {
        throw except::Exception(Ctxt("ImageCoordinates should be "
                "a string of 60 characters"));
    }

    std::pair<std::string, std::string> coordinates;
    const size_t startPosition = index *
            (LAT_STRING_LENGTH + LON_STRING_LENGTH);
    coordinates.first = imageCoordinates.substr(
            startPosition, LAT_STRING_LENGTH);
    coordinates.second = imageCoordinates.substr(
            startPosition + LAT_STRING_LENGTH, LON_STRING_LENGTH);
    return coordinates;
}

