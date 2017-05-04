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
 * License along with this program; If not,
 * see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef __NITF_UTILS_HPP__
#define __NITF_UTILS_HPP__

#include "nitf/System.hpp"
#include <string>

namespace nitf
{
struct Utils
{
    static bool isNumeric(std::string str);

    static bool isAlpha(std::string str);

    static void decimalToGeographic(double decimal, int* degrees, int* minutes,
                                    double* seconds);

    static double geographicToDecimal(int degrees, int minutes, double seconds);

    static double parseDecimalString(const std::string& decimal);

    static double geographicStringToDecimal(const std::string& geo);

    static char cornersTypeAsCoordRep(nitf::CornersType type);

    static std::string decimalLatToGeoString(double decimal);

    static std::string decimalLonToGeoString(double decimal);

    static std::string decimalLatToString(double decimal);

    static std::string decimalLonToString(double decimal);

    static std::pair<std::string, std::string> splitIgeolo(
            const std::string& imageCoordinates, size_t index);

    const static size_t LAT_STRING_LENGTH = 7;

    const static size_t LON_STRING_LENGTH = 8;

};

}

#endif
