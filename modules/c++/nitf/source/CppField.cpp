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

#include <mem/ScopedArray.h>
#include <sys/Conf.h>
#include <except/Exception.h>
#include <str/Manip.h>
#include <nitf/CppField.hpp>

namespace nitf
{
std::string StringConverter::toNitfString(const std::string& value,
        size_t length) const
{
    if (length == 0)
    {
        length = mLength;
    }
    if (value.size() > length)
    {
        throw except::Exception(Ctxt(value + " is too long. Should be <= " +
                str::toString(mLength)));
    }
    if (mFieldType == NITF_BCS_N && !str::containsOnly(value, "0123456789.+-"))
    {
        throw except::Exception(Ctxt(value + " includes characters not in "
                    "BCS-N"));
    }
    const size_t padLength = length - value.size();
    char padCharacter = mFieldType == NITF_BCS_A ? ' ' : '0';
    const std::string padding(padLength, padCharacter);
    return mFieldType == NITF_BCS_A ? value + padding : padding + value;
}

std::string StringConverter::toNitfString(const DateTime& value) const
{
    const std::string formattedDateTime = value.format(NITF_DATE_FORMAT_21);
    if (mLength == std::string("YYYYMMDD").size())
    {
        return formattedDateTime.substr(0, mLength);
    }
    return toNitfString(formattedDateTime);
}

std::string StringConverter::toNitfString(double value) const
{
    return toNitfString(realToString(value));
}

std::string StringConverter::toNitfString(float value) const
{
    return toNitfString(realToString(value));
}

std::string StringConverter::realToString(double value) const
{
    const std::string integerPart =
            str::toString(static_cast<Int64>(value));
    if (integerPart.size() > mLength)
    {
        throw except::Exception(Ctxt(str::toString(value) +
                    " is too long. Should be <= " +
                    str::toString(mLength) + "digits long"));
    }
    else if (integerPart.size() == mLength ||
            integerPart.size() == mLength - 1)
    {
        return str::toString(static_cast<Int64>(value + .5));
    }
    else
    {
        std::ostringstream stream;
        stream << std::fixed
               << std::setprecision(mLength - integerPart.size() - 1)
               << value;
        return stream.str();
    }
}
}

