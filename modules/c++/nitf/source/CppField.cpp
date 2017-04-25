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
AlphaNumericFieldImpl::AlphaNumericFieldImpl(size_t size):
    mSize(size)
{
    mData.reserve(size);
}

void AlphaNumericFieldImpl::set(const std::string& value)
{
    if (value.size() > mSize)
    {
        throw except::Exception(Ctxt(value + " is too long. Should be <= " +
                str::toString(mData.size())));
    }
    const size_t padLength = mSize - value.size();
    const std::string padding(padLength, ' ');
    mData = value + padding;
}

void AlphaNumericFieldImpl::setReal(double value)
{
    const std::string integerPart =
            str::toString(static_cast<nitf::Int64>(value));
    if (integerPart.size() > mSize)
    {
        throw except::Exception(Ctxt(str::toString(value) +
                    " is too long. Should be <= " +
                    str::toString(mData.size()) + "digits long"));
    }
    else if (integerPart.size() == mSize ||
            integerPart.size() == mSize - 1)
    {
        set(str::toString(static_cast<nitf::Int64>(value + .5)));
    }
    else
    {
        std::ostringstream stream;
        stream << std::fixed
               << std::setprecision(mSize - integerPart.size() - 1)
               << value;
        set(stream.str());
        stream.unsetf(std::ios_base::fixed);
        stream.clear();
    }
}

void AlphaNumericFieldImpl::setDateTime(const nitf::DateTime& value)
{
    set(value.format(NITF_DATE_FORMAT_21));
}

nitf::DateTime AlphaNumericFieldImpl::asDateTime(
        const std::string& format) const
{
    return nitf::DateTime(mData, format);
}

}

