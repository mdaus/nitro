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
 * License along with this program; if not, If not,
 * see <http://www.gnu.org/licenses/>.
 *
 */

#include "nitf/Field.hpp"

void nitf::Field::get_(NITF_DATA* outval, nitf_ConvType vtype, size_t length) const
{
    nitf_Error e;
    const NITF_BOOL x = nitf_Field_get(getNativeOrThrow(), outval, vtype, length, &e);
    if (!x)
        throw nitf::NITFException(&e);
}

void nitf::Field::set(NITF_DATA* inval, size_t length)
{
    const NITF_BOOL x = nitf_Field_setRawData(getNativeOrThrow(), inval, length, &error);
    if (!x)
        throw nitf::NITFException(&error);
}
