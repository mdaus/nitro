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

#ifndef __NITF_SYSTEM_HPP__
#define __NITF_SYSTEM_HPP__

/*!
 *  \file System.hpp
 */

#include "nitf/System.h"
#include "nitf/Field.h"
#include "nitf/Types.h"

namespace nitf
{
typedef uint64_t Uint64;
typedef uint32_t Uint32;
typedef uint16_t Uint16;
typedef uint8_t Uint8;
typedef int64_t Int64;
typedef int32_t Int32;
typedef int16_t Int16;
typedef int8_t Int8;
typedef nitf_Off Off;
typedef nitf_Version Version;
typedef nitf_ConvType ConvType;
typedef nitf_FieldType FieldType;
typedef nitf_AccessFlags AccessFlags;
typedef nitf_CreationFlags CreationFlags;
typedef nitf_CornersType CornersType;
}
#endif
