/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
 * (C) Copyright 2022 , Maxar Technologies, Inc.
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

#ifndef NITRO_nitf_FieldDescriptor_hpp_INCLUDED_
#define NITRO_nitf_FieldDescriptor_hpp_INCLUDED_
#pragma once

#include <string>

#include "nitf/FieldDescriptor.h"

namespace nitf
{
    // wrap nitf_StructFieldDescriptor
    class FieldDescriptor final
    {
        const nitf_StructFieldDescriptor* pNative = nullptr;
    public:
        enum class Type
        {
            Field = NITF_FieldType_Field, // nitf_Field*
            FileSecurity = NITF_FieldType_FileSecurity, // nitf_FileSecurity*
            ComponentInfo = NITF_FieldType_ComponentInfo, // nitf_ComponentInfo*
            PComponentInfo = NITF_FieldType_PComponentInfo, // nitf_ComponentInfo** aka nitf_PComponentInfo*
            Extensions = NITF_FieldType_Extensions, // nitf_Extensions*
        };
        FieldDescriptor(const nitf_StructFieldDescriptor& native) noexcept : pNative(&native) {}
        ~FieldDescriptor() = default;
        FieldDescriptor(const FieldDescriptor&) = default;
        FieldDescriptor& operator=(const FieldDescriptor&) = default;
        FieldDescriptor(FieldDescriptor&&) = default;
        FieldDescriptor& operator=(FieldDescriptor&&) = default;

        Type type() const noexcept
        {
            return static_cast<Type>(pNative->type);
        }
        std::string name() const
        {
            return pNative->name;
        }
        size_t offset() const noexcept
        {
            return pNative->offset;
        }
    };
}
#endif // NITRO_nitf_FieldDescriptor_hpp_INCLUDED_
