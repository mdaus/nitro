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

#ifndef __NITF_TYPES_H__
#define __NITF_TYPES_H__
#pragma once

/* Enum for the supported version types */
typedef enum _nitf_Version
{
    NITF_VER_20 = 100,
    NITF_VER_21,
    NITF_VER_UNKNOWN
} nitf_Version;

/* These macros check the NITF Version */
#define IS_NITF20(V) ((V) == NITF_VER_20)
#define IS_NITF21(V) ((V) == NITF_VER_21)


typedef void NITF_DATA;

//
// Support a simple "reflection" scheme.
//

/* Enum for the types of fields in structs */
typedef enum nitf_StructFieldType_
{
    NITF_FieldType_Field, // nitf_Field*
    NITF_FieldType_FileSecurity, // nitf_FileSecurity*
    NITF_FieldType_ComponentInfo, // nitf_ComponentInfo*
    NITF_FieldType_Extensions, // nitf_Extensions*
} nitf_StructFieldType;

typedef struct nitf_StructFieldDescriptor_
{
    nitf_StructFieldType type;
    const char* name;
    size_t offset;
} nitf_StructFieldDescriptor;

#define NITF_StructFieldDescriptor_value_(type, s, m) {type, #m, offsetof(s, m) }
#define NITF_StructFieldDescriptor_value(name, s, m) NITF_StructFieldDescriptor_value_(NITF_FieldType_##name, nitf_##s, m)

#define NITF_DECLARE_struct_(name, fields, descriptors) typedef struct _##nitf_##name { fields; } nitf_##name; \
    static const nitf_StructFieldDescriptor nitf_##name##_fields[] = { descriptors }

#define NITF_DECLARE_struct_2_(t1, f1, t2, f2) nitf_##t1* f1; nitf_##t2* f2
#define NITF_StructFieldDescriptor_value_2_(name, t1, f1, t2, f2) \
    NITF_StructFieldDescriptor_value(t1, name, f1), \
    NITF_StructFieldDescriptor_value(t2, name, f2)
#define NITF_DECLARE_struct_2(name, t1, f1, t2, f2) \
    NITF_DECLARE_struct_(name, NITF_DECLARE_struct_2_(t1, f1, t2, f2), NITF_StructFieldDescriptor_value_2_(name, t1, f1, t2, f2))

#define NITF_DECLARE_struct_3_(t1, f1, t2, f2, t3, f3)  NITF_DECLARE_struct_2_(t1, f1, t2, f2); nitf_##t3* f3
#define NITF_StructFieldDescriptor_value_3_(name, t1, f1, t2, f2, t3, f3) \
    NITF_StructFieldDescriptor_value_2_(name, t1, f1, t2, f2), \
    NITF_StructFieldDescriptor_value(t3, name, f3)
#define NITF_DECLARE_struct_3(name, t1, f1, t2, f2, t3, f3) NITF_DECLARE_struct_(name, \
    NITF_DECLARE_struct_3_(t1, f1, t2, f2, t2, f3), \
    NITF_StructFieldDescriptor_value_3_(name, t1, f1, t2, f2, t3, f3))

#define NITF_DECLARE_struct_4_(t1, f1, t2, f2, t3, f3, t4, f4)  NITF_DECLARE_struct_3_(t1, f1, t2, f2, t3, f3); nitf_##t4* f4
#define NITF_StructFieldDescriptor_value_4_(name, t1, f1, t2, f2, t3, f3, t4, f4) \
    NITF_StructFieldDescriptor_value_3_(name, t1, f1, t2, f2, t3, f3), \
    NITF_StructFieldDescriptor_value(t4, name, f4)
#define NITF_DECLARE_struct_4(name, t1, f1, t2, f2, t3, f3, t4, f4) NITF_DECLARE_struct_(name, \
    NITF_DECLARE_struct_4_(t1, f1, t2, f2, t2, f3, t4, f4), \
    NITF_StructFieldDescriptor_value_4_(name, t1, f1, t2, f2, t3, f3, t4, f4))

#endif
