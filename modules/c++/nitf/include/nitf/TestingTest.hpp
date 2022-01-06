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

#include "nitf/TestingTest.h"
#include "nitf/System.hpp"
#include "nitf/Field.hpp"
#include "nitf/Object.hpp"

namespace nitf
{

// wrap the test classes, ala FileHeader, etc.

DECLARE_CLASS(testing_Test1a)
{
public:
    testing_Test1a() noexcept(false);
    ~testing_Test1a() = default;
    testing_Test1a(const testing_Test1a & x);
    testing_Test1a& operator=(const testing_Test1a & x);

    //! Set native object
    using native_t = nitf_testing_Test1a;
    testing_Test1a(native_t * x);

    std::vector<nitf::Field> getFields() const;

private:
    mutable nitf_Error error{};
};

DECLARE_CLASS(testing_Test1b)
{
public:
    testing_Test1b() noexcept(false);
    ~testing_Test1b() = default;
    testing_Test1b(const testing_Test1b & x);
    testing_Test1b& operator=(const testing_Test1b & x);

    //! Set native object
    using native_t = nitf_testing_Test1b;
    testing_Test1b(native_t * x);

    std::vector<nitf::Field> getFields() const;

private:
    mutable nitf_Error error{};
};

}
#endif // NITRO_nitf_FieldDescriptor_hpp_INCLUDED_
