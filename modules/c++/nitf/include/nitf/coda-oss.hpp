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

#pragma once

#pragma warning(push)
#pragma warning(disable: 26432) // If you define or delete any default operation in the type '...', define or delete them all(c.21).
#pragma warning(disable: 26455) // Default constructor may not throw. Declare it '...' (f.6).
#pragma warning(disable: 26440) // Function '...' can be declared '...' (f.6).

#include <std/bit> // std::endian
#include <std/cstddef> // std::byte
#include <std/filesystem>
#include <std/span>

#pragma warning(disable: 26447) // The function is declared '...' but calls function '..' which may throw exceptions (f.6).
#pragma warning(disable: 26433) // Function '...' should be marked with '...' (c.128).
#pragma warning(disable: 26481) // Don't use pointer arithmetic. Use span instead (bounds.1).
#pragma warning(disable: 26487) // Don't return a pointer '...' that may be invalid (lifetime.4).
 
 // TODO: get rid of these someday? ... from Visual Studio code-analysis
#pragma warning(disable: 26401) // Do not delete a raw pointer that is not an owner<T>(i.11).
#pragma warning(disable: 26434) // Function '...' hides a non-virtual function '...' (c.128).
#pragma warning(disable: 26446) // Prefer to use gsl::at() instead of unchecked subscript operator (bounds.4)
#pragma warning(disable: 26481) // Don't use pointer arithmetic. Use span instead (bounds.1).
#pragma warning(disable: 26485) // Expression '...' : No array to pointer decay(bounds.3).
#pragma warning(disable: 26486) // Don't pass a pointer that may be invalid to a function. Parameter 3 '...' in call to '...' may be invalid (lifetime.3).
#pragma warning(disable: 26487) // Don't return a pointer '...' that may be invalid(lifetime.4).
#pragma warning(disable: 26440) // Function '...' can be declared '...' (f.6).
#pragma warning(disable: 26409) // Avoid calling new and delete explicitly, use std::make_unique<T> instead (r.11).
#pragma warning(disable: 26456) // Operator '...' hides a non-virtual operator '...' (c.128).
#pragma warning(disable: 26429) // Symbol '...' is never tested for nullness, it can be marked as not_null (f.23).

//#define CODA_OSS_Throwable_isa_std_exception 1 // except::Throwable derives from std::exception
#include <sys/Conf.h>
#include <except/Throwable.h>

#include <import/str.h>
#include <import/sys.h>
#pragma warning(push)
#pragma warning(disable: 5039) //	'...': pointer or reference to potentially throwing function passed to 'extern "C"' function under - EHc.Undefined behavior may occur if this function throws an exception.
#include <import/mt.h>
#pragma warning(pop)
#include <import/io.h>

#include <import/types.h>
#include <import/mem.h>
#include <gsl/gsl.h>

#pragma warning(pop)
