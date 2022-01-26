/* =========================================================================
 * This file is part of coda_oss-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2018, MDA Information Systems LLC
 * (C) Copyright 2022, Maxar Technologies, Inc.
 *
 * coda_oss-c++ is free software; you can redistribute it and/or modify
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

#ifndef CODA_OSS_coda_oss_memory_h_INCLUDED_
#define CODA_OSS_coda_oss_memory_h_INCLUDED_
#pragma once

#include <utility>
#include <memory>

#include "config/compiler_extensions.h"

#include "coda_oss/memory_.h"

CODA_OSS_disable_warning_push
#if _MSC_VER
#pragma warning(disable: 26409) // Avoid calling new and delete explicitly, use std::make_unique<T> instead (r .11).
__pragma(warning(disable: 26409)) // Avoid calling new and delete explicitly, use std::make_unique<T> instead (r .11).
#endif
#include "bpstd/memory.hpp"
CODA_OSS_disable_warning_pop

#include "coda_oss/bpstd_.h"

namespace coda_oss
{
    template <typename T, typename... TArgs>
    std::unique_ptr<T> make_unique(TArgs&&... args)
    {
        CODA_OSS_disable_warning_push
        #if _MSC_VER
        #pragma warning(disable: 26409) // Avoid calling new and delete explicitly, use std::make_unique<T> instead (r .11).
        __pragma(warning(disable: 26409)) // Avoid calling new and delete explicitly, use std::make_unique<T> instead (r .11).
        #endif

        // Let the actual make_unique implementation do all the template magic. 
        #if CODA_OSS_coda_oss_USE_BPSTD_
        return bpstd::make_unique<T>(std::forward<TArgs>(args)...);
        #else
        return details::make_unique<T>(std::forward<TArgs>(args)...);
        #endif

        CODA_OSS_disable_warning_pop
    }
} // namespace coda_oss

#endif // CODA_OSS_coda_oss_memory_h_INCLUDED_
