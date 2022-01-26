/* =========================================================================
 * This file is part of coda_oss-c++
 * =========================================================================
 *
 * (C) Copyright 2020, Maxar Technologies, Inc.
 *
 * sys-c++ is free software; you can redistribute it and/or modify
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
 * License along with this program; If not, http://www.gnu.org/licenses/.
 *
 */
#ifndef CODA_OSS_coda_oss_optional_h_INCLUDED_
#define CODA_OSS_coda_oss_optional_h_INCLUDED_
#pragma once

#include <utility>

#include "config/compiler_extensions.h"

#include "coda_oss/optional_.h"
CODA_OSS_disable_warning_push
#if _MSC_VER
#pragma warning(disable: 4582) // '...': constructor is not implicitly called
#pragma warning(disable: 4583) // '...': destructor is not implicitly called
#pragma warning(disable: 4625) // '...': copy constructor was implicitly defined as deleted
#pragma warning(disable: 4626) // '...': assignment operator was implicitly defined as deleted
#pragma warning(disable: 5026) // '...': move constructor was implicitly defined as deleted
#pragma warning(disable: 5027) // '...': move assignment operator was implicitly defined as deleted

#pragma warning(disable: 26495) // Variable '...' is uninitialized. Always initialize a member variable (type.6).
#pragma warning(disable: 26455) // Default constructor may not throw. Declare it '...' (f.6).
#pragma warning(disable: 26457) // (void) should not be used to ignore return values, use '...' instead (es.48).
#pragma warning(disable: 26440) // Function '...' can be declared '...'
#pragma warning(disable: 26409) // Avoid calling new and delete explicitly, use std::make_unique<T> instead (r.11).
#endif
#include "bpstd/optional.hpp"
CODA_OSS_disable_warning_pop

#include "coda_oss/bpstd_.h"

namespace coda_oss
{
#if CODA_OSS_coda_oss_USE_BPSTD_
template<typename T>
using optional = bpstd::optional<T>;
#else
template<typename T>
using optional = details::optional<T>;
#endif

// https://en.cppreference.com/w/cpp/utility/optional/make_optional
template <typename T, typename... TArgs>
inline optional<T> make_optional(TArgs&&... args)
{
    #if CODA_OSS_coda_oss_USE_BPSTD_
    return bpstd::make_optional<T>(std::forward<TArgs>(args)...);
    #else
    return details::make_optional<T>(std::forward<TArgs>(args)...);
    #endif
}
}

#endif  // CODA_OSS_coda_oss_optional_h_INCLUDED_
