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

#include "coda_oss/optional_.h"
#include "bpstd/optional.hpp"
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
