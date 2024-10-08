/* =========================================================================
 * This file is part of str-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
 * (C) Copyright 2020, 2021, 2022, Maxar Technologies, Inc.
 *
 * str-c++ is free software; you can redistribute it and/or modify
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

#pragma once
#ifndef CODA_OSS_str_Encoding_h_INCLUDED_
#define CODA_OSS_str_Encoding_h_INCLUDED_

#include <string.h>
#include <wchar.h>
#include <stdint.h>

#include <memory>
#include <string>
#include <type_traits>

#include "coda_oss/string.h"
#include "gsl/gsl.h"
#include "config/Exports.h"
#include "str/W1252string.h"

namespace str
{
namespace details
{
template <typename TReturn, typename TChar>
inline auto cast(const TChar* s)
{
    // This is OK as UTF-8 can be stored in std::string
    // Note that casting between the string types will CRASH on some
    // implementations. NO: reinterpret_cast<const std::string&>(value).
    // Instead, use c_str() or str(), below.
    const void* const pStr = s;
    auto const retval = static_cast<TReturn>(pStr);
    static_assert(sizeof(*retval) == sizeof(*s), "sizeof(*TReturn) != sizeof(*TChar)"); 
    return retval;
}
}
template <typename TBasicStringT, typename TChar>
inline auto c_str(const std::basic_string<TChar>& s)
{
    using return_t = typename TBasicStringT::const_pointer;
    return details::cast<return_t>(s.c_str());
}
template <typename TBasicStringT, typename TChar>
inline auto str(const std::basic_string<TChar>& s)
{
    return TBasicStringT(c_str<TBasicStringT>(s), s.length()); // avoid extra strlen() call
}
template <typename TBasicStringT, typename TChar>
inline TBasicStringT make_string(TChar* p)
{
    using const_pointer = typename TBasicStringT::const_pointer;
    return details::cast<const_pointer>(p);  // copy into RV
}

/************************************************************************/
// When the encoding is important, we want to "traffic" in coda_oss::u8string (UTF-8), not
// str::W1252string (Windows-1252) or std::string (unknown).  Make it easy to get those from other encodings.
CODA_OSS_API coda_oss::u8string to_u8string(str::W1252string::const_pointer, size_t);
CODA_OSS_API coda_oss::u8string to_u8string(std::u16string::const_pointer, size_t);
CODA_OSS_API coda_oss::u8string to_u8string(std::u32string::const_pointer, size_t);
inline coda_oss::u8string to_u8string(coda_oss::u8string::const_pointer p, size_t sz)
{
    return coda_oss::u8string(p, sz);
}
// Explicit overloads so template can be used for a different purpose.
inline auto to_u8string(const coda_oss::u8string& s)
{
    return to_u8string(s.c_str(), s.length());
}
inline auto to_u8string(const str::W1252string& s)
{
    return to_u8string(s.c_str(), s.length());
}
inline auto to_u8string(const std::u16string& s)
{
    return to_u8string(s.c_str(), s.length());
}
inline auto to_u8string(const std::u32string& s)
{
    return to_u8string(s.c_str(), s.length());
}
// These two routines are "dangerous" as they make it easy to convert
// a `char*` **already** in UTF-8 encoding to UTF-8; the result is garbage.
// Use u8FromNative() or u8FromNative() which is a bit more explicit.
coda_oss::u8string to_u8string(std::string::const_pointer, size_t) = delete;
coda_oss::u8string to_u8string(std::wstring::const_pointer, size_t) = delete;

// Template parameter specifies how `std::string` is encoded.  As opposed
// to figuring it out a run-time based on the platform.
template <typename TBasicString>
inline auto to_u8string(const std::string& s)  // UTF-8 or Windows-1252
{
    return to_u8string(str::c_str<TBasicString>(s), s.length());
}
template <typename TBasicString>
inline auto to_u8string(const std::wstring& s)  // UTF-16 or UTF-32
{
    return to_u8string(str::c_str<TBasicString>(s), s.length());
}

/************************************************************************/
// UTF-16 is the default on Windows.
CODA_OSS_API std::u16string to_u16string(coda_oss::u8string::const_pointer, size_t);
CODA_OSS_API std::u16string to_u16string(str::W1252string::const_pointer, size_t);
inline auto to_u16string(const coda_oss::u8string& s)
{
    return to_u16string(s.c_str(), s.length());
}
inline auto to_u16string(const str::W1252string& s)
{
    return to_u16string(s.c_str(), s.length());
}

/************************************************************************/
// UTF-32 is convenient because each code-point is a single 32-bit integer.
// It's typically std::wstring::value_type on Linux, but NOT Windows.
CODA_OSS_API std::u32string to_u32string(coda_oss::u8string::const_pointer, size_t);
CODA_OSS_API std::u32string to_u32string(str::W1252string::const_pointer, size_t);
inline auto to_u32string(const coda_oss::u8string& s)
{
    return to_u32string(s.c_str(), s.length());
}
inline auto to_u32string(const str::W1252string& s)
{
    return to_u32string(s.c_str(), s.length());
}

/************************************************************************/
// Windows-1252 (almost the same as ISO8859-1) is the default single-byte encoding on Windows.
CODA_OSS_API str::W1252string to_w1252string(coda_oss::u8string::const_pointer p, size_t sz);
inline auto to_w1252string(const coda_oss::u8string& s)
{
    return to_w1252string(s.c_str(), s.length());
}

/************************************************************************/

inline auto u8FromNative(const std::string& s)  // platform determines Windows-1252 or UTF-8 input
{
    #if _WIN32
    const auto p = str::c_str<str::W1252string>(s); // std::string is Windows-1252 on Windows
    #else
    const auto p = str::c_str<coda_oss::u8string>(s); // assume std::string is UTF-8 on any non-Windows platform
    #endif   
    return str::to_u8string(p, s.length());
}

namespace details
{
inline auto c_str(const std::wstring& s)
{
    #if _WIN32
    return str::c_str<std::u16string>(s); // std::wstring is UTF-16 on Windows
    #else
    return str::c_str<std::u32string>(s); // assume std::wstring is UTF-32 on any non-Windows platform
    #endif   
}
}
inline auto u8FromNative(const std::wstring& s) // platform determines UTF16 or UTF-32 input
{
    return str::to_u8string(details::c_str(s), s.length());
}

/************************************************************************/

// The resultant `std::string`s have "native" encoding (which is lost) depending
// on the platform: UTF-8 on Linux and Windows-1252 on Windows.
namespace details
{
  inline std::string to_string(const std::string& s)
  {
    return s;
  }
CODA_OSS_API std::string to_string(const coda_oss::u8string&);
CODA_OSS_API std::string to_string(const std::wstring&); // input is UTF-16 or UTF-32 depending on the platform
CODA_OSS_API std::wstring to_wstring(const std::string&); // platform determines Windows-1252 or UTF-8 input and output encoding
CODA_OSS_API std::wstring to_wstring(const coda_oss::u8string&); // platform determines UTF-16 or UTF-32 output encoding
}
namespace testing
{
CODA_OSS_API std::string to_string(const str::W1252string&);    
CODA_OSS_API std::wstring to_wstring(const str::W1252string&); // platform determines UTF-16 or UTF-32 output encoding
}

inline std::string to_native(const coda_oss::u8string& s) // cf., std::filesystem::native(), https://en.cppreference.com/w/cpp/filesystem/path/native
{
    return details::to_string(s);
}

}

#endif // CODA_OSS_str_Encoding_h_INCLUDED_
