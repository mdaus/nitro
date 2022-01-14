/* =========================================================================
 * This file is part of str-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
 * (C) Copyright 2022, Maxar Technologies, Inc.
 *
 * xml.lite-c++ is free software; you can redistribute it and/or modify
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

#ifndef CODA_OSS_str_EncodedStringView_h_INCLLUDED_
#define CODA_OSS_str_EncodedStringView_h_INCLLUDED_
#pragma once

#include <string>
#include <ostream>
#include <memory>

#include "str/Encoding.h"

/*!
 * \file EncodedStringView.h
 * \brief A String that can be either UTF-8 or "native" 
 *
 * On Linux, there is good support for UTF-8, so a std::string encoded
 * as UTF-8 will display the "foreign" characters properly.  On Windows,
 * the preferred way to do that is by using UTF-16 (WCHAR, std::wstring),
 * but little (none?) of our existing code bases do that.  (On Linux, std::wstring
 * is typically UTF-32.)
 *
 */

namespace str
{
class EncodedStringView final
{
    struct Impl;
    std::unique_ptr<Impl> pImpl;

public:
    EncodedStringView();
    ~EncodedStringView();
    EncodedStringView(const EncodedStringView&);
    EncodedStringView& operator=(const EncodedStringView&);
    EncodedStringView(EncodedStringView&&);
    EncodedStringView& operator=(EncodedStringView&&);

    // Need these overloads to avoid creating temporary std::basic_string<> instances.
    // Routnes always return a copy, never a reference, so there's no additional overhead
    // with storing a raw pointer rather than a pointer to  std::basic_string<>.
    explicit EncodedStringView(sys::U8string::const_pointer);
    explicit EncodedStringView(str::W1252string::const_pointer);
    explicit EncodedStringView(std::string::const_pointer);  // Assume platform native encoding: UTF-8 on Linux, Windows-1252 on Windows

    explicit EncodedStringView(const sys::U8string&);
    explicit EncodedStringView(const str::W1252string&);
    explicit EncodedStringView(const std::string&);  // Assume platform native encoding: UTF-8 on Linux, Windows-1252 on Windows

    EncodedStringView& operator=(sys::U8string::const_pointer);
    EncodedStringView& operator=(str::W1252string::const_pointer);
    EncodedStringView& operator=(std::string::const_pointer);  // Assume platform native encoding: UTF-8 on Linux, Windows-1252 on Windows
    EncodedStringView& operator=(const sys::U8string&);
    EncodedStringView& operator=(const str::W1252string&);
    EncodedStringView& operator=(const std::string&);  // Assume platform native encoding: UTF-8 on Linux, Windows-1252 on Windows
    
    // Input is encoded as specified on all platforms.
    template <typename TBasicString>
    EncodedStringView& assign(const char* s)
    {
        using const_pointer = typename TBasicString::const_pointer;
        *this = str::cast<const_pointer>(s);
        return *this;
    }
    template <typename TBasicString>
    EncodedStringView& assign(const std::string& s)
    {
        return assign<TBasicString>(s.c_str());
    }

    // Input is encoded as specified on all platforms.
    template <typename TBasicString>
    static EncodedStringView create(const char* s)
    {
        using const_pointer = typename TBasicString::const_pointer;
        return EncodedStringView(str::cast<const_pointer>(s));
    }
    template <typename TBasicString>
    static EncodedStringView create(const std::string& s)
    {
        return create<TBasicString>(s.c_str());
    }

    // Regardless of what string we're looking at, return a string in platform
    // native encoding: UTF-8 on Linux, Windows-1252 on Windows; this
    // might result in string conversion.
    std::string native() const; // c.f. std::filesystem::path::native()

    // Convert (perhaps) whatever we're looking at to UTF-8
    sys::U8string to_u8string() const;
    std::string& toUtf8(std::string&) const; // std::string is encoded as UTF-8, always.

    // Only casting done, no conversion.  This should be OK as all three
    // string types are 8-bit encodings.
    //
    // Intentionally a bit of a mouth-full as these routines should be used sparingly.
    template <typename TConstPointer>
    TConstPointer cast() const;  // returns NULL if stored pointer not of the desired type

    bool operator_eq(const EncodedStringView&) const;

private:
    template <typename TReturn, typename T2, typename T3>
    typename TReturn::const_pointer cast_(const TReturn& retval, const T2& t2, const T3& t3) const;
};

inline bool operator==(const EncodedStringView& lhs, const EncodedStringView& rhs)
{
    return lhs.operator_eq(rhs);
}
inline bool operator!=(const EncodedStringView& lhs, const EncodedStringView& rhs)
{
    return !(lhs == rhs);
}

template<typename TChar>
inline bool operator==(const EncodedStringView& lhs, const TChar* rhs)
{
    return lhs == EncodedStringView(rhs);
}
template <typename TChar>
inline bool operator==(const TChar* lhs, const EncodedStringView& rhs)
{
    return rhs == lhs;
}
template <typename TChar>
inline bool operator!=(const EncodedStringView& lhs, const TChar* rhs)
{
    return !(lhs == rhs);
}
template <typename TChar>
inline bool operator!=(const TChar* lhs, const EncodedStringView& rhs)
{
    return !(lhs == rhs);
}

template <typename TChar>
inline bool operator==(const EncodedStringView& lhs, const std::basic_string<TChar>& rhs)
{
    return lhs == EncodedStringView(rhs);
}
template <typename TChar>
inline bool operator==(const std::basic_string<TChar>& lhs, const EncodedStringView& rhs)
{
    return rhs == lhs;
}
template <typename TChar>
inline bool operator!=(const EncodedStringView& lhs, const std::basic_string<TChar>& rhs)
{
    return !(lhs == rhs);
}
template <typename TChar>
inline bool operator!=(const std::basic_string<TChar>& lhs, const EncodedStringView& rhs)
{
    return !(lhs == rhs);
}

inline std::ostream& operator<<(std::ostream& os, const EncodedStringView& esv)
{
    os << esv.native();
    return os;
}

}
#endif  // CODA_OSS_str_EncodedStringView_h_INCLLUDED_
