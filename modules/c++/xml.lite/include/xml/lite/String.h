/* =========================================================================
 * This file is part of xml.lite-c++ 
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

#ifndef CODA_OSS_xml_lite_String_h_INCLLUDED_
#define CODA_OSS_xml_lite_String_h_INCLLUDED_
#pragma once

#include <string>
#include <ostream>

#include "str/Encoding.h"
#include "sys/String.h"
#include "sys/Optional.h"
#include "sys/CStdDef.h"
#include "mem/Span.h"
#include "xml/lite/QName.h" // StringEncoding

/*!
 * \file String.h
 * \brief A String that can be either UTF-8 or "native" 
 *
 * This isn't really part of XML, but it's a convenient place to put it (since
 * StringEncoding is already here).  The need for it also crops up often
 * with XML as those files must be encoded in UTF-8.
 * 
 * On Linux, there is good support for UTF-8, so a std::string encoded
 * as UTF-8 will display the "foreign" characters properly.  On Windows,
 * the preferred way to do that is by using UTF-16 (WCHAR, std::wstring),
 * but little (none?) of our existing code bases do that.  (On Linux, std::wstring
 * is typically UTF-32.)
 *
 */

namespace xml
{
namespace lite
{
class String final
{
    StringEncoding mEncoding;
    sys::Optional<std::string> mpString;
    sys::Optional<sys::U8string> mpU8String;

public:
    String();
    String(const std::string&, StringEncoding);
    //explicit String(const std::string&);
    explicit String(const char*); // really should know encoding if using std::string
    String(const sys::U8string&);
    String(const str::W1252string&);

    ~String() = default;
    String(const String&) = default;
    String& operator=(const String&) = default;
    String(String&&) = default;
    String& operator=(String&&) = default;

    size_t size() const;
    size_t length() const
    {
        return size();
    }

    StringEncoding encoding() const
    {
        return mEncoding;
    }

    std::string native() const;  // conversion might occur
    StringEncoding string(std::string&) const;  // conversion might occur
    sys::U8string u8string() const;  // conversion might occur
    StringEncoding u8string(std::string&) const;  // conversion might occur

    template <typename T>
    bool has() const;

    template<typename T>
    const T& cref() const;  // no conversion, might throw
    template <typename T>
    T& ref();  // no conversion, might throw
    template <typename T>
    const T& ref() const
    {
        return cref<T>();
    }

    // either mpString->c_str() or mpU8String->c_str()
    const char* c_str() const;
    const void* data() const // for I/O
    {
        return c_str();
    }
    mem::Span<const sys::Byte> bytes() const;

    friend bool operator_eq(const String& lhs, const String& rhs);
};

// GCC wants the specializations outside of the class
template <>
inline bool String::has<std::string>() const
{
    return mpString.has_value();
}
template <>
inline bool String::has<sys::U8string>() const
{
    return mpU8String.has_value();
}

template <>
inline const std::string& String::cref() const  // no conversion, might throw
{
    return *mpString;
}
template <>
inline std::string& String::ref()  // no conversion, might throw
{
    return *mpString;
}
template <>
inline const sys::U8string& String::cref() const  // no conversion, might throw
{
    return *mpU8String;
}
template <>
inline sys::U8string& String::ref()  // no conversion, might throw
{
    return *mpU8String;
}

inline bool operator==(const String& lhs, const String& rhs)
{
    return operator_eq(lhs, rhs);
}
inline bool operator!=(const String& lhs, const String& rhs)
{
    return !(lhs == rhs);
}

inline bool operator==(const String& lhs, const sys::U8string& rhs)
{
    return lhs == String(rhs); // might be able to avoid converting
}
inline bool operator==(const sys::U8string& lhs, const String& rhs)
{
    return rhs == lhs;
}
inline bool operator!=(const String& lhs, sys::U8string& rhs)
{
    return !(lhs == rhs);
}
inline bool operator!=(const sys::U8string& lhs, const String& rhs)
{
    return !(lhs == rhs);
}

// need to know encoding to use std::string
inline bool operator==(const String& lhs, const char* rhs) 
{
    return lhs == String(rhs);  // might be able to avoid converting
}
inline bool operator==(const char* lhs, const String& rhs)
{
    return rhs == lhs;
}
inline bool operator!=(const String& lhs, const char* rhs)
{
    return !(lhs == rhs);
}
inline bool operator!=(const char* lhs, const String& rhs)
{
    return !(lhs == rhs);
}


inline std::ostream& operator<<(std::ostream& os, const String& s)
{
    os << s.native();
    return os;
}

}
}
#endif  // CODA_OSS_xml_lite_String_h_INCLLUDED_
