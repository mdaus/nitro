/* =========================================================================
 * This file is part of str-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
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

#ifndef __STR_CONVERT_H__
#define __STR_CONVERT_H__
#pragma once

#include <import/except.h>
#include <cerrno>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <typeinfo>

// This is a fairly low-level file, so don't #include a lot of our other files
#include "str/String_.h"

namespace str
{
template <typename T>
int getPrecision(const T& type);

template <typename T>
int getPrecision(const std::complex<T>& type);

template <typename T>
std::string toString(const T& value)
{
    std::ostringstream buf;
    buf.precision(getPrecision(value));
    buf << std::boolalpha << value;
    return buf.str();
}

template <>
std::string toString(const uint8_t& value);

template <>
std::string toString(const int8_t& value);

template <>
inline std::string toString(const std::nullptr_t&)
{
    return "<nullptr>";
}

template <typename T>
std::string toString(const T& real, const T& imag)
{
    return toString(std::complex<T>(real, imag));
}

template <typename TReturn, typename TChar>
inline TReturn cast(const TChar* s)
{
    // This is OK as UTF-8 can be stored in std::string
    // Note that casting between the string types will CRASH on some
    // implementatons. NO: reinterpret_cast<const std::string&>(value)
    const void* const pStr = s;
    auto const retval = static_cast<TReturn>(pStr);
    static_assert(sizeof(*retval) == sizeof(*s), "sizeof(*TReturn) != sizeof(*TChar)"); 
    return retval;
}
template <typename TReturn, typename TChar>
inline TReturn c_str(const std::basic_string<TChar>& s)
{
    return cast<TReturn>(s.c_str());
}

template <>
inline std::string toString(const sys::U8string& value)
{
    return c_str<std::string::const_pointer>(value);  // copy
}

// This is to make it difficult to get encodings mixed up; it's here (in a .h
// file) as we want to unit-test it. Windows1252_T for Windows-1252 characters
enum class Windows1252_T : unsigned char { };  // https://en.cppreference.com/w/cpp/language/types
using W1252string = std::basic_string<Windows1252_T>;  // https://en.cppreference.com/w/cpp/string
template <>
inline std::string toString(const W1252string& value)
{
    return c_str<std::string::const_pointer>(value);  // copy
}

void windows1252to8(W1252string::const_pointer, size_t, sys::U8string&); // c.f. utf16to8

void utf16to1252(std::u16string::const_pointer, size_t, W1252string&);
void utf32to1252(std::u32string::const_pointer, size_t, W1252string&);
void wsto1252(std::wstring::const_pointer, size_t, W1252string&);

// assume std::string is Windows-1252 **ON ALL PLATFORMS**
sys::U8string fromWindows1252(std::string::const_pointer, size_t);
inline sys::U8string fromWindows1252(const std::string& s)
{
    return fromWindows1252(s.c_str(), s.size());
}

//////////////////////////////////////////////////////////////////////////////////////////
// These use utf8:: routines; see utf8.h
void utf16to8(std::u16string::const_pointer, size_t, sys::U8string&);
void utf32to8(std::u32string::const_pointer, size_t, sys::U8string&);
void wsto8(std::wstring::const_pointer, size_t, sys::U8string&);
inline void strto8(std::u16string::const_pointer p, size_t sz, sys::U8string& result)
{
    utf16to8(p, sz, result);
}
inline void strto8(std::u32string::const_pointer p, size_t sz, sys::U8string& result)
{
    utf32to8(p, sz, result);
}
inline void strto8(std::wstring::const_pointer p, size_t sz, sys::U8string& result)
{
    wsto8(p, sz, result);
}

inline void utf16to8(const std::u16string& s, sys::U8string& result)
{
    utf16to8(s.c_str(), s.size(), result);
}
inline void utf32to8(const std::u32string& s, sys::U8string& result)
{
    utf32to8(s.c_str(), s.size(), result);
}
inline void wsto8(const std::wstring& s, sys::U8string& result)
{
    wsto8(s.c_str(), s.size(), result);
}
inline void strto8(const std::u16string& s, sys::U8string& result)
{
    utf16to8(s, result);
}
inline void strto8(const std::u32string& s, sys::U8string& result)
{
    utf32to8(s, result);
}
inline void strto8(const std::wstring& s, sys::U8string& result)
{
    wsto8(s, result);
}

//////////////////////////////////////////////////////////////////////////////////////////

bool mbsrtowcs(sys::U8string::const_pointer, size_t, std::wstring&);
bool mbsrtowcs(const sys::U8string&, std::wstring&);
bool wcsrtombs(std::wstring::const_pointer, size_t, sys::U8string&);
bool wcsrtombs(const std::wstring&, sys::U8string&);

// When the encoding is important, we want to "traffic" in either std::wstring (UTF-16 or UTF-32) or sys::U8string (UTF-8),
// not str::W1252string (Windows-1252) or std::string (unknown).  Make it easy to get those from other encodings.
std::wstring to_wstring(std::string::const_pointer, size_t); // assume Windows-1252 or UTF-8  based on platform
std::wstring to_wstring(sys::U8string::const_pointer, size_t);
std::wstring to_wstring(str::W1252string::const_pointer, size_t);
template<typename TChar>
inline std::wstring to_wstring(const std::basic_string<TChar>& s)
{
    return to_wstring(s.c_str(), s.size());
}

sys::U8string to_u8string(std::string::const_pointer, size_t);  // assume Windows-1252 or UTF-8  based on platform
sys::U8string to_u8string(std::wstring::const_pointer, size_t);
sys::U8string to_u8string(str::W1252string::const_pointer, size_t);
template <typename TChar>
inline sys::U8string to_u8string(const std::basic_string<TChar>& s)
{
    return to_u8string(s.c_str(), s.size());
}

// Encoding information is lost, will be UTF-8 or Windows-1252 or UTF-8 based on platform
std::string to_string(const std::wstring&);

template <typename T>
T toType(const std::string& s)
{
    if (s.empty())
        throw except::BadCastException(
                except::Context(__FILE__,
                                __LINE__,
                                std::string(""),
                                std::string(""),
                                std::string("Empty string")));

    T value;

    std::stringstream buf(s);
    buf.precision(str::getPrecision(value));
    buf >> value;

    if (buf.fail())
    {
        throw except::BadCastException(
                except::Context(__FILE__,
                                __LINE__,
                                std::string(""),
                                std::string(""),
                                std::string("Conversion failed: '") + s +
                                        std::string("' -> ") +
                                        typeid(T).name()));
    }

    return value;
}

template <>
bool toType<bool>(const std::string& s);
template <>
std::string toType<std::string>(const std::string& s);

/**
 *  strtoll wrapper for msvc compatibility.
 */
long long strtoll(const char* str, char** endptr, int base);
/**
 *  strtoull wrapper for msvc compatibility.
 */
unsigned long long strtoull(const char* str, char** endptr, int base);

/**
 *  Convert a string containing a number in any base to a numerical type.
 *
 *  @param s a string containing a number in base base
 *  @param base the base of the number in s
 *  @return a numberical representation of the number
 *  @throw BadCastException thrown if cast cannot be performed.
 */
template <typename T>
T toType(const std::string& s, int base)
{
    char* end;
    errno = 0;
    const char* str = s.c_str();

    T res;
    bool overflow = false;
    if (std::numeric_limits<T>::is_signed)
    {
        const long long longRes = str::strtoll(str, &end, base);
        if (longRes < static_cast<long long>(std::numeric_limits<T>::min()) ||
            longRes > static_cast<long long>(std::numeric_limits<T>::max()))
        {
            overflow = true;
        }
        res = static_cast<T>(longRes);
    }
    else
    {
        const unsigned long long longRes = str::strtoull(str, &end, base);
        if (longRes < static_cast<unsigned long long>(
                              std::numeric_limits<T>::min()) ||
            longRes > static_cast<unsigned long long>(
                              std::numeric_limits<T>::max()))
        {
            overflow = true;
        }
        res = static_cast<T>(longRes);
    }

    if (overflow || errno == ERANGE)
        throw except::BadCastException(
                except::Context(__FILE__,
                                __LINE__,
                                std::string(""),
                                std::string(""),
                                std::string("Overflow: '") + s +
                                        std::string("' -> ") +
                                        typeid(T).name()));
    // If the end pointer is at the start of the string, we didn't convert
    // anything.
    else if (end == str)
        throw except::BadCastException(
                except::Context(__FILE__,
                                __LINE__,
                                std::string(""),
                                std::string(""),
                                std::string("Conversion failed: '") + s +
                                        std::string("' -> ") +
                                        typeid(T).name()));

    return res;
}

/**
 *  Determine the precision required for the data type.
 *
 *  @param type A variable of the type whose precision argument is desired.
 *  @return The integer argument required by ios::precision() to represent
 *  this type.
 */
template <typename T>
int getPrecision(const T&)
{
    return 0;
}

template <typename T>
int getPrecision(const std::complex<T>& type)
{
    return getPrecision(type.real());
}

template <>
int getPrecision(const float& type);

template <>
int getPrecision(const double& type);

template <>
int getPrecision(const long double& type);

/** Generic casting routine; used by explicitly overloaded
 conversion operators.

 @param value A variable of the type being cast to.
 @return The internal representation of GenericType, converted
 to the desired type, if possible.
 @throw BadCastException thrown if cast cannot be performed.
 */
template <typename T>
T generic_cast(const std::string& value)
{
    return str::toType<T>(value);
}

// Some string conversion routines only work with the right locale; the default
// is platform-dependent.
class setlocale final
{
    char* const locale_;
public:
    setlocale(const char* locale = "en_US.utf8");
    ~setlocale() noexcept(false);
};

}

#endif
