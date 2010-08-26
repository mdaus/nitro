/* =========================================================================
 * This file is part of xstr-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * xstr-c++ is free software; you can redistribute it and/or modify
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
#include "xstr/String.h"
#include <import/except.h>
#include <algorithm>

using namespace xstr;


char String::charAt(size_t index)
{
    return (*this)[index];
}

char String::operator[](size_t index) const
{
    if (index >= length())
        throw except::IndexOutOfRangeException(String::format(
                        "index out of bounds: %d", index).mString);
    return mString[index];
}

String String::format(const char *format, ...)
{
    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsprintf(buffer, format, args);
    va_end(args);
    return buffer;
}

std::ostream& operator<<(std::ostream& os, const String& s)
{
    os << s.str();
    return os;
}

bool String::operator==(const String& s) const
{
    return mString == s.mString;
}

const char* String::toCharArray() const
{
    return mString.c_str();
}

String String::substring(size_t beginIndex, size_t endIndex) const
{
    if (endIndex >= length())
        endIndex = String::npos;
 
    if (endIndex < beginIndex)
        throw except::IndexOutOfRangeException(String::format(
                        "substring indices out of bounds: %d, %d",
                        beginIndex, endIndex).mString);

    return mString.substr(beginIndex, endIndex != String::npos ? endIndex
            - beginIndex : endIndex);
}

String String::toLowerCase() const
{
    std::string s = mString;
    std::transform(s.begin(), s.end(), s.begin(), (int(*)(int)) tolower);
    return s;
}

String String::toUpperCase() const
{
    std::string s = mString;
    std::transform(s.begin(), s.end(), s.begin(), (int(*)(int)) toupper);
    return s;
}

String String::trim() const
{
    std::string s = mString;
    size_t i;
    for (i = 0; i < s.length(); i++)
        if (!isspace(s[i]))
            break;
    s.erase(0, i);

    for (i = s.length() - 1; (int) i >= 0; i--)
        if (!isspace(s[i]))
            break;
    if (i + 1 < s.length())
        s.erase(i + 1);
    return s;
}

std::string String::str() const
{
    return mString;
}

bool String::startsWith(const String& s) const
{
    int sLen = s.length();
    int len = length();
    for (int i = 0; i < len && i < sLen; ++i)
        if (!(mString[i] == s[i]))
            return false;
    return len >= sLen;
}

bool String::endsWith(const String& s) const
{
    size_t sLen = s.length();
    size_t len = length();
    for (size_t i = 0; i < len && i < sLen; ++i)
        if (!(mString[len - i - 1] == s[sLen - i - 1]))
            return false;
    return len >= sLen;
}


std::vector<String> String::split(String pattern) const
{
    re::PCRE expr;
    expr.compile(pattern.mString);
    std::vector<std::string> parts;
    expr.split(mString, parts);
    std::vector<String> sParts(parts.size());
    for(size_t i = 0, len = parts.size(); i < len; ++i)
        sParts[i] = parts[i];
    return sParts;
}

bool String::matches(const String& pattern) const
{
    re::PCRE expr;
    expr.compile(pattern.mString);
    return expr.matches(mString);
}

size_t String::indexOf(const String& s, size_t fromIndex) const
{
    return mString.find(s.mString, fromIndex);
}

size_t String::lastIndexOf(const String& s, size_t fromIndex) const
{
    return mString.rfind(s.mString,
            fromIndex != String::npos ? fromIndex : length() - 1);
}

bool String::contains(const String& s) const
{
    return indexOf(s) != String::npos;
}

template <> std::string String::toType<std::string>()
{
    return mString;
}

template <> String String::toType<String>()
{
    return *this;
}
