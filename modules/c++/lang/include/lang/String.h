/* =========================================================================
 * This file is part of lang-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2010, General Dynamics - Advanced Information Systems
 *
 * lang-c++ is free software; you can redistribute it and/or modify
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

#ifndef __LANG_STRING_H__
#define __LANG_STRING_H__

#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include <import/sys.h>
#include <import/re.h>

namespace lang
{

/*!
 * \brief String class
 *
 * This class wraps a std::string object and provides a similar API to that of
 * the Java String class.
 */
class String
{
public:

    static const size_t npos;

    String() :
        mString("")
    {
    }

    String(const char* s) :
        mString(s)
    {
    }

    String(char c)
    {
        *this = String::valueOf(c);
    }

    String(const std::string& s) :
        mString(s)
    {
    }

    String(const String& s) :
        mString(s.toCharArray())
    {
    }

    ~String()
    {
    }

    typedef std::string::iterator iterator;
    iterator begin() { return mString.begin(); }
    iterator end() { return mString.end(); }

    /*!
     * Returns the char value at the specified index. An index ranges from 0 to
     * length() - 1. The first char value of the sequence is at index 0, the
     * next at index 1, and so on, as for array indexing.
     */
    char charAt(size_t index);

    char operator[](size_t index) const;

    /*!
     * Returns the length of this string.
     */
    size_t length() const
    {
        return mString.size();
    }

    template <typename T> String & operator=(const T& value)
    {
        return (*this = String::valueOf(value));
    }

    /*!
     * Equality check
     */
    bool operator==(const String&) const;

    /*!
     * Returns the char array representation of the String
     */
    const char* toCharArray() const;

    /*!
     * Returns a substring of the String
     * \param beginIndex
     * \param endIndex
     * \return String substring, or throws an IndexOutOfRangeException
     */
    String substring(size_t beginIndex, size_t endIndex = String::npos) const;

    /*!
     * Returns a copy of the String, converted to lower case
     */
    String toLowerCase() const;

    /*!
     * Returns a copy of the String, converted to upper case
     */
    String toUpperCase() const;

    /*!
     * Returns a copy of the String with leading and trailing white spaces
     * removed
     */
    String trim() const;

    /*!
     * Returns the std::string representation
     * TODO - might not need this...
     */
    std::string str() const;

    /*!
     * Returns true if the String starts with the passed-in String
     */
    bool startsWith(const String& s) const;

    /*!
     * Returns true if the String ends with the passed-in String
     */
    bool endsWith(const String&) const;

    /*!
     * Split the String by occurrences of the pattern
     * \param pattern
     *          The regex pattern to use
     */
    std::vector<String> split(String pattern = " ") const;

    /*!
     * Returns a boolean denoting whether the String matches the given
     * regex pattern
     * \param pattern
     *          The regex pattern to match against
     */
    bool matches(const String& pattern) const;

    size_t indexOf(const String& s, size_t fromIndex = 0) const;

    size_t lastIndexOf(const String& s, size_t fromIndex = String::npos) const;

    bool contains(const String& s) const;

    /*!
     * Attempts to return a representation of the internal String as another
     * type.
     */
    template <typename T> T toType()
    {
        if (mString.empty())
            return T();

        std::istringstream buf(mString);
        T value;
        buf >> value;

        if (buf.fail())
        {
            throw except::BadCastException(std::string("std::string") +
                    buf.str() + typeid(T).name());
        }
        return value;
    }

    /*!
     * Converts the passed in type to a String
     */
    template <typename T> static String valueOf(const T& value)
    {
        std::ostringstream buf;
        buf.precision(str::setPrecision(value));
        buf << std::boolalpha << value;
        return buf.str();
    }

    /*!
     * Varargs formatting of the given message, much like a C-style printf
     * statement.
     */
    static String format(const char *format, ...);

protected:
    std::string mString;
};

template <> String String::toType<String>();
template <> std::string String::toType<std::string>();

}

std::ostream& operator<<(std::ostream& os, const lang::String& s);

#endif
