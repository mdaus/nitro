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

#include "xml/lite/String.h"

#include <assert.h>

#include <stdexcept>

#include "sys/OS.h"
#include "str/Convert.h"

constexpr auto PlatformEncoding = sys::Platform == sys::PlatformType::Windows
        ? xml::lite::StringEncoding::Windows1252
        : xml::lite::StringEncoding::Utf8;

xml::lite::String::String(const std::string& s, StringEncoding encoding) :
    mpString(s), mEncoding(encoding)
{
}
xml::lite::String::String(const char* s) : String(s, PlatformEncoding) // need to know encoding to use std::string
{
}
xml::lite::String::String() : String("")
{
}
xml::lite::String::String(const sys::U8string& s) :
    mpU8String(s), mEncoding(xml::lite::StringEncoding::Utf8)
{
}
xml::lite::String::String(const str::W1252string& s) :
    String(str::c_str<std::string::const_pointer>(s), xml::lite::StringEncoding::Windows1252)
{
}

static std::string to_native(const std::string& s, xml::lite::StringEncoding encoding)
{
    if (encoding == PlatformEncoding)
    {
        return s; // already in the platform's native encoding
    }

     if (encoding == xml::lite::StringEncoding::Utf8)
    {
        return str::toString(s); // UTF-8 to Windows-1252
    }

    if (encoding == xml::lite::StringEncoding::Windows1252)
    {
        const auto utf8 = str::fromWindows1252(s); // Windows-1252 to UTF-8
        return str::c_str<std::string::const_pointer>(utf8); // copy
    }

    throw std::logic_error("Unknown encoding.");
}

static std::string to_native(const sys::U8string& s)
{
    return PlatformEncoding == xml::lite::StringEncoding::Utf8
            ? str::c_str<std::string::const_pointer>(s)
            : str::toString(s);
}

std::string xml::lite::String::native() const
{
    if (mpString.has_value())
    {
        assert(!mpU8String.has_value());
        return to_native(*mpString, mEncoding);
    }
    
    assert(mpU8String.has_value());
    assert(mEncoding == xml::lite::StringEncoding::Utf8);
    return to_native(*mpU8String);
}

xml::lite::StringEncoding xml::lite::String::string(std::string& result) const
{
    if (mpString.has_value())
    {
        assert(!mpU8String.has_value());
        result = *mpString;
    }
    else
    {
        assert(mEncoding == xml::lite::StringEncoding::Utf8);
        result = str::toString(*mpU8String);
    }
    return mEncoding;
}
sys::U8string xml::lite::String::u8string() const
{
    if (mpU8String.has_value())
    {
        assert(!mpString.has_value());
        assert(mEncoding == xml::lite::StringEncoding::Utf8);
        return *mpU8String;
    }

    if (mEncoding == xml::lite::StringEncoding::Utf8)
    {
        return str::fromUtf8(*mpString);
    }
    if (mEncoding == xml::lite::StringEncoding::Windows1252)
    {
        return str::fromWindows1252(*mpString);
    }

    throw std::logic_error("Unknown encoding.");
}
xml::lite::StringEncoding xml::lite::String::u8string(std::string& result) const
{
    if (mpString.has_value())
    {
        if (mEncoding == xml::lite::StringEncoding::Utf8)
        {
            result = *mpString;
        }
        else if (mEncoding == xml::lite::StringEncoding::Windows1252)
        {
            result = str::c_str<std::string::const_pointer>(str::fromWindows1252(*mpString));
        }
        else
        {
            throw std::logic_error("Unknown encoding.");
        }
    }
    else
    {
        assert(mpU8String.has_value());
        assert(mEncoding == xml::lite::StringEncoding::Utf8);
        result = str::c_str<std::string::const_pointer>(*mpU8String);   
    }

    return mEncoding;
}

bool xml::lite::operator_eq(const String& lhs, const String& rhs)
{
    if (lhs.mpU8String.has_value())
    {
        assert(!lhs.mpString.has_value());
        assert(lhs.mEncoding == xml::lite::StringEncoding::Utf8); 

        auto& str = *lhs.mpU8String;
        if (rhs.mpU8String.has_value())  // both sys::U8string
        {
            assert(!rhs.mpString.has_value());
            assert(rhs.mEncoding == xml::lite::StringEncoding::Utf8);
            return str == (*rhs.mpU8String);
        }

        return str == rhs.u8string();  // convert RHS to UTF-8 and compare
    }

    // prefer to use UTF-8 if available
    if (rhs.mpU8String.has_value())
    {
        assert(!rhs.mpString.has_value());
        assert(rhs.mEncoding == xml::lite::StringEncoding::Utf8); 
        assert(!lhs.mpU8String.has_value()); // already checked, above
        return rhs == lhs; // recurse, swapping LHS and RHS
    }

    assert(lhs.mpString.has_value()); // lhs.mpU8String.has_value() == false, above
    assert(rhs.mpString.has_value());  // rhs.mpU8String.has_value() == false, above
    auto& str = *lhs.mpString;
    if (lhs.mEncoding == rhs.mEncoding)
    {
        // both std::string values are encoded the same, just compare
        return str == (*rhs.mpString); // just compare std::string
    }
    if (lhs.mEncoding == xml::lite::StringEncoding::Utf8)
    {
        assert(rhs.mEncoding != xml::lite::StringEncoding::Utf8); // would have compared ==, above
        std::string rhs_;
        rhs.u8string(rhs_); // get RHS as UTF-8
        return str == rhs_;
    }
    if (rhs.mEncoding == xml::lite::StringEncoding::Utf8)
    {
        return rhs == lhs;  // recurse, swapping LHS and RHS    
    }

    assert(false); // This "can't happen" since there are only two possibilities for xml::lite::StringEncoding
    return lhs.native() == rhs.native();
}

size_t xml::lite::String::size() const
{
    return mpString.has_value() ? mpString->size() : mpU8String->size();
}
const char* xml::lite::String::c_str() const
{
    return mpString.has_value() ? mpString->c_str() : str::c_str<const char*>(*mpU8String);
}
mem::Span<const sys::Byte> xml::lite::String::bytes() const
{
    auto pBytes = static_cast<const sys::Byte*>(data());
    return mem::Span<const sys::Byte>(pBytes, size());
}
