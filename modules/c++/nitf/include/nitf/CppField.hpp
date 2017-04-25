/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2004 - 2017, MDA Information Systems LLC
 *
 * NITRO is free software; you can redistribute it and/or modify
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
 * License along with this program; if not, If not,
 * see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef __NITF_CPPFIELD_HPP__
#define __NITF_CPPFIELD_HPP__

#include <cstdio>
#include <string>
#include <str/Convert.h>
#include <nitf/DateTime.hpp>
#include <nitf/System.hpp>

/*!
 * File CppField.hpp
 * Contains pure C++ implemntation for Field
 */
namespace nitf
{

class AlphaNumericFieldImpl
{
public:
    AlphaNumericFieldImpl(size_t size);
    std::string toString() const
    {
        return mData;
    }

    void set(const std::string& value);
    template<typename T> T getInteger() const
    {
        if (std::numeric_limits<T>::is_signed)
        {
            return static_cast<T>(str::toType<nitf::Int64>(mData));
        }
        return static_cast<T>(str::toType<nitf::Uint64>(mData));
    }

    template <typename T> T getReal() const
    {
        return str::toType<T>(mData);
    }
    void setReal(double value);
    void setDateTime(const nitf::DateTime& value);
    nitf::DateTime asDateTime(
            const std::string& format=NITF_DATE_FORMAT_21) const;

private:
    std::string mData;
    const size_t mSize;
};

template <size_t N>
class CppField
{
public:
    virtual ~CppField()
    {
    }
    virtual std::string toString() const = 0;
    virtual FieldType getType() const = 0;
};

template <size_t N>
class AlphaNumericField : public CppField<N>
{
public:
    AlphaNumericField() :
        mImpl(N)
    {
    }

    virtual ~AlphaNumericField()
    {
    }
    inline virtual FieldType getType() const
    {
        return NITF_BCS_A;
    }
    inline virtual size_t getLength() const
    {
        return N;
    }
    inline virtual std::string toString() const
    {
        return mImpl.toString();
    }

    nitf::DateTime asDateTime(
            const std::string& format=NITF_DATE_FORMAT_21) const
    {
        return mImpl.asDateTime();
    }
    AlphaNumericField& operator=(const std::string& value)
    {
        mImpl.set(value);
        return *this;
    }
    AlphaNumericField& operator=(const char* value)
    {
        mImpl.set(std::string(value));
        return *this;
    }
    AlphaNumericField& operator=(nitf::Int64 value)
    {
        mImpl.set(str::toString<nitf::Int64>(value));
        return *this;
    }
    AlphaNumericField& operator=(nitf::Uint64 value)
    {
        mImpl.set(str::toString<nitf::Uint64>(value));
        return *this;
    }
    AlphaNumericField& operator=(float value)
    {
        mImpl.setReal(value);
        return *this;
    }
    AlphaNumericField& operator=(double value)
    {
        mImpl.setReal(value);
        return *this;
    }
    AlphaNumericField& operator=(const nitf::DateTime& value)
    {
        mImpl.setDateTime(value);
        return *this;
    }
    template <size_t T>
    AlphaNumericField& operator=(const nitf::CppField<T>& field)
    {
        mImpl.set(field.toString());
        return *this;
    }
    template<typename T>
    AlphaNumericField& operator=(T value)
    {
        if (std::numeric_limits<T>::is_signed)
        {
            mImpl.set(str::toString<nitf::Int32>(value));
        }
        else
        {
            mImpl.set(str::toString<nitf::Uint32>(value));
        }
        return *this;
    }
    operator nitf::Int8() const
    {
        return mImpl.getInteger<nitf::Int8>();
    }
    operator nitf::Int16() const
    {
        return mImpl.getInteger<nitf::Int16>();
    }
    operator nitf::Int32() const
    {
        return mImpl.getInteger<nitf::Int32>();
    }
    operator nitf::Int64() const
    {
        return mImpl.getInteger<nitf::Int64>();
    }
    operator nitf::Uint8() const
    {
        return mImpl.getInteger<nitf::Uint8>();
    }
    operator nitf::Uint16() const
    {
        return mImpl.getInteger<nitf::Uint16>();
    }
    operator nitf::Uint32() const
    {
        return mImpl.getInteger<nitf::Uint32>();
    }
    operator nitf::Uint64() const
    {
        return mImpl.getInteger<nitf::Uint64>();
    }
    operator float() const
    {
        return mImpl.getReal<float>();
    }
    operator double() const
    {
        return mImpl.getReal<double>();
    }
    operator std::string() const
    {
        return toString();
    }
private:
    AlphaNumericFieldImpl mImpl;
};

}

#endif

