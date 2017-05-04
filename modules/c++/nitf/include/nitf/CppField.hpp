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
#include <sys/Conf.h>
#include <nitf/DateTime.hpp>
#include <nitf/System.hpp>

/*!
 * File CppField.hpp
 * Contains pure C++ implemntation for Field
 */
namespace nitf
{

/*!
 * \class StringConverter
 * \brief Contains methods to convert arguments to Strings formatted for
 * NITF Fields
 */
class StringConverter
{
public:
    /*!
     * Construct a StringConverter
     * \param length How many character should the string be?
     * \param type NITF_BCS_A or NITF_BCS_N
     */
    StringConverter(size_t length, FieldType type) :
        mLength(length),
        mFieldType(type)
    {
    }

    //! Convert the value to a string of proper length
    std::string realToString(double value) const;
    std::string toNitfString(double value) const;
    std::string toNitfString(float value) const;
    std::string toNitfString(const DateTime& value) const;


    /*!
     * Convert the value to a string of proper length
     * \param value The value to convert
     * \param length How long should the string be? Defaults to mLength.
     * Intended for internal use.
     *  \return Formatted string
     */
    std::string toNitfString(const std::string& value, size_t length = 0) const;

    template<typename T>
    std::string toNitfString(const T& value, size_t length = 0) const
    {
        if (!std::numeric_limits<T>::is_integer)
        {
            throw except::Exception(Ctxt("Not implemented for this type!"));
        }
        if (length == 0)
        {
            length = mLength;
        }
        if (std::numeric_limits<T>::is_signed && value < 0)
        {
            return "-" + toNitfString(value * -1, length - 1);
        }
        std::string unformatted;
        if (sizeof(T) > 4)
        {
            unformatted = str::toString(value);
        }
        else
        {
            // toString doesn't work with 8- and 16- bit types for some reason
            // so have to cast up
            if (std::numeric_limits<T>::is_signed)
            {
                unformatted = str::toString(static_cast<Int32>(value));
            }
            else
            {
                unformatted = str::toString(static_cast<Uint32>(value));
            }
        }
        return toNitfString(unformatted, length);
    }

private:
    const size_t mLength;
    const FieldType mFieldType;
};

/*!
 * \class CppField
 * Abstract class representing a field in a NITF
 * \tparam T Type to be stored in the field
 * \tparam N Length of the field
 */
//TODO: It would probably be useful to give everything a string constructor
//So I just take the appropriate chunk from the file and shove it in
template<typename T, size_t N>
class CppField
{
public:
    /*!
     * Construct a Field
     * \param type NITF_BCS_A or NITF_BCS_N
     */
    CppField(FieldType type) :
        mConverter(N, type)
    {
    }

    virtual ~CppField()
    {
    }

    /*!
     * Assignment operator. Set field to store value
     * \param value The value to assign
     */
    CppField& operator=(T value)
    {
        mData = value;
        mRepresentation = toNitfString(value);
        return *this;
    }

    //! Retrieve the data stored in the field
    virtual operator T() const
    {
        return mData;
    }

    //! Return a string representation of length N
    inline std::string toString() const
    {
        return mRepresentation;
    }

    //! Retrieve the data stored in the field
    virtual T getValue() const
    {
        return mData;
    }

protected:
    T mData;
    std::string mRepresentation;
    inline std::string toNitfString(const T& value) const
    {
        return mConverter.toNitfString(value);
    }
    void set(const T& value)
    {
        this->mData = value;
        this->mRepresentation = this->toNitfString(value);
    }

private:
    const StringConverter mConverter;
};

/*!
 * \class BCSA
 * Class representing a BCS-A Field in a NITF
 * The range of allowable characters consists of space to tilde
 * The class does not enforce this, but rather assumes you are not going to do
 * something silly with it
 * \tparam T Type to be stored in the field
 * \tparam N Length of the field
 */
template<typename T, size_t N>
class BCSA : public CppField<T, N>
{
public:
    //! Construct a BCSA
    BCSA():
        CppField<T, N>(NITF_BCS_A)
    {
        this->mData = T();
        this->mRepresentation = std::string(N, ' ');
        //this->set(T());
    }

    //! Copy constructor
    BCSA(const T& value):
        CppField<T, N>(NITF_BCS_A)
    {
        this->set(value);
    }

    //! Copy constructor
    BCSA(const BCSA<T, N>& value):
        CppField<T, N>(NITF_BCS_A)
    {
        this->set(value);
    }

    //! Assignment operator
    BCSA& operator=(const T& value)
    {
        this->set(value);
        return *this;
    }

    //! Assignment operator
    BCSA& operator=(const BCSA<T, N>& value)
    {
        this->set(value);
        return *this;
    }
};

/*!
 * \class BCSA
 * Partial specialization of BCSA with T = DateTime
 * DateTimes have some specialize cases where the default value is a string
 * with N spaces, and that needs to be handled specially
 * \tparam N Length of the field. Should be 8 or 14
 */
template<size_t N>
class BCSA<DateTime, N> : public CppField<DateTime, N>
{
public:
    //! Construct a BCSA
    BCSA():
        CppField<DateTime, N>(NITF_BCS_A)
    {
        if (N != std::string("YYYYMMDD").size() &&
            N != std::string("YYYYMMDDhhmmss").size())
        {
            throw except::Exception(Ctxt("DateTime fields have lengths of "
                    "either 14 or 8. Got: " + str::toString(N)));
        }
        this->mRepresentation = std::string(N, ' ');
    }

    //! Copy constructor
    BCSA(const DateTime& value):
        CppField<DateTime, N>(NITF_BCS_A)
    {
        this->set(value);
    }

    //! Copy constructor
    BCSA(const BCSA<DateTime, N>& value):
        CppField<DateTime, N>(NITF_BCS_A)
    {
        this->set(value);
    }

    //! Assignment operator
    BCSA& operator=(const DateTime& value)
    {
        this->set(value);
        return *this;
    }

    //! Assignment operator
    BCSA& operator=(const BCSA<DateTime, N>& value)
    {
        this->set(value);
        return *this;
    }

    //! Throw if invalid
    virtual operator DateTime() const
    {
        if (this->mRepresentation == std::string(N, ' '))
        {
            throw except::Exception(Ctxt("Field uninitialized"));
        }
        return this->mData;
    }

    //! Retrieve the data stored in the field
    virtual DateTime getValue() const
    {
        if (this->mRepresentation == std::string(N, ' '))
        {
            throw except::Exception(Ctxt("Field uninitialized"));
        }
        return this->mData;
    }
};

/*!
 * \class CppField
 * Class representing a BCS-N Field in a NITF
 * The range of allowable characters consists of digits 0-9, +, and -
 * Fields such as "+09-03" are valid, for example, so it can make sense
 * to have T = std::string
 * \tparam T Type to be stored in the field
 * \tparam N Length of the field
 */
template<typename T, size_t N>
class BCSN : public CppField<T, N>
{
public:
    //! Construct a BCSN
    BCSN():
        CppField<T, N>(NITF_BCS_N)
    {
        this->set(T());
    }

    //! Copy constructor
    BCSN(const T& value):
        CppField<T, N>(NITF_BCS_N)
    {
        this->set(value);
    }

    //! Copy constructor
    BCSN(const BCSN<T, N>& value):
        CppField<T, N>(NITF_BCS_N)
    {
        this->set(value);
    }

    //! Assignment operator
    BCSN& operator=(const T& value)
    {
        this->set(value);
        return *this;
    }

    //! Assignment operator
    BCSN& operator=(const BCSN<T, N>& value)
    {
        this->set(value);
        return *this;
    }
};
}

#endif

