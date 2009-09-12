/* =========================================================================
 * This file is part of except-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * except-c++ is free software; you can redistribute it and/or modify
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


#ifndef __EXCEPT_EXCEPTION_H__
#define __EXCEPT_EXCEPTION_H__

/*!
 * \file Exception.h
 * \brief Contains the classes to do with exception handling.
 *
 * A Throwable has two possible classifications (according to java, and for our
 * purposes, that is good enough): Errors and Exceptions.
 * This class deals with the latter.
 *
 */
#include <string>
#include <sstream>
#include "except/Throwable.h"

namespace except
{

/*!
 * \class Exception
 * \brief (typically non-fatal) throwable.
 *
 * This class is the base for all exceptions.
 */
class Exception : public Throwable
{
public:

    /*!
     * Constructor.
     */
    Exception() : Throwable()
    {}

    /*!
     * Constructor. Takes a Context
     * \param c The Context
     */
    Exception(const Context& c) : Throwable(c)
    {}

    /*!
     * Constructor. Takes an Throwable and a Context
     * \param t The Throwable
     * \param c The Context
     */
    Exception(const Throwable& t, const Context& c) : Throwable(t, c)
    {}

    /*!
     * Constructor.  Takes a message
     * \param message The message
     */
    Exception(const std::string& message) : Throwable(message)
    {}

    //! Destructor
    virtual ~Exception()
    {}
};

/*!
 * \class IOException
 * \brief Throwable related to IO problems.
 */
class IOException : public Exception
{
public:
    //! Default Constructor
    IOException()
    {}
    /*!
     *  User constructor.  Sets the exception context.
     *  \param c The exception context
     */
    IOException(const Context& c) : Exception(c)
    {}
    /*!
     *  User constructor.  Sets the exception message.
     *  \param message the exception message
     */
    IOException(const std::string& message) : Exception(message)
    {}

    /*!
     * User constructor. Takes an Throwable and a Context
     * \param t The Throwable
     * \param c The Context
     */
    IOException(const Throwable& t, const Context& c) : Exception(t, c)
    {}

    //!  Destructor
    virtual ~IOException()
    {}
};

/*!
 * \class BadCastException
 * \brief Exception for bad casting operations
 */
class BadCastException : public Exception
{
public:
    //! Default Constructor
    BadCastException()
    {}
    /*!
     *  User constructor.  Sets the exception context.
     *  \param c The exception context
     */
    BadCastException(const Context& c) : Exception(c)
    {}
    /*!
     *  User constructor.  Sets the exception message.
     *  \param message the exception message
     */
    BadCastException(const std::string& message) : Exception(message)
    {}

    /*!
     * User constructor. Takes an Throwable and a Context
     * \param t The Throwable
     * \param c The Context
     */
    BadCastException(const Throwable& t, const Context& c) : Exception(t, c)
    {}

    //!  Destructor
    virtual ~BadCastException()
    {}
};
/*!
 * \class InvalidFormatException
 * \brief Throwable related to an invalid file format.
 */
class InvalidFormatException : public Exception
{
public:
    //! Default Constructor
    InvalidFormatException()
    {}
    /*!
     *  User constructor.  Sets the exception context.
     *  \param c The exception context
     */
    InvalidFormatException(const Context& c) : Exception(c)
    {}
    /*!
     *  User constructor.  Sets the exception message.
     *  \param message the exception message
     */
    InvalidFormatException(const std::string& message) : Exception(message)
    {}

    /*!
     * User constructor. Takes an Throwable and a Context
     * \param t The Throwable
     * \param c The Context
     */
    InvalidFormatException(const Throwable& t, const Context& c) : Exception(t, c)
    {}

    //!  Destructor
    ~InvalidFormatException()
    {}
};

/*!
 * \class IndexOutOfRangeException
 * \brief Throwable related to an index being out of range.
 */
class IndexOutOfRangeException : public Exception
{
public:
    //! Default Constructor
    IndexOutOfRangeException()
    {}
    /*!
     *  User constructor.  Sets the exception message.
     *  \param where where the exception was thrown
     *  \param lbound the left bound of the ranged structure
     *  \param rbound the right bound of the ranged structure
     */
    IndexOutOfRangeException(int where, int lbound, int rbound)
    {
        std::ostringstream s;
        s << "Index out of range (" << lbound << " <= X < " << rbound << "): " << where;
        mMessage = s.str();
    }
    /*!
     *  User constructor.  Sets the exception context.
     *  \param c The exception context
     */
    IndexOutOfRangeException(const Context& c) : Exception(c)
    {}
    /*!
     *  User constructor.  Sets the exception message.
     *  \param message the exception message
     */
    IndexOutOfRangeException(const std::string& message) : Exception(message)
    {}

    /*!
     * User constructor. Takes an Throwable and a Context
     * \param t The Throwable
     * \param c The Context
     */
    IndexOutOfRangeException(const Throwable& t, const Context& c) : Exception(t, c)
    {}

    //!  Destructor
    ~IndexOutOfRangeException()
    {}
};

/*!
 * \class OutOfMemoryException
 * \brief Throwable related to memory allocation problems.
 */
class OutOfMemoryException : public Exception
{
public:
    //! Default Constructor
    OutOfMemoryException()
    {}
    /*!
     *  User constructor.  Sets the exception message.
     *  \param message the exception message
     */
    OutOfMemoryException(const std::string& message) : Exception(message)
    {}
    /*!
     *  User constructor.  Sets the exception context.
     *  \param c The exception context
     */
    OutOfMemoryException(const Context& c) : Exception(c)
    {}

    /*!
     * User constructor. Takes an Throwable and a Context
     * \param t The Throwable
     * \param c The Context
     */
    OutOfMemoryException(const Throwable& t, const Context& c) : Exception(t, c)
    {}

    //!  Destructor
    ~OutOfMemoryException()
    {}
};

/*!
 * \class FileNotFoundException
 * \brief Throwable related to a file not found.
 */
class FileNotFoundException : public Exception
{
public:
    //! Default Constructor
    FileNotFoundException()
    {}
    /*!
     *  User constructor.  Sets the exception context.
     *  \param c The exception context
     */
    FileNotFoundException(const Context& c) : Exception(c)
    {}
    /*!
     *  User constructor.  Sets the exception message.
     *  \param message the exception message
     */
    FileNotFoundException(const std::string& message) : Exception(message)
    {}

    /*!
     * User constructor. Takes an Throwable and a Context
     * \param t The Throwable
     * \param c The Context
     */
    FileNotFoundException(const Throwable& t, const Context& c) : Exception(t, c)
    {}

    //!  Destructor
    ~FileNotFoundException()
    {}
};

/*!
 * \class NullPointerReference
 * \brief This is responsible for handling a null pointer ref/deref
 * 
 * This class is currently treated as an exception, meaning that its
 * behavior is not necessarily fatal. 
 */
class NullPointerReference : public Exception
{
public:
    //!  Constructor
    NullPointerReference()
    {}

    /*!
     *  Construct from context
     *  \param c The exception context.
     */
    NullPointerReference(const Context& c) : Exception(c)
    {}

    /*!
     *  Construct from message
     *  \param message The exception message
     */
    NullPointerReference(const std::string& message) : Exception(message)
    {}

    /*!
     * User constructor. Takes an Throwable and a Context
     * \param t The Throwable
     * \param c The Context
     */
    NullPointerReference(const Throwable& t, const Context& c) : Exception(t, c)
    {}

    //!  Destructor
    ~NullPointerReference()
    {}
};
/*!
 * \class NoSuchKeyException
 * \brief Throwable related to unknown keys.
 */
class NoSuchKeyException : public Exception
{
public:
    //! Default Constructor
    NoSuchKeyException()
    {}
    /*!
     *  User constructor.  Sets the exception context.
     *  \param c The exception context
     */
    NoSuchKeyException(const Context& c) : Exception(c)
    {}
    /*!
     *  User constructor.  Sets the exception message.
     *  \param message the exception message
     */
    NoSuchKeyException(const std::string& message) : Exception(message)
    {}

    /*!
     * User constructor. Takes an Throwable and a Context
     * \param t The Throwable
     * \param c The Context
     */
    NoSuchKeyException(const Throwable& t, const Context& c) : Exception(t, c)
    {}

    //!  Destructor
    ~NoSuchKeyException()
    {}
};

/*!
 * \class NoSuchReferenceException
 * \brief Throwable related to unknown references.
 */
class NoSuchReferenceException : public Exception
{
public:
    //! Default Constructor
    NoSuchReferenceException()
    {}
    /*!
     *  User constructor.  Sets the exception context.
     *  \param c The exception context
     */
    NoSuchReferenceException(const Context& c) : Exception(c)
    {}
    /*!
     *  User constructor.  Sets the exception message.
     *  \param message the exception message
     */
    NoSuchReferenceException(const std::string& message) : Exception(message)
    {}

    /*!
     * User constructor. Takes an Throwable and a Context
     * \param t The Throwable
     * \param c The Context
     */
    NoSuchReferenceException(const Throwable& t, const Context& c) : Exception(t, c)
    {}

    //!  Destructor
    ~NoSuchReferenceException()
    {}
};

/*!
 * \class KeyAlreadyExistsException
 * \brief Throwable related to duplicate keys.
 */
class KeyAlreadyExistsException : public Exception
{
public:
    //! Default Constructor
    KeyAlreadyExistsException()
    {}
    /*!
     *  User constructor.  Sets the exception context.
     *  \param c The exception context
     */
    KeyAlreadyExistsException(const Context& c) : Exception(c)
    {}
    /*!
     *  User constructor.  Sets the exception message.
     *  \param message the exception message
     */
    KeyAlreadyExistsException(const std::string& message) : Exception(message)
    {}

    /*!
     * User constructor. Takes an Throwable and a Context
     * \param t The Throwable
     * \param c The Context
     */
    KeyAlreadyExistsException(const Throwable& t, const Context& c) : Exception(t, c)
    {}

    //!  Destructor
    ~KeyAlreadyExistsException()
    {}
};


/*!
 * \class NotImplementedException
 * \brief Throwable related to code not being implemented yet.
 */
class NotImplementedException : public Exception
{
public:
    NotImplementedException()
    {}
    /*!
     *  User constructor.  Sets the exception context.
     *  \param c The exception context
     */
    NotImplementedException(const Context& c) : Exception(c)
    {}
    /*!
     *  User constructor.  Sets the exception message.
     *  \param message the exception message
     */
    NotImplementedException(const std::string& message) : Exception(message)
    {}

    /*!
     * User constructor. Takes an Throwable and a Context
     * \param t The Throwable
     * \param c The Context
     */
    NotImplementedException(const Throwable& t, const Context& c) : Exception(t, c)
    {}

    //!  Destructor
    ~NotImplementedException()
    {}
};
}

#endif

