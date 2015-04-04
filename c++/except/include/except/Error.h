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


#ifndef __XPC_EXCEPT_ERROR_H__
#define __XPC_EXCEPT_ERROR_H__

/*!
 *  \file Error.h
 *  \brief Contains the classes to do with error handling
 *
 *  An error is a throwable pertaining to rather serious errors
 *  i.e., ones that could cripple the system if handled improperly.
 *  For that reason, it is worth considering, upon designing a handler
 *  whether or not to simply abort the program upon receiving an error.
 */
#include "Throwable.h"


namespace except
{
/*!
 * \class Error
 * \brief Represents a serious unexpected occurrence during the program
 *
 * The error class is a representation of a throwable object, occurring
 * under serious conditions.  It may be undesirable to handle an error in
 * the manner that you handle an exception.  For this reason, the distinction is
 * made
 */
class Error : public Throwable
{
public:
    /*!
     * Default constructor
     */
    Error() : Throwable()
    {}

    /*!
     * Constructor. Takes a Context
     * \param c The Context
     */
    Error(const Context& c) : Throwable(c)
    {}

    /*!
     * Constructor.  Takes a message
     * \param message The message
     */
    Error(const std::string& message) : Throwable(message)
    {}

    /*!
     * Constructor. Takes an Throwable and a Context
     * \param t The Throwable
     * \param c The Context
     */
    Error(const Throwable& t, const Context& c) : Throwable(t, c)
    {}

    /*\!
     * Get the type id
     * \return The type
     */ 
    //     virtual const char* getType() const { return __XpcTHISTYPENAME(); }
    //!  Destructor
    virtual ~Error()
    {}
}
;

/*!
 * \class InvalidDerivedTypeError
 * \brief Represents an invalid derived type error.
 */
class InvalidDerivedTypeError : public Error
{
public:
    //! Default constructor
    InvalidDerivedTypeError()
    {}

    /*!
     *  User constructor
     *  \param c the error context
     */
    InvalidDerivedTypeError(const Context& c) : Error(c)
    {}

    /*!
     *  User constructor
     *  \param message the error message
     */
    InvalidDerivedTypeError(const std::string& message) :
            Error(message)
    {}

    /*!
     * User constructor. Takes an Throwable and a Context
     * \param t The Throwable
     * \param c The Context
     */
    InvalidDerivedTypeError(const Throwable& t, const Context& c) : Error(t, c)
    {}

    //!  Destructor
    ~InvalidDerivedTypeError()
    {}
}
;




}

#endif
