/* =========================================================================
 * This file is part of xml.lite-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
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

#ifndef __XML_LITE_XML_EXCEPTION_H__
#define __XML_LITE_XML_EXCEPTION_H__

#include "except/Exception.h"

/*!
 *  \file XMLException.h
 *  \brief Contains the exceptions specific to XML
 *
 *  This file contains all of the specialized XML exceptions used by
 *  the xml::lite package
 */
namespace xml
{
namespace lite
{

/*!
 *  \class XMLException
 *  \brief The base XML exception class
 *
 *  This is the default XML exception, for when
 *  other, more specialized exception make no sense
 */
class XMLException: public except::Exception
{
public:
    //!  Default constructor
    XMLException()
    {}

    /*!
     *  Constructor taking a message
     *  \param message The exception message
     */
    XMLException(const char *message): except::Exception(message)
    {}

    /*!
     *  Constructor taking a message
     *  \param message The exception message
     */
    XMLException(const std::string & message): except::Exception(message)
    {}

    /*!
     *  Constructor taking a Context
     *  \param c The exception Context
     */
    XMLException(const except::Context& c): except::Exception(c)
    {}


    //! Destructor
    virtual ~ XMLException()
    {}
};

/*!
 *  \class XMLNotRecognizedException
 *  \brief Specialized for badly formatted/incorrectly processed data
 *
 *  Provides the derived implementation for bad formatting or
 *  for incorrect processing
 */
class XMLNotRecognizedException: public XMLException
{
public:
    //!  Default constructor
    XMLNotRecognizedException()
    {}

    /*!
     *  Constructor taking a message
     *  \param message The exception message
     */
    XMLNotRecognizedException(const char *message): XMLException(message)
    {}

    /*!
     *  Constructor taking a message
     *  \param message The exception message
     */
    XMLNotRecognizedException(const std::
                              string & message): XMLException(message)
    {}

    /*!
            *  Constructor taking a context
            *  \param c The exception context
            */
    XMLNotRecognizedException(const except::
                              Context& c): XMLException(c)
    {}


    //! Destructor
    virtual ~ XMLNotRecognizedException()
    {}
};

/*!
 *  \class XMLNotSupportedException
 *  \brief You might get this if we dont support some XML feature
 *
 *  This is specifically for problems that occur with incompleteness
 *  of implementation, or with custom implementations on other
 *  systems that are not supported by the SAX/DOM standard
 *
 */
class XMLNotSupportedException: public XMLException
{
public:
    //!  Default constructor
    XMLNotSupportedException()
    {}

    /*!
     *  Constructor taking a message
     *  \param message The exception message
     */
    XMLNotSupportedException(const char *message): XMLException(message)
    {}

    /*!
     *  Constructor taking a message
     *  \param message The exception message
     */
    XMLNotSupportedException(const std::
                             string & message): XMLException(message)
    {}

    /*!
            *  Constructor taking a context
            *  \param c The exception context
            */
    XMLNotSupportedException(const except::
                             Context& c): XMLException(c)
    {}


    //! Destructor
    virtual ~ XMLNotSupportedException()
    {}
};

/*!
 *  \class XMLParseException
 *  \brief The interface for parsing exception
 *
 *  This class provides the exception interface for handling
 *  XML exception while processing documents
 *
 */
class XMLParseException: public XMLException
{
public:
    /*!
     *  Construct a parse exception
     *  \param message A message as presented by the parser
     *  \param row As reported by the parser
     *  \param column As reported by the parser
     */
    XMLParseException(const char *message,
                      int row = 0,
                      int column = 0): XMLException(message)
    {
        form(row, column);
    }

    /*!
     *  Construct a parse exception
     *  \param message A message as presented by the parser
     *  \param row As reported by the parser
     *  \param column As reported by the parser
     *  \param errNum An error number given by the parser
     */
    XMLParseException(const std::string & message,
                      int row = 0,
                      int column = 0,
                      int errNum = 0): XMLException(message)
    {
        form(row, column);
    }

    /*!
            *  Construct a parse exception
            *  \param c A context with a message as presented by the parser
            *  \param row As reported by the parser
            *  \param column As reported by the parser
            *  \param errNum An error number given by the parser
            */
    XMLParseException(const except::Context& c,
                      int row = 0,
                      int column = 0,
                      int errNum = 0): XMLException(c)
    {
        form(row, column);
    }


    //! Destructor
    virtual ~ XMLParseException()
    {}

private:
    /*!
     *  Creates the actual message
     *  \param row As reported by the constructor
     *  \param col As reported by the constructor
     *
     */
    void form(int row,
              int column)
    {
        std::ostringstream oss;
        oss << " (" << row << ',' << column << "): " << mMessage;
        mMessage = oss.str();
    }
};
}
}
#endif
