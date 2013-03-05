/* =========================================================================
 * This file is part of xml.lite-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2011, General Dynamics - Advanced Information Systems
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

#ifndef __XML_LITE_VALIDATOR_INTERFACE_H__
#define __XML_LITE_VALIDATOR_INTERFACE_H__

/*!
 * \file ValidatorInterface.h
 * \brief This is the API for the validator interface.
 *
 * This object is the holder of a single schema in memory
 * and gives the ability to validate any xml against this
 * schema.
 */

#include <string>
#include <vector>
#include <io/InputStream.h>
#include <str/Convert.h>

namespace xml
{
namespace lite
{

/*!
 * \class ValidationInfo
 * \brief This is the information for one 
 *        schema validation error.
 */
class ValidationInfo
{
public:
    ValidationInfo(const std::string& message,
                   const std::string& level,
                   const std::string& file,
                   size_t line) :
        mMessage(message), mLevel(level), 
        mFile(file), mLine(line) 
    {
    }

    std::string getMessage() const { return mMessage; }
    std::string getLevel() const { return mLevel; }
    std::string getFile() const { return mFile; }
    size_t getLine() const { return mLine; }

    std::ostream& operator<< (std::ostream& out) const
    {
        out << "[" << this->getLevel() << "]" << 
            " from File: " << this->getFile() << 
            " on Line: " << str::toString(this->getLine()) << 
            " with Message: " << this->getMessage();
        return out;
    }

    //! stream to a string
    std::string toString() const
    {
        std::ostringstream oss;
        oss << this;
        return oss.str();
    }

private:
    std::string mMessage;
    std::string mLevel;
    std::string mFile;
    size_t mLine;
};


/*!
 * \class ValidatorInterface
 * \brief Schema validation is done here.
 *
 * This class is the interface for schema validators
 */
class ValidatorInterface
{
public:

    enum ValidationErrorType
    {
        VALIDATION_WARNING = 0,
        VALIDATION_ERROR,
        VALIDATION_FATAL,
    };

    //! Constructor.
    ValidatorInterface(const std::vector<std::string>& schemaPaths, 
                       bool recursive = true) {}

    //! Destructor.
    virtual ~ValidatorInterface() {}

    /*!
     *  Validation against the internal schema
     *  \param errors  Object for returning errors found
     *  \param is      This is the input stream to feed the parser
     *  \param size    This is the size of the stream to feed the parser
     */
    virtual bool validate(std::vector<ValidationInfo>& errors,
                          const std::string& xmlID,
                          io::InputStream& xml, 
                          sys::SSize_T size = io::InputStream::IS_END);

};
}
}

#endif
