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

#ifndef __XML_LITE_VALIDATOR_XERCES_H__
#define __XML_LITE_VALIDATOR_XERCES_H__

#ifdef USE_XERCES

#include <memory>
#include <vector>

#include "xml/lite/UtilitiesXerces.h"
#include "xml/lite/ValidatorInterface.h"

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/parsers/DOMLSParserImpl.hpp>
#include <xercesc/framework/XMLGrammarPool.hpp>
#include <xercesc/framework/XMLGrammarPoolImpl.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/impl/DOMLSInputImpl.hpp>

#include <xercesc/framework/MemBufInputSource.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>

namespace xml
{
namespace lite
{

typedef xercesc::DOMError ValidationError;

class ValidationErrorHandler : public xercesc::DOMErrorHandler
{
public:
    ValidationErrorHandler() {}

    //! handle the errors during validation
    virtual bool handleError (const ValidationError& err);

    //! get the raw information
    const std::vector<ValidationInfo>& getErrorLog() const
    {
        return mErrorLog;
    }

    void clearErrorLog() { mErrorLog.clear(); }

    //! set the id to differentiate between errors
    void setID(const std::string& id) { mID = id; }

    //! stream to a string
    std::string toString() const
    {
        std::ostringstream oss;
        oss << this;
        return oss.str();
    }

protected:
    std::vector<ValidationInfo> mErrorLog;
    std::string mID;
};

/*!
 * \class ValidatorXerces
 * \brief Schema validation is done here.
 *
 * This class is the Xercesc schema validator
 */
class ValidatorXerces
{
private:
    XercesContext mCtxt;    //! this must be the first member listed

public:

    ValidatorXerces(const std::vector<std::string>& schemaPaths, 
                    bool recursive = true);

    //! Destructor.
    virtual ~ValidatorXerces();

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

protected:

    std::auto_ptr<xercesc::XMLGrammarPool> mSchemaPool;
    std::auto_ptr<xml::lite::ValidationErrorHandler> mErrorHandler;
    std::auto_ptr<xercesc::DOMLSParser> mValidator;

};
}
}

//! stream the entire log -- newline separated
std::ostream& operator<< (
    std::ostream& out, 
    const xml::lite::ValidationErrorHandler& errorHandler);

#endif

#endif
