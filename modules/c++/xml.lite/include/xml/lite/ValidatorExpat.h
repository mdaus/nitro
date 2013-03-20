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

#ifndef __XML_LITE_VALIDATOR_EXPAT_H__
#define __XML_LITE_VALIDATOR_EXPAT_H__

#ifdef USE_EXPAT

#include <except/Exception.h>

namespace xml
{
namespace lite
{

/*!
 * \class ValidatorExpat
 * \brief Schema validation is done here.
 *
 * This class is the Expat schema validator
 */
class ValidatorExpat
{
public:

    ValidatorExpat(const std::vector<std::string>& schemaPaths, 
                   bool recursive = true)
    {
        throw except::Exception(Ctxt("Expat Validation is Unsupported"));
    }

    /*!
     *  Validation against the internal schema pool
     *  \param xml     Input stream to the xml document to validate
     *  \param xmlID   Identifier for this input xml within the error log
     *  \param errors  Object for returning errors found (errors are appended)
     */
    virtual bool validate(io::InputStream& xml,
                          const std::string& xmlID,
                          std::vector<ValidationInfo>& errors) const
    {
        throw except::Exception(Ctxt("Expat Validation is Unsupported"));
    }

    /*!
     *  Validation against the internal schema pool
     *  \param xml     The xml document string to validate
     *  \param xmlID   Identifier for this input xml within the error log
     *  \param errors  Object for returning errors found (errors are appended)
     */
    virtual bool validate(const std::string& xml,
                          const std::string& xmlID,
                          std::vector<ValidationInfo>& errors) const
    {
        throw except::Exception(Ctxt("Expat Validation is Unsupported"));
    }

};
}
}

#endif

#endif
