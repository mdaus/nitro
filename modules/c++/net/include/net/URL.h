/* =========================================================================
 * This file is part of net-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * net-c++ is free software; you can redistribute it and/or modify
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

#ifndef __NET_URL_H__
#define __NET_URL_H__

#include <string>
#include <map>
#include "net/NetExceptions.h"
#include "net/NetUtils.h"

/*! \file URL.h
 *  \brief Class for abstract locations
 *
 *  Allows several ways of specifying location.
 *
 */

namespace net
{

class URL
{
public:

    URL(const std::string url = "");

    /*!
     *  Copy constructor.
     *  \param url A right-hand-side URL
     */
    URL(const URL& url);

    virtual ~URL()
    {
    }

    void set(std::string url);

    std::string getProtocol() const;
    std::string getHost() const;
    int getPort() const;
    std::string getPath() const;
    std::string getFragment() const;
    std::string getQuery() const;
    std::string getServer() const;
    std::string getDocument() const;
    std::map<std::string, std::string>& getParams();
    const std::map<std::string, std::string>& getParams() const;
    std::string toString() const;

    /*!
     * Are these URLs equal
     * \param url A URL to compare
     */
    bool operator==(const URL& url) const;

protected:
    friend class URLBuilder;
    std::string mProtocol;
    std::string mHost;
    std::string mPath;
    std::map<std::string, std::string> mParams;
    std::string mFragment;
};
}
#endif

