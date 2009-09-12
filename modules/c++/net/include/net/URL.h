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
#include "net/NetExceptions.h"
#include "sys/Dbg.h"
#include "re/PCRE.h"
#include "str/Tokenizer.h"


/*! \file URL.h
 *  \brief Class for abstract locations
 *
 *  Allows several ways of specifying location.
 *
 */

namespace net
{

/*!
 *  Class to abstract the details of a location on the internet.
 *  The URL-type format is host:port, and the internal representation
 *  is std::string, int
 *
 */
class URL
{
public:
    const static char *MATCH()
    {
        return "(([a-zA-Z]+)://)?([^/]+)(.*)";
    }
    enum { DEFAULT_HTTP_PORT = 80 };
    enum { DEFAULT_HTTPS_PORT = 443 };

    //! Default constructor.
    URL() : mProto("http"), mPort(DEFAULT_HTTP_PORT)
    {}

    /*!
     *  Takes a full URL or a host:port formatted string and
     *  parses out the host and port.
     *  \param  url   The location
     */
    URL(const std::string& url) : mProto("http")
    {
        set(url);
    }

    /*!
     *  Take host/port representation
     *  \param  host    The host name
     *  \param  port    The port number
     */
    URL(std::string host, int port)
    {
        set(host, port);
    }

    /*!
     *  Copy constructor.
     *  \param url A right-hand-side URL
     */
    URL(const URL& url)
    {
        mProto = url.mProto;
        mHost = url.mHost;
        mPort = url.mPort;
        mDocument = url.mDocument;
    }

    /*!
     *  Assignment operator
     *  \param url A right-hand-side URL
     *  \return *this
     */
    URL& operator=(const URL& url)
    {
        if (this != &url)
        {
            mProto = url.mProto;
            mHost = url.mHost;
            mPort = url.mPort;
            mDocument = url.mDocument;
        }
        return *this;
    }

    //! Default deconstructor.
    ~URL()
    {}

    /*!
     * Set method
     * \param url A well-formed url
     */
    void set(const char *url)
    {
        set(std::string(url));
    }

    /*!
     * Set method
     * \param url A well-formed url
     */
    void set(const std::string& url);

    /*!
     * Set method
     * \param host The host name
     * \param port The port number
     */
    void set(const std::string& host,
                 int port)
    {
        mHost = host; mPort = port;
    }

    /*!
     * Set method
     * \param host The host name
     * \param port The port number
     */
    void set(const char* host, int port)
    {
        mHost = host; mPort = port;
    }

    /*!
     * Are these URLs equal
     * \param url A URL to compare
     */
    bool operator==(const URL& url)
    {
        return (mPort == url.mPort &&
                mHost == url.mHost &&
                mProto == url.mProto &&
                mDocument == url.mDocument);
    }

    /*!
     *  Get URL-style formatting.  This used to use stringstream
     *  and then string.  Both hung.  I replaced their calls with 
     *  sprintf and now it works.  I will revert back if I can.
     *  It looks like some bug in the Regex library is smothering
     *  everything -- causing the system to hang.  That is the 
     *  next order of business.
     *
     *  \return  String with host:port format
     */
    std::string toString() const
    {
        char url[512];
        memset(url, 0, 512);
        int  mark = 0;
        if (mProto.length())
        {
            sprintf(url, "%s://", mProto.c_str() );
            mark += (int)strlen(url);
        }
        // Size check
        sprintf(url + mark, "%s", mHost.c_str());
        mark = (int)strlen(url);

        if (mPort != DEFAULT_HTTP_PORT && mPort != DEFAULT_HTTPS_PORT)
        {
            sprintf(url + mark, ":%d", mPort);
            mark = (int)strlen(url);
        }
        if (mDocument.length())
        {
            // Size check
            sprintf(url + mark, "%s", mDocument.c_str());
        }

        return std::string(url);

    }

    //! Get document location
    std::string getDocument() const
    {
        return mDocument;
    }

    //! Get connetion protocol
    std::string getProtocol() const
    {
        return mProto;
    }

    //! Get host name
    std::string getHost() const
    {
        return mHost;
    }

    //! Get port number
    int getPort() const
    {
        return mPort;
    }

private:
    std::string  mDocument;
    std::string  mProto;
    std::string  mHost;
    int          mPort;
};
}
#endif

