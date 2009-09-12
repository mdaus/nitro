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

#include "net/URL.h"

void net::URL::set(const std::string& url)
{
    re::PCRE rx;
    rx.compile(net::URL::MATCH());

    re::PCREMatch matches;
    if (!rx.match(url, matches))
        throw net::MalformedURLException(Ctxt(url));

    std::string protocol = matches[2];
    if (protocol.length())
        mProto = protocol;

    std::string line = matches[3];
    std::vector<std::string> hostPortPair =
        str::Tokenizer(line, ":");

    int hostPortPairNum = (int)hostPortPair.size();
    if (hostPortPairNum != 1 && hostPortPairNum != 2)
        throw except::Exception(Ctxt("Host and port pair must contain at least a host"));

    mHost = hostPortPair[0];

    if (hostPortPairNum == 1)
    {
        if (mProto == "https")
        {
            mPort = DEFAULT_HTTPS_PORT;
        }
        else
        {
            mPort = DEFAULT_HTTP_PORT;
        }
    }
    else
    {
        mPort = atoi(hostPortPair[1].c_str());
    }

    if (matches.size() == 5)
    {
        dbg_printf("Found document: %s\n",
                   matches[4].c_str());
        mDocument = matches[4];
    }
    if (mDocument.length() == 0)
    {
        mDocument = "/index.html";
    }
}

