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
#include <import/str.h>
#include <import/re.h>

net::URL::URL(const std::string url)
{
    if (!url.empty())
        set(url);
}

net::URL::URL(const net::URL& url)
{
    mProtocol = url.getProtocol();
    mHost = url.getHost();
    mPath = url.getPath();
    mFragment = url.getFragment();
    std::map<std::string, std::string> params = url.getParams();
    for (std::map<std::string, std::string>::const_iterator it = params.begin(); it
            != params.end(); ++it)
    {
        mParams[it->first] = it->second;
    }
}

void net::URL::set(std::string url)
{
    std::vector<std::string> parts = net::urlSplit(url);
    mProtocol = parts[0];
    mHost = parts[1];
    mPath = parts[2];
    std::string params = parts[3];
    mFragment = parts[4];

    if (!params.empty())
    {
        std::vector<std::string> paramParts = str::split(params, "&");
        for (size_t i = 0, size = paramParts.size(); i < size; ++i)
        {
            std::string param = paramParts[i];
            size_t pos = param.find("=");
            if (pos > 0 && pos < (param.length() - 1))
            {
                mParams[param.substr(0, pos)] = param.substr(pos + 1);
            }
            else
            {
                mParams[param] = "";
            }
        }
    }
}

std::string net::URL::getProtocol() const
{
    return mProtocol;
}
std::string net::URL::getHost() const
{
    return mHost;
}
int net::URL::getPort() const
{
    re::PCRE regex;
    regex.compile("[^:]:(\\d+)");
    re::PCREMatch match;
    if (regex.match(getHost(), match))
    {
        return str::toType<int>(match[1]);
    }
    return net::DEFAULT_PORT_HTTP;
}
std::string net::URL::getPath() const
{
    return mPath;
}
std::string net::URL::getFragment() const
{
    return mFragment;
}
std::string net::URL::getQuery() const
{
    std::ostringstream s;
    bool firstParam = true;
    for (std::map<std::string, std::string>::const_iterator it =
            mParams.begin(); it != mParams.end(); ++it)
    {
        if (!firstParam)
            s << "&";
        s << net::quote(it->first) << "=" << net::quote(it->second);
        firstParam = false;
    }
    return s.str();
}

std::string net::URL::getDocument() const
{
    std::ostringstream doc;
    doc << "/" << getPath();
    std::string query = getQuery();
    if (!query.empty())
        doc << "?" << query;
    std::string fragment = getFragment();
    if (!fragment.empty())
        doc << "#" << fragment;
    return doc.str();
}

std::string net::URL::getServer() const
{
    std::ostringstream server;
    server << getProtocol() << "://" << getHost();
    return server.str();
}

std::map<std::string, std::string>& net::URL::getParams()
{
    return mParams;
}
const std::map<std::string, std::string>& net::URL::getParams() const
{
    return mParams;
}

std::string net::URL::toString() const
{
    return net::urlJoin(getProtocol(), getHost(), getPath(), getQuery(),
            getFragment());
}

bool net::URL::operator==(const net::URL& url) const
{
    return toString() == url.toString();
}
