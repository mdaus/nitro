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

#include "net/NetUtils.h"
#include "net/NetExceptions.h"
#include <import/str.h>
#include <import/re.h>
#include <iostream>

std::vector<std::string> net::urlSplit(std::string url)
{
    re::PCRE regex;
    regex.compile(
            "([A-Za-z]+)://([^/?#]+)(/[^?#]+)?(?:[?]([^&#/]+(?:[&][^&#/]+)*)?)?(?:[#](.*))?");

    re::PCREMatch match;
    if (regex.match(url, match))
    {
        size_t matchLen = match.size();
        std::vector<std::string> parts(5, "");
        for (int i = 1; i <= 6; ++i)
        {
            if (i < matchLen)
                parts[i - 1] = match[i];
        }
        return parts;
    }
    else
    {
        throw net::MalformedURLException(url.c_str());
    }
}

std::string net::urlJoin(std::string scheme, std::string location,
        std::string path, std::string query, std::string fragment)
{
    std::ostringstream url;
    url << scheme << "://" << location;
    if (!path.empty())
    {
        if (!str::startsWith(path, "/"))
            url << "/";
        url << path;
    }
    if (!query.empty())
        url << "?" << query;
    if (!fragment.empty())
        url << "#" << fragment;
    return url.str();
}

std::string net::urlJoin(const std::vector<std::string>& parts)
{
    size_t numParts = parts.size();
    if (numParts < 2)
        throw net::MalformedURLException("No URL provided");
    std::string scheme, location, path, query, fragment;
    scheme = parts[0];
    location = parts[1];
    if (numParts > 2)
    {
        path = parts[2];
        if (numParts > 3)
        {
            query = parts[3];
            if (numParts > 4)
                fragment = parts[4];
        }
    }
    return urlJoin(scheme, location, path, query, fragment);
}

std::string net::quote(std::string s)
{
    std::ostringstream quoted;
    re::PCRE regex;
    regex.compile("[A-Za-z0-9+-._]");
    for (size_t i = 0, len = s.length(); i < len; ++i)
    {
        std::string c = s.substr(i, 1);
        if (regex.matches(c))
            quoted << c[0];
        else
            quoted << "%" << std::hex << ((int) c[0]) << std::dec;
    }
    return quoted.str();
}

std::string net::unquote(std::string s)
{
    std::ostringstream unquoted;
    std::vector<std::string> parts = str::split(s, "%");
    size_t numParts = parts.size();
    if (numParts > 0)
        unquoted << parts[0];
    for (size_t i = 1; i < numParts; ++i)
    {
        std::string part = parts[i];
        std::string hexStr = "0x" + part.substr(0, 2);
        long val = strtol(hexStr.c_str(), NULL, 16);
        unquoted << (char) val;
        if (part.length() > 2)
            unquoted << part.substr(2);
    }
    return unquoted.str();
}
