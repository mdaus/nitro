/* =========================================================================
 * This file is part of re-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
 *
 * re-c++ is free software; you can redistribute it and/or modify
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

#include "re/PCRE.h" // this has to come before the #ifdef checks below

// we use this file if we're not using actual PCRE itself
#if defined(USE_PCRE) && defined(__CODA_CPP11)

re::PCRE::PCRE(const std::string& pattern, int flags) :
    mPattern(pattern)
{
    if (!mPattern.empty())
        compile(mPattern, flags);
}

void re::PCRE::destroy()
{
}

re::PCRE::~PCRE()
{
    destroy();
}

re::PCRE::PCRE(const re::PCRE& rhs)
{
    mPattern = rhs.mPattern;
    compile(mPattern);
}

re::PCRE& re::PCRE::operator=(const re::PCRE& rhs)
{
    if (this != &rhs)
    {
        destroy();

        mPattern = rhs.mPattern;

        compile(mPattern);
    }

    return *this;
}

re::PCRE& re::PCRE::compile(const std::string& pattern,
                            int flags)
{
    mPattern = (flags==PCRE_DOTALL) ? replace_dot(pattern) : pattern;
    try 
    {
        mRegex = std::regex(mPattern);
    }
    catch (const std::regex_error& e)
    {
        throw PCREException(Ctxt(FmtX("PCRE std::regex constructor error: %s",
                                      e.what())));
    }

    return *this;
}


const std::string& re::PCRE::getPattern() const
{
    return mPattern;
}

bool re::PCRE::matches(const std::string& str, int) const
{
    return std::regex_match(str, mRegex);
}

bool re::PCRE::match(const std::string& str,
                     PCREMatch & matchObject,
                     int )
{
    std::smatch matches;
    bool result = std::regex_search(str, matches, mRegex);

    // copy resulting substrings into matchObject
    matchObject.resize( matches.size() );

    // This causes a crash for some reason
    //std::copy(matches.begin(), matches.end(), matchObject.begin());

    for(size_t i=0; i<matches.size(); ++i) {
        matchObject[i] = matches[i].str();
    }

    return result;
}

std::string re::PCRE::search(const std::string& matchString,
                             int startIndex,
                             int)
{
    std::smatch matches;

    // search the string starting at index "startIndex"
    bool result = std::regex_search(matchString.begin()+startIndex, 
                                    matchString.end(), matches, mRegex);
    
    // if successful, return the substring matching the regex,
    // otherwise return empty string
    if (result && matches.size() > 0)
    {
        return matches[0].str();
    }
    else
    {
        return std::string("");
    }
}

void re::PCRE::searchAll(const std::string& matchString,
                         PCREMatch& v)
{
    // this iterates mRegex over the input string, returning an
    // iterator to the match objects
    auto words_begin = 
        std::sregex_iterator(matchString.begin(), matchString.end(), mRegex);
    auto words_end = std::sregex_iterator();
 
    // copy the matches into v
    for (std::sregex_iterator match_iter = words_begin; 
         match_iter != words_end;++match_iter)
    {
        std::string match_str = match_iter->str(); 
        v.push_back(match_str);
    }
}

void re::PCRE::split(const std::string& str,
                     std::vector< std::string > & v)
{
    int idx = 0;
    auto flags = std::regex_constants::match_default;
    std::smatch match;
    while (std::regex_search(str.begin()+idx, str.end(), match, mRegex, flags))
    {
        v.push_back( str.substr(idx, match.position()) );
        idx += (match.position() + match.length());

        // not sure this will ever be needed for a split() operation,
        // but we'll be safe (matches after first match will not match
        // beginning-of-line))
        if(flags == std::regex_constants::match_default)
            flags = std::regex_constants::match_not_bol;
    }

    // Push on last bit if there is some
    if (str.substr(idx).length() > 0)
    {
        v.push_back(str.substr(idx));
    }
}

std::string re::PCRE::sub(const std::string& str,
                          const std::string& repl)
{
    std::string toReplace = str;

    int idx = 0;
    auto flags = std::regex_constants::match_default;
    std::smatch match;
    while (std::regex_search(toReplace.cbegin()+idx, toReplace.cend(), match, mRegex, flags))
    {
        toReplace.replace(idx + match.position(), match.length(), repl);
        idx += (match.position() + match.length());

        // matches after first match will not match beginning-of-line
        if(flags == std::regex_constants::match_default)
            flags = std::regex_constants::match_not_bol;
    }

    return toReplace;
}

std::string re::PCRE::escape(const std::string& str) const
{
    std::string r;
    for (unsigned int i = 0; i < str.length(); i++)
    {
        if (!isalpha(str.at(i)) && !isspace(str.at(i)))
        {
            r += '\\';
        }
        r += str.at(i);
    }
    return r;
}

std::string re::PCRE::replace_dot(const std::string& str) const
{
    std::regex reg("([^\\\\])\\.");
    std::string newstr = std::regex_replace(str, reg, "$1[\\s\\S]");
    return newstr;
}

#endif
