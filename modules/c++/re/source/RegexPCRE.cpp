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

#include "re/Regex.h"  // __CODA_CPP11 is pulled in here

#ifndef __CODA_CPP11

#include <sstream>

const int re::Regex::mOvectorCount = 999;

re::Regex::Regex(const std::string& pattern) :
    mPattern(pattern), mPCRE(NULL), mOvector(mOvectorCount)
{
    if (!mPattern.empty())
    {
        compile(mPattern);
    }
}

void re::Regex::destroy()
{
    if (mPCRE != NULL)
    {
        pcre_free(mPCRE);
    }
}

re::Regex::~Regex()
{
    destroy();
}

re::Regex::Regex(const re::Regex& rhs) :
    mPattern(rhs.mPattern), mPCRE(NULL), mOvector(mOvectorCount)
{
    compile(mPattern);
}

re::Regex& re::Regex::operator=(const re::Regex& rhs)
{
    if (this != &rhs)
    {
        destroy();

        mPattern = rhs.mPattern;

        compile(mPattern);
    }

    return *this;
}

re::Regex& re::Regex::compile(const std::string& pattern)
{
    mPattern = pattern;
    int flags = PCRE_DOTALL;

    int erroffset;
    const char *errorptr;

    if (mPCRE != NULL)
    {
        pcre_free(mPCRE);
    }

    mPCRE = pcre_compile(mPattern.c_str(),
                         flags,
                         &errorptr,
                         &erroffset,
                         NULL);

    if (mPCRE == NULL)
    {
        std::stringstream ss;
        ss << "PCRE compile error at offset "
           << erroffset << ": " << errorptr;
        throw RegexException(Ctxt(ss.str()));
    }

    return *this;
}


const std::string& re::Regex::getPattern() const
{
    return mPattern;
}

bool re::Regex::matches(const std::string& str) const
{
    int x = pcre_exec(mPCRE, NULL, str.c_str(), str.length(), 0, 0, NULL, 0);
    /* zero if there is a match */
    return ( (x == -1) ? (false) : (true) );
}

bool re::Regex::match(const std::string& str,
                      RegexMatch & matchObject)
{
    int numMatches(0);
    int result(0);
    int startOffset(0);

    // Clear the output vector
    mOvector.assign(mOvector.size(), 0);
    numMatches = pcre_exec(mPCRE,          // the compiled pattern
                           NULL,           // no extra data - not studied
                           str.c_str(),    // the subject string
                           str.length(),   // the subject length
                           startOffset,    // the starting offset in subject
                           0,              // options
                           &mOvector[0],   // the output vector
                           mOvectorCount); // the output vector size
    result = numMatches;
    /**************************************************************************
     * (From pcre source code, pcredemo.c)                                    *
     * We have found the first match within the subject string. If the output *
     * vector wasn't big enough, set its size to the maximum. Then output any *
     * substrings that were captured.                                         *
     **************************************************************************/

    /* The output vector wasn't big enough */

    if (numMatches == 0)
    {
        numMatches = mOvectorCount / 3;
    }

    // Load up the match object
    for (int i = 0; i < numMatches; i++)
    {
        int index = mOvector[2*i];
        int subStringLength = mOvector[2*i+1] - index;
        int subStringCheck = index + subStringLength;
        if (subStringCheck > (int)str.length())
        {
            throw RegexException(Ctxt(FmtX("Match: Match substring out of range (%d,%d) for string of length %d", index, subStringCheck, str.length())));
        }
        else if (subStringLength == 0)
        {
            matchObject.push_back("");
        }
        else if (index >= 0)
        {
            matchObject.push_back(str.substr(index, subStringLength));
        }
        //otherwise, it was likely a non-capturing group
    }

    if (result >= 0)
    {
        return true;
    }
    else if (result == PCRE_ERROR_NOMATCH)
    {
        return false;
    }
    else
    {
        throw RegexException
            (Ctxt(FmtX("Error in matching %s", str.c_str())));
    }
}

std::string re::Regex::search(const std::string& matchString,
                              int startIndex)
{
    return search(matchString, startIndex, 0);
}

std::string re::Regex::search(const std::string& matchString,
                              int startIndex,
                              int flags)
{
    int numMatches(0);
    int result(0);
    int startOffset(0);

    // Clear the output vector
    mOvector.assign(mOvector.size(), 0);
    numMatches = pcre_exec(mPCRE,                             // the compiled pattern
                           NULL,                              // no extra data
                           matchString.c_str() + startIndex,  // the subject string
                           matchString.length() - startIndex, // the subject length
                           startOffset,                       // starting offset
                           flags,                             // options
                           &mOvector[0],                      // output vector
                           mOvectorCount);                    // output vector size

    result = numMatches;

    if (result == 0)
    {
        numMatches = mOvectorCount / 3;
    }

    if (result >= 0)
    {
        if (((mOvector[0] + startIndex) +
             (mOvector[1] - mOvector[0])) >
            (int)matchString.length() )
        {
            result = PCRE_ERROR_NOMATCH;
        }
    }

    if (result >= 0)
    {
        // output vector start offset = i*2
        // output vector end   offset = i*2+1
        // i = 0
        int index = mOvector[0] + startIndex;
        int subStringLength = mOvector[1] - mOvector[0];
        int subStringCheck = index + subStringLength;
        if (subStringCheck > (int)matchString.length())
        {
            throw RegexException(Ctxt(FmtX("Search: Match substring out of range (%d,%d) for string of length %d", index, subStringCheck, matchString.length())));
        }
        return matchString.substr(index, subStringLength);
    }
    else if (result == PCRE_ERROR_NOMATCH)
    {
        return std::string("");
    }
    else
    {
        throw RegexException
            (Ctxt(FmtX("Error in searching for %s", matchString.c_str())));
    }
}

void re::Regex::searchAll(const std::string& matchString,
                          RegexMatch & v)
{
    std::string result = search(matchString, 0, 0);

    int idx = 0;
    while (result.size() != 0)
    {
        v.push_back(result);
        idx += (int)result.size() + mOvector[0];
        result = search(matchString, idx, PCRE_NOTBOL);
    }
}

void re::Regex::split(const std::string& str,
                      std::vector<std::string> & v)
{
    size_t idx = 0;
    std::string result = search(str, 0, 0);
    while (result.size() != 0)
    {
        v.push_back(str.substr(idx, mOvector[0]));
        idx += mOvector[1];
        result = search(str, idx, PCRE_NOTBOL);
    }
    // Push on last bit if there is some
    if (!str.substr(idx).empty())
    {
        v.push_back(str.substr(idx));
    }
}

std::string re::Regex::sub(const std::string& str,
                           const std::string& repl)
{
    std::string toReplace = str;
    std::string result = search(str, 0, 0);
    size_t idx = 0;
    while (result.size() != 0)
    {
        toReplace.replace(idx + mOvector[0], (int)result.size(), repl);
        idx += (int)repl.size() + mOvector[0];
        result = search(toReplace, idx, PCRE_NOTBOL);
    }

    return toReplace;
}

std::string re::Regex::escape(const std::string& str) const
{
    std::string r;
    for (size_t i = 0; i < str.length(); i++)
    {
        if (!isalpha(str[i]) && !isspace(str[i]))
        {
            r += '\\';
        }
        r += str[i];
    }
    return r;
}

#endif
