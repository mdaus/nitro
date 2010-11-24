/* =========================================================================
 * This file is part of re-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
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

#if defined(USE_PCRE)

#include "re/PCRE.h"

re::PCRE::PCRE(const std::string& pattern, int flags) :
    mPattern(pattern), mMatchString(""), mPCRE(NULL), mNumMatches(0)
{
    if (!mPattern.empty())
        compile(mPattern, flags);
}

void re::PCRE::destroy()
{
    if (mPCRE != NULL)
    {
        pcre_free(mPCRE);
    }
}

re::PCRE::~PCRE()
{
    destroy();
}

re::PCRE::PCRE(const re::PCRE& rhs)
{
    mPCRE = NULL;
    mPattern = rhs.mPattern;
    mMatchString = rhs.mMatchString;

    compile(mPattern);
}
re::PCRE& re::PCRE::operator=(const re::PCRE& rhs)
{
    if (this != &rhs)
    {
        destroy();

        mPattern = rhs.mPattern;
        mMatchString = rhs.mMatchString;

        compile(mPattern);
    }

    return *this;
}

re::PCRE& re::PCRE::compile(const std::string& pattern,
                            int flags)
{
    mPattern = pattern;

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
        throw PCREException(Ctxt(FmtX("PCRE compile error at offset %d: %s",
                                      erroffset, errorptr)));
    }

    return *this;
}


const std::string& re::PCRE::getPattern() const
{
    return mPattern;
}
bool re::PCRE::matches(const std::string& str, int flags)
{

    mMatchString = str;
    int x = pcre_exec(mPCRE, NULL, mMatchString.c_str(), mMatchString.length(), 0, flags, NULL, 0);
    /* zero if there is a match */
    return ( (x == -1) ? (false) : (true) );
}

bool re::PCRE::match(const std::string& str,
                     PCREMatch & matchObject,
                     int flags)
{
    mNumMatches = 0;
    mMatchString = str;

    int result(0);
    int startOffset(0);

    // Clear the output vector
    memset(mOvector, 0, OVECCOUNT);
    mNumMatches =    pcre_exec(mPCRE,                // the compiled pattern
                               NULL,                 // no extra data - not studied
                               mMatchString.c_str(), // the subject string
                               mMatchString.length(),// the subject length
                               startOffset,          // the starting offset in subject
                               flags,                // options
                               mOvector,             // the output vector
                               OVECCOUNT);           // the output vector size

    result = mNumMatches;

    /**************************************************************************
     * (From pcre source code, pcredemo.c)                                    *
     * We have found the first match within the subject string. If the output *
     * vector wasn't big enough, set its size to the maximum. Then output any *
     * substrings that were captured.                                         *
     **************************************************************************/

    /* The output vector wasn't big enough */

    if (mNumMatches == 0)
    {
        mNumMatches = OVECCOUNT / 3;
    }

    // Load up the match object
    for (int i = 0; i < mNumMatches; i++)
    {
        int index = mOvector[2*i];
        int subStringLength = mOvector[2*i+1] - index;
        int subStringCheck = index + subStringLength;
        if (subStringCheck > (int)mMatchString.length())
        {
            throw PCREException(Ctxt(FmtX("Match: Match substring out of range (%d,%d) for string of length %d", index, subStringCheck, mMatchString.length())));
        }
        else if (subStringLength == 0)
        {
            matchObject.push_back("");
        }
        else if (index >= 0)
        {
            matchObject.push_back(mMatchString.substr(index, subStringLength));
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
        throw PCREException
        (Ctxt(FmtX("Error in matching %s", mMatchString.c_str())));
    }
}

std::string re::PCRE::getMatchString() const
{
    return mMatchString;
}

std::string re::PCRE::search(const std::string& matchString,
                             int startIndex,
                             int flags)
{
    mNumMatches = 0;
    mMatchString = matchString;

    int result(0);
    int startOffset(0);

    // Clear the output vector
    memset(mOvector, 0, OVECCOUNT);
    mNumMatches =    pcre_exec(mPCRE,                     // the compiled pattern
                               NULL,                      // no extra data
                               matchString.c_str() + startIndex, // the subject string
                               matchString.length(),             // the subject length
                               startOffset,                      // starting offset
                               flags,                          // options
                               mOvector,                         // output vector
                               OVECCOUNT);                       // output vector size

    result = mNumMatches;

    if (result == 0)
    {
        mNumMatches = OVECCOUNT / 3;
    }

    if (result >= 0)
    {
        if (((mOvector[0] + startIndex) +
                (mOvector[1] - mOvector[0])) >
                (int)mMatchString.length() )
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
        if (subStringCheck > (int)mMatchString.length())
        {
            throw PCREException(Ctxt(FmtX("Search: Match substring out of range (%d,%d) for string of length %d", index, subStringCheck, mMatchString.length())));
        }
        return mMatchString.substr(index, subStringLength);
    }
    else if (result == PCRE_ERROR_NOMATCH)
    {
        return std::string("");
    }
    else
    {
        throw PCREException
        (Ctxt(FmtX("Error in searching for %s", mMatchString.c_str())));
    }
}

void re::PCRE::searchAll(const std::string& matchString,
                         PCREMatch & v)
{
    std::string result = search(matchString);

    int idx = 0;
    while (result.size() != 0)
    {
        v.push_back(result);
        idx += (int)result.size() + mOvector[0];
        result = search(matchString, idx, PCRE_NOTBOL);
    }
}

void re::PCRE::split(const std::string& str,
                     std::vector< std::string > & v)
{
    int idx = 0;
    std::string result = search(str);
    while (result.size() != 0)
    {
        v.push_back(str.substr(idx, mOvector[0]));
        idx += mOvector[1];
        result = search(str, idx, PCRE_NOTBOL);
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
    std::string result = search(str);
    int idx = 0;
    while (result.size() != 0)
    {
        toReplace.replace(idx + mOvector[0], (int)result.size(), repl);
        idx += (int)repl.size() + mOvector[0];
        result = search(toReplace, idx, PCRE_NOTBOL);
    }

    return toReplace;
}

std::string re::PCRE::escape(const std::string& str)
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

#endif
