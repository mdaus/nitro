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

#ifdef USE_PCRE2
#ifndef __CODA_CPP11

#include <sstream>

namespace
{
class ScopedMatchData
{
public:
    ScopedMatchData(const pcre2_code* code) :
        mCode(code),
        mMatchData(pcre2_match_data_create_from_pattern(code, NULL))
    {
        if (mMatchData == NULL)
        {
            throw re::RegexException(Ctxt(
                    "pcre2_match_data_create_from_pattern() failed to "
                    "allocate memory"));
        }
    }

    ~ScopedMatchData()
    {
        pcre2_match_data_free(mMatchData);
    }

    pcre2_match_data* get()
    {
        return mMatchData;
    }

    PCRE2_SIZE* getOutputVector()
    {
        return pcre2_get_ovector_pointer(mMatchData);
    }

    // Returns the number of matches
    size_t match(const std::string& subject,
                 PCRE2_SIZE startOffset = 0,
                 sys::Uint32_T options = 0)
    {
        // This returns the number of matches
        // But for no matches, it returns PCRE2_ERROR_NOMATCH
        // Other return codes less than 0 indicate an error
        const int returnCode =
                pcre2_match(mCode,
                            reinterpret_cast<PCRE2_SPTR>(subject.c_str()),
                            subject.length(),
                            startOffset,
                            options,
                            mMatchData,
                            NULL); // Match context

        if (returnCode == PCRE2_ERROR_NOMATCH)
        {
            return 0;
        }
        else if (returnCode < 0)
        {
            // Some error occurred
            throw re::RegexException(Ctxt("pcre2_match() failed"));
        }
        else
        {
            return returnCode;
        }
    }

private:
    const pcre2_code* const mCode;
    pcre2_match_data* const mMatchData;
};
}

re::Regex::Regex(const std::string& pattern) :
    mPattern(pattern), mPCRE(NULL)
{
    if (!mPattern.empty())
    {
        compile(mPattern);
    }
}

void re::Regex::destroy()
{
	// TODO: Use an RAII class for just this object
    if (mPCRE != NULL)
    {
        pcre2_code_free(mPCRE);
        mPCRE = NULL;
    }
}

re::Regex::~Regex()
{
    destroy();
}

re::Regex::Regex(const re::Regex& rhs) :
    mPattern(rhs.mPattern), mPCRE(NULL)
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

    destroy();

    static const int FLAGS = PCRE2_DOTALL;
    int errorCode;
    PCRE2_SIZE errorOffset;
    mPCRE = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(mPattern.c_str()),
    		              mPattern.length(),
                          FLAGS,
						  &errorCode,
						  &errorOffset,
                          NULL); // Use default compile context

    if (mPCRE == NULL)
    {
    	PCRE2_UCHAR buffer[256];
    	pcre2_get_error_message(errorCode, buffer, sizeof(buffer));
    	std::ostringstream ostr;
    	ostr << "PCRE compilation failed at offset " << errorOffset
    	     << ": " << buffer;
        throw RegexException(Ctxt(ostr.str()));
    }

    return *this;
}

// TODO: This can be inlined in the base class instead of implemented for both
const std::string& re::Regex::getPattern() const
{
    return mPattern;
}

bool re::Regex::matches(const std::string& str) const
{
    ScopedMatchData matchData(mPCRE);
    return (matchData.match(str) > 0);
}

bool re::Regex::match(const std::string& str,
                      RegexMatch& matchObject)
{
    ScopedMatchData matchData(mPCRE);
    const size_t numMatches = matchData.match(str);
    matchObject.resize(numMatches);

    if (numMatches == 0)
    {
        return false;
    }

    const PCRE2_SIZE* const outVector = matchData.getOutputVector();

    // TODO: Make this a convenience function too to get one match at a time?
    for (size_t ii = 0; ii < numMatches; ++ii)
    {
        const size_t index = outVector[ii * 2];
        const size_t end = outVector[ii * 2 + 1];

        if (end > str.length())
        {
            // Presumably this never happens
            std::ostringstream ostr;
            ostr << "Match: Match substring out of range ("
                 << index << ", " << end << ") for string of length "
                 << str.length();
            throw RegexException(Ctxt(ostr.str()));
        }

        const size_t subStringLength = end - index;
        matchObject[ii] = str.substr(index, subStringLength);
    }

    return true;
}

// TODO: Change startIndex here and in other function to be a size_t
std::string re::Regex::search(const std::string& matchString,
                              int startIndex)
{
    return search(matchString, startIndex, 0);
}

#if 0
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
                           OVECTOR_COUNT);                    // output vector size

    result = numMatches;

    if (result == 0)
    {
        numMatches = OVECTOR_COUNT / 3;
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
#endif
#endif
