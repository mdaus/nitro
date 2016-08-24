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

    const PCRE2_SIZE* getOutputVector() const
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

    std::string getMatch(const std::string& str, size_t idx) const
    {
    	const PCRE2_SIZE* const outVector = getOutputVector();

        const size_t index = outVector[idx * 2];
        const size_t end = outVector[idx * 2 + 1];

        if (end > str.length())
        {
            // Presumably this never happens
            std::ostringstream ostr;
            ostr << "Match: Match substring out of range ("
                 << index << ", " << end << ") for string of length "
                 << str.length();
            throw re::RegexException(Ctxt(ostr.str()));
        }

        const size_t subStringLength = end - index;
        return str.substr(index, subStringLength);
    }

private:
    // Noncopyable
    ScopedMatchData(const ScopedMatchData& );
    ScopedMatchData& operator=(const ScopedMatchData& );

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
    matchObject.resize(numMatches); // TODO: Are we upposed to do this or just push back??

    if (numMatches == 0)
    {
        return false;
    }

    for (size_t ii = 0; ii < numMatches; ++ii)
    {
        matchObject[ii] = matchData.getMatch(str, ii);
    }

    return true;
}

// TODO: Change startIndex here and in other function to be a size_t
std::string re::Regex::search(const std::string& matchString,
                              int startIndex)
{
	size_t begin;
	size_t end;
    return search(matchString, startIndex, 0, begin, end);
}

std::string re::Regex::search(const std::string& matchString,
						      size_t startIndex,
							  sys::Uint32_T flags,
							  size_t& begin,
							  size_t& end)
{
    ScopedMatchData matchData(mPCRE);
    const size_t numMatches = matchData.match(matchString, startIndex, flags);

    if (numMatches > 0)
    {
    	// TODO: Does startIndex work properly with this?
    	begin = matchData.getOutputVector()[0];
    	end = matchData.getOutputVector()[1];
		return matchData.getMatch(matchString, 0);
    }
    else
    {
    	begin = end = 0;
    	return "";
    }
}

void re::Regex::searchAll(const std::string& matchString,
                          RegexMatch& v)
{
	size_t begin;
	size_t end;
    std::string result = search(matchString, 0, 0, begin, end);

    size_t idx = 0;
    while (!result.empty())
    {
        v.push_back(result);
        idx += result.size() + begin;
        result = search(matchString, idx, PCRE2_NOTBOL, begin, end);
    }
}

void re::Regex::split(const std::string& str,
                      std::vector<std::string> & v)
{
	size_t begin;
	size_t end;
    size_t idx = 0;
    std::string result = search(str, 0, 0, begin, end);
    while (!result.empty())
    {
        v.push_back(str.substr(idx, begin));
        idx += end;
        result = search(str, idx, PCRE2_NOTBOL, begin, end);
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
	size_t begin;
	size_t end;
    std::string toReplace = str;
    std::string result = search(str, 0, 0, begin, end);
    size_t idx = 0;
    while (!result.empty())
    {
        toReplace.replace(idx + begin, result.size(), repl);
        idx += repl.size() + begin;
        result = search(toReplace, idx, PCRE2_NOTBOL, begin, end);
    }

    return toReplace;
}

std::string re::Regex::escape(const std::string& str) const
{
	// TODO: Put this in common class
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
