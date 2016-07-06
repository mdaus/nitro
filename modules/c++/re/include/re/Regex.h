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


#ifndef __RE_REGEX_H__
#define __RE_REGEX_H__

#include <sys/sys_config.h>

#include "sys/Err.h"
#include "re/RegexException.h"

#ifdef __CODA_CPP11
#include <regex>
#else
#include <pcre.h>
#include <pcreposix.h>
#endif

#include <string>
#include <vector>

/*!
 *  \file Regex.h
 *  \brief C++ wrapper for the PCRE library or std::regex if C++11 is available
 */

namespace re
{

    typedef std::vector<std::string> RegexMatch;
    /*!
     *  \class Regex
     *  \brief C++ wrapper object for regular expressions. If C++11 is
     *  available, std::regex is used, otherwise the PCRE library is
     *  used.  For further documentation regarding the underlying PCRE
     *  library, especially for flag information, can be found at
     *  http://www.pcre.org.
     */
    class Regex
    {
    public:
        /*!
         *  The default constructor
         */
        Regex(const std::string& pattern = "");

        //!  Destructor
        ~Regex();

        /*!
         *  Copy constructor
         *  \param rhs The Regex to copy from
         */
        Regex(const Regex& rhs);

        /*!
         *  Assignment operator.  Check for self assignment
         *  \param rhs The Regex to copy from
         *  \return This
         */
        Regex& operator=(const Regex& rhs);

        /*!
         *  Destroy all data used for matching.
         */
        void destroy();

        /*!
         *  Set the match pattern
         *  \param pattern  A pattern to match
         *  \throw  exception on error
         */
        Regex& compile(const std::string& pattern);

        /*!
         *  \todo Add non-const reference
         *  A const reference return for the pattern
         *  \return the pattern
         */
        const std::string& getPattern() const;

        /*!
         *  Match this input string against our pattern and populate the
         *  data structure
         *  \param str  The string to match against our pattern
         *  \param matches  RegexMatch container to fill
         *  \return  True on success, False otherwise
         *  \throw  RegexException on fatal error
         */
        bool match(const std::string& str,
                   RegexMatch& matchObject);

        bool matches(const std::string& str) const;

        /*!
         *  Search the matchString
         *  \param matchString  The string to try and match
         *  \param startIndex  Starting where?
         *  \return  Matched substring
         *  \throw  RegexException on fatal error
         */
        std::string search(const std::string& matchString,
                           int startIndex = 0);

        /*!
         *  Search the matchString and get the sub-expressions, by ref
         *  \param matchString  The string to match
         *  \param v  The vector to fill with sub-expressions
         */
        void searchAll(const std::string& matchString,
                       RegexMatch& v);

        /*!
         *  Split the string by occurrences of the pattern
         *  \param str  The string to split
         *  \param v    The resulting container of matches split from str
         */
        void split(const std::string& str,
                   std::vector<std::string>& v);

        /*!
         *  Replace occurrences of the pattern in the string
         *  \param str  The string in which to replace the pattern
         *  \param repl  The replacement
         *  \return  The resulting string
         */
        std::string sub(const std::string& str,
                        const std::string& repl);

        /*!
         *  Backslash all non-alphanumeric characters
         *  \param str  The string to escape
         *  \return  The escaped string
         */
        std::string escape(const std::string& str) const;

    private:
        std::string mPattern;

#ifdef __CODA_CPP11
        /*!
         *  Replace non-escaped "." with "[\s\S]" to get PCRE_DOTALL newline behavior
         *  \param str  The string to modify
         *  \return  The modified string
         */
        std::string replaceDot(const std::string& str) const;

        /*!
         *  Search using std::regex appropriately based on input string:
         *   regexps starting with ^ are forced to match at beginning
         *   regexps ending with $ (and no ^ at beginning) cause exception
         *   regexps bracketed with ^ and $ match beginning and end
         *   regexps with neither ^ or $ search normally
         *   regexps with either ^ or $ in locations besides beginning and end throw excpetions
         *  \param inputIterBegin  The beginning of the string to search
         *  \param inputIterEnd  The end of the string to search
         *  \param match  The match object for search results
         *  \param matchBeginning  If false, do not match ^ to beginning of string
         *  \return  True on success, otherwise False
         *  \throw  RegexException on error
         */
        bool searchWithContext(std::string::const_iterator inputIterBegin,
                               std::string::const_iterator inputIterEnd,
                               std::smatch& match,
                               bool matchBeginning=true) const;
        
        //! The regex object
        std::regex mRegex;
#else
        // Internal function for passing flags to pcre_exec()
        std::string search(const std::string& matchString,
                           int startIndex, int flag);

        // Size of the output vector, must be a multiple of 3
        // The output vector is filled up to 2/3 (666) full for matches
        // so the maximum number of substrings is 333 (333 start
        // offsets and 333 end offsets)
        static const int mOvectorCount;

        //! The pcre object
        pcre* mPCRE;

        //! The output/offset vector
        std::vector<int> mOvector;
#endif
    };
}

#endif
