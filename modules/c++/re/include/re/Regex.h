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


#ifndef __RE_PCRE_H__
#define __RE_PCRE_H__

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
 *  \brief C++ wrapper for the PCRE library
 */

namespace re
{

    typedef std::vector<std::string> RegexMatch;
    /*!
     *  \class Regex
     *  \brief C++ wrapper object for the PCRE library, 
     *  further documentation regarding the underlying C 
     *  library, especially for flag information, can 
     *  be found at http://www.pcre.org.
     */
    class Regex
    {
    public:
        /*!
         *  The default constructor
         */

        //#ifdef __CODA_CPP11
        enum Flag {
            Regex_NONE=0,
            Regex_DOTALL=1
        };
        //#endif

        Regex(const std::string& pattern = "", Flag flags = Regex_DOTALL);

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
         *  See the pcre man pages (pcre_compile) for 
         *  further info
         *  on the parameterized flags.
         *  \param pattern A pattern to match
         *  \param flags  Any flags to pass (default of Regex_DOTALL to
         *  match newlines with the .)
         *  \throw  RegexException on fatal error
         */
        Regex& compile(const std::string& pattern, Flag flags = Regex_DOTALL);

        /*!
         *  \todo Add non-const reference
         *  A const reference return for the pattern
         *  \return the pattern
         */
        const std::string& getPattern() const;

        /*!
         *  Match this input string against our pattern and populate the
         *  data structure
         *  See the pcre man pages (pcre_exec) for further 
         *  info on the optional flags.
         *  \param str The string to match against our pattern
         *  \param matches RegexMatch container to fill
         *  \param flags  Any flags
         *  \return  True on success, False otherwise
         *  \throw  RegexException on fatal error
         */
        bool match(const std::string& str,
                   RegexMatch& matchObject,
                   int flags = 0);

        bool matches(const std::string& str, int flags = 0) const;

        /*!
         *  Search the matchString
         *  See the pcre man pages (pcre_exec) for further
         *  info on the optional flags.
         *  \param matchString The string to try and match
         *  \param startIndex Starting where?
         *  \param flags  Any flags
         *  \return  Matched substring
         *  \throw  RegexException on fatal error
         */
        std::string search(const std::string& matchString,
                           int startIndex = 0,
                           int flags = 0);

        /*!
         *  Search the matchString and get the sub-expressions, by ref
         *  \param matchString The string to match
         *  \param v The sub-expression
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
         *  Replace non-escaped "." with "[\s\S]" to get Regex_DOTALL newline behavior
         *  \param str  The string to modify
         *  \return  The modified string
         */
        std::string replaceDot(const std::string& str) const;

        std::regex mRegex;
#else
        // Size of the output vector, must be a multiple of 3
        // The output vector is filled up to 2/3 (666) full for matches
        // so the maximum number of substrings is 333 (333 start
        // offsets and 333 end offsets)
        const int OVECCOUNT;

        //! The pcre object
        pcre* mPCRE;

        //! The output/offset vector
        std::vector<int> mOvector;
#endif
    };
}

#endif
