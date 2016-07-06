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

#include <import/re.h>
#include "TestCase.h"
#include <map>

TEST_CASE(testCompile)
{
    re::Regex rx;
    // test that an invalid regexp throws an exception
    TEST_THROWS(rx.compile("^("));

    // test that a valid regexp compiles
    try
    {
        rx.compile("^(foo)");
    }
    catch (...)
    {
        TEST_FAIL("Compiling a valid regexp should not have thrown exception!");
    }
}

TEST_CASE(testMatches)
{
    re::RegexMatch matches;
    re::Regex rx("^([^:]+):[ ]*([^\r\n]+)\r\n(.*)");
    TEST_ASSERT(rx.match("Proxy-Connection: Keep-Alive\r\n", matches));
    TEST_ASSERT_EQ(matches.size(), 4);
}

TEST_CASE(testSearch)
{
    re::Regex rx("jud");
    std::string result = rx.search("arabsdsarbjudarc34ardnjfsdveqvare3arfarg");
    TEST_ASSERT_EQ(result, "jud");
}

TEST_CASE(testSearchAll)
{
    re::RegexMatch matches;
    re::Regex rx("ar");
    rx.searchAll("arabsdsarbjudarc34ardnjfsdveqvare3arfarg", matches);
    TEST_ASSERT_EQ(matches.size(), 7);
}

TEST_CASE(testDotAllFlag)
{
    // This should match both the "3.3" and "4\n2"
    re::RegexMatch matches1;
    re::Regex rx1("\\d.\\d");
    rx1.searchAll("3.3 4\n2", matches1);
    TEST_ASSERT_EQ(matches1.size(), 2);

    // This should only match the "3.3" if the replace_dot() function
    // is working correctly
    re::RegexMatch matches3;
    re::Regex rx3("\\d\\.\\d");
    rx3.searchAll("3.3 4\n2", matches3);
    TEST_ASSERT_EQ(matches3.size(), 1);
}

TEST_CASE(testMultilineBehavior)
{
    re::RegexMatch matches;
    re::Regex rx;
    std::string inputString = 
        "3.3 4\n2\nx\r\ns\r\n;sjf sfkgsdkie\n shfihfoisu\nha hosd\nhvfoef\n";

    // This should match just the beginning
    rx.compile("^.");
    rx.searchAll(inputString, matches);
    TEST_ASSERT_EQ(matches.size(), 1);

    // This should match nothing
    matches.clear();
    rx.compile("^.$");
    TEST_ASSERT_FALSE(rx.match(inputString, matches));
    TEST_ASSERT_EQ(matches.size(), 0);

    // This should match the whole inputString
    matches.clear();
    rx.compile("^.*$");
    TEST_ASSERT_TRUE(rx.match(inputString, matches));
    TEST_ASSERT_EQ(matches.size(), 1);
    TEST_ASSERT_EQ(matches[0].length(), inputString.length());

#ifdef __CODA_CPP11
    // These exercise our limitations and should all throw exceptions (sigh)
    matches.clear();
    rx.compile(".$");
    TEST_EXCEPTION(rx.match(inputString, matches));

    matches.clear();
    rx.compile("foo^bar");
    TEST_EXCEPTION(rx.match(inputString, matches));

    matches.clear();
    rx.compile("^foo$bar");
    TEST_EXCEPTION(rx.match(inputString, matches));
#endif
}

TEST_CASE(testSub)
{
    re::RegexMatch matches;
    re::Regex rx("ar");
    std::string subst = rx.sub("Hearo", "ll");
    TEST_ASSERT_EQ(subst, "Hello");
    subst = rx.sub("Hearo Keary!", "ll");
    TEST_ASSERT_EQ(subst, "Hello Kelly!");
}

TEST_CASE(testSplit)
{
    re::RegexMatch matches;
    re::Regex rx("ar");
    std::vector<std::string> vec;
    rx.split("ONEarTWOarTHREE", vec);
    TEST_ASSERT_EQ(vec.size(), 3);
    TEST_ASSERT_EQ(vec[0], "ONE");
    TEST_ASSERT_EQ(vec[1], "TWO");
    TEST_ASSERT_EQ(vec[2], "THREE");
}

// This was copied out of re/tests/RegexTest3.cpp
TEST_CASE(testHttpResponse)
{
    const char
        *request =
        "GET http://pluto.beseen.com:1113 HTTP/1.0\r\nProxy-Connection: Keep-Alive\r\nUser-Agent: Mozilla/4.75 [en] (X11; U; SunOS 5.6 sun4u)\r\nAccept: image/gif, image/x-xbitmap, image/jpeg, image/pjpeg, image/png, */*\r\nAccept-Encoding: gzip\r\nAccept-Language: en\r\nAccept-Charset: iso-8859-1,*,utf-8\r\nContent-Type: application/x-www-form-urlencoded\r\nContent-Length: 96\r\n\r\n";

    class HttpParser
    {
    public:

        HttpParser()
        {
            mMatchRequest.compile(
                "^([^ ]+) (http:[^ ]+) HTTP/([0-9]+\\.[0-9]+)\r\n(.*)");
            mMatchPair.compile("^([^:]+):[ ]*([^\r\n]+)\r\n(.*)");
            mMatchEndOfHeader.compile("^\r\n");
            mMatchResponse.compile("^HTTP/([^ ]+) ([^\r\n]+)\r\n(.*)");
        }

        void parse(const char* header, size_t length)
        {
            mHeader = std::string(header, length);
            if (!parseRequest())
                if (!parseResponse())
                    assert(0);
        }

        void parseRest(const std::string& restOfChunk)
        {
            std::string rest = restOfChunk;

            re::RegexMatch matches;
            while (!mMatchEndOfHeader.match(rest, matches))
            {
                re::RegexMatch keyVals;
                if (mMatchPair.match(rest, keyVals))
                {
                    mKeyValuePair[keyVals[1]] = keyVals[2];

                    rest = keyVals[3];
                }
                else
                {
                    std::cout << "'rest' doesn't match." << std::endl;
                }
            }
        }

        bool parseResponse()
        {
            re::RegexMatch responseVals;
            if (mMatchResponse.match(mHeader, responseVals))
            {
                mVersion = responseVals[1];
                mReturnVal = responseVals[2];

                parseRest(responseVals[3]);
                return true;
            }
            else
            {
                return false;
            }
        }

        bool parseRequest()
        {
            re::RegexMatch requestVals;
            if (mMatchRequest.match(mHeader, requestVals))
            {
                mMethod = requestVals[1];
                mUrl = requestVals[2];
                mVersion = requestVals[3];

                parseRest(requestVals[4]);
                return true;
            }
            else
            {
                return false;
            }
        }

        std::string getReturnVal()
        {
            return mReturnVal;
        }
        std::string getUrl()
        {
            return mUrl;
        }
        std::string getVersion()
        {
            return mVersion;
        }
        std::string getMethod()
        {
            return mMethod;
        }

        std::string getContentType()
        {
            std::string key = "Content-Type";
            return getAssociatedValue(key);
        }
        std::string getContentLength()
        {
            std::string key = "Content-Length";
            return getAssociatedValue(key);
        }

        std::string getAssociatedValue(const std::string& key)
        {
            std::map<std::string, std::string>::const_iterator p = mKeyValuePair.find(key);
            if (p == mKeyValuePair.end())
            {
                return std::string("");
            }
            return mKeyValuePair[key];
        }

    protected:
        re::Regex mMatchRequest;
        re::Regex mMatchPair;
        re::Regex mMatchEndOfHeader;
        re::Regex mMatchResponse;
        std::map<std::string, std::string>mKeyValuePair;

        std::string mReturnVal;
        std::string mUrl;
        std::string mVersion;
        std::string mMethod;
        std::string mHeader;
    };

    HttpParser p;
    p.parse(request, strlen(request));

    TEST_ASSERT_EQ(p.getReturnVal(), "");
    TEST_ASSERT_EQ(p.getMethod(), "GET");
    TEST_ASSERT_EQ(p.getUrl(), "http://pluto.beseen.com:1113");
    TEST_ASSERT_EQ(p.getVersion(), "1.0");
    TEST_ASSERT_EQ(p.getAssociatedValue("User-Agent"), "Mozilla/4.75 [en] (X11; U; SunOS 5.6 sun4u)");
    TEST_ASSERT_EQ(p.getAssociatedValue("Accept-Encoding"), "gzip");
    TEST_ASSERT_EQ(p.getContentType(), "application/x-www-form-urlencoded");
    TEST_ASSERT_EQ(p.getContentLength(), "96");
}

int main(int, char**)
{
    TEST_CHECK( testCompile);
    TEST_CHECK( testMatches);
    TEST_CHECK( testSearch);
    TEST_CHECK( testSearchAll);
    TEST_CHECK( testDotAllFlag);
    TEST_CHECK( testMultilineBehavior);
    TEST_CHECK( testSub);
    TEST_CHECK( testSplit);
    TEST_CHECK( testHttpResponse);
    return 0;
}
