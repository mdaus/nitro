/* =========================================================================
 * This file is part of io-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2019, MDA Information Systems LLC
 *
 * io-c++ is free software; you can redistribute it and/or modify
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

#include <string>
#include <clocale>

#include "io/StringStream.h"
#include <TestCase.h>

#include "xml/lite/MinidomParser.h"

static const std::string text("TEXT");
static const std::string strXml = "<root><doc><a>" + text + "</a></doc></root>";
static const std::string iso88591Text("T\xc9XT");  // ISO8859-1, "TÉXT"
static const std::string utf8Text("T\xc3\x89XT");  // UTF-8,  "TÉXT"
static const auto strUtf8Xml = "<root><doc><a>" + utf8Text + "</a></doc></root>";

TEST_CASE(testXmlParseSimple)
{
    io::StringStream ss;
    ss.stream() << strXml;
    TEST_ASSERT_EQ(ss.stream().str(), strXml);

    xml::lite::MinidomParser xmlParser;
    xmlParser.parse(ss);
    const auto doc = xmlParser.getDocument();
    TEST_ASSERT(doc != nullptr);
    const auto root = doc->getRootElement();
    TEST_ASSERT(root != nullptr);
    
    {
        const auto aElements = root->getElementsByTagName("a", true /*recurse*/);
        TEST_ASSERT_FALSE(aElements.empty());
        TEST_ASSERT_EQ(aElements.size(), 1);
        const auto& a = *(aElements[0]);
        const auto characterData = a.getCharacterData();
        TEST_ASSERT_EQ(characterData, text);
    }
    
    const auto docElements = root->getElementsByTagName("doc");
    TEST_ASSERT_FALSE(docElements.empty());
    TEST_ASSERT_EQ(docElements.size(), 1);
    {
        const auto aElements = docElements[0]->getElementsByTagName("a");
        TEST_ASSERT_FALSE(aElements.empty());
        TEST_ASSERT_EQ(aElements.size(), 1);
        const auto& a = *(aElements[0]);
        const auto characterData = a.getCharacterData();
        TEST_ASSERT_EQ(characterData, text);
    }
}

TEST_CASE(testXmlPreserveCharacterData)
{
    io::StringStream stream;
    stream.stream() << strUtf8Xml;
    TEST_ASSERT_EQ(stream.stream().str(), strUtf8Xml);

    xml::lite::MinidomParser xmlParser;
    // This is needed in Windows, because the default locale is *.1252 (more-or-less ISO8859-1)
    // Unfortunately, there doesn't seem to be a way of testing this ...
    // calling parse() w/o preserveCharacterData() throws ASSERTs, even after calling
    // _set_error_mode(_OUT_TO_STDERR) so there's no way to use TEST_EXCEPTION
    xmlParser.preserveCharacterData(true);
    xmlParser.parse(stream);
    TEST_ASSERT_TRUE(true);
}

TEST_CASE(testXmlUtf8Legacy)
{
    io::StringStream stream;
    stream.stream() << strUtf8Xml;
    TEST_ASSERT_EQ(stream.stream().str(), strUtf8Xml);

    xml::lite::MinidomParser xmlParser;
    xmlParser.preserveCharacterData(true);
    xmlParser.parse(stream);

    // This is LEGACY behavior, it is INCORRECT on Linux!
    const auto aElements =
            xmlParser.getDocument()->getRootElement()->getElementsByTagName("a", true /*recurse*/);
    TEST_ASSERT_EQ(aElements.size(), 1);
    const auto& a = *(aElements[0]);
    auto actual = a.getCharacterData();
    #ifdef _WIN32
    TEST_ASSERT_EQ(actual.length(), 4);
    TEST_ASSERT_EQ(actual, iso88591Text);
    #else
    TEST_ASSERT_EQ(actual.length(), 4);
    #endif

    const auto pEncoding = a.getEncoding();
    TEST_ASSERT_NULL(pEncoding);
}

TEST_CASE(testXmlUtf8)
{
    io::StringStream stream;
    stream.stream() << strUtf8Xml;
    TEST_ASSERT_EQ(stream.stream().str(), strUtf8Xml);

    xml::lite::MinidomParser xmlParser(true /*storeEncoding*/);
    xmlParser.preserveCharacterData(true);
    xmlParser.parse(stream);

    const auto aElements =
            xmlParser.getDocument()->getRootElement()->getElementsByTagName("a", true /*recurse*/);
    TEST_ASSERT_EQ(aElements.size(), 1);
    const auto& a = *(aElements[0]);
    const auto actual = a.getCharacterData();
    const auto pEncoding = a.getEncoding();
    TEST_ASSERT(pEncoding != nullptr);
#ifdef _WIN32
    TEST_ASSERT_EQ(actual, iso88591Text);
    TEST_ASSERT(*pEncoding == xml::lite::string_encoding::windows_1252);
#else
    TEST_ASSERT_EQ(actual, utf8Text);
    TEST_ASSERT(*pEncoding == xml::lite::string_encoding::utf_8);
#endif
}

TEST_CASE(testXmlPrintLegacy)
{
    io::StringStream input;
    input.stream() << strUtf8Xml;

    // This is LEGACY behavior, it is INCORRECT on Windows and won't even parse on Linux!
    {
        xml::lite::MinidomParser xmlParser;
        xmlParser.preserveCharacterData(true);
        xmlParser.parse(input);
        const auto pRootElement = xmlParser.getDocument()->getRootElement();

        io::StringStream output;
        pRootElement->print(output);
        const auto actual = output.stream().str();
        #ifdef _WIN32
        const auto strBadXml = "<root><doc><a>" + iso88591Text + "</a></doc></root>"; // XML must be UTF-8
        TEST_ASSERT_EQ(actual, strBadXml);
        #else
        const auto strBadXml = "<root><doc><a>"; // Failed to parse UTF-8
        TEST_ASSERT_EQ(actual.find(strBadXml), 0);
        #endif
    }
}

TEST_CASE(testXmlPrint)
{
    io::StringStream input;
    input.stream() << strUtf8Xml;

    xml::lite::MinidomParser xmlParser(true /*storeEncoding*/);
    xmlParser.preserveCharacterData(true);
    xmlParser.parse(input);
    const auto pRootElement = xmlParser.getDocument()->getRootElement();

    io::StringStream output;
    pRootElement->print(output, xml::lite::string_encoding::utf_8 /*write UTF-8*/);
    const auto actual = output.stream().str();
    TEST_ASSERT_EQ(actual, strUtf8Xml);
}

int main(int, char**)
{
    TEST_CHECK(testXmlParseSimple);
    TEST_CHECK(testXmlPreserveCharacterData);
    TEST_CHECK(testXmlUtf8Legacy);
    TEST_CHECK(testXmlUtf8);
    TEST_CHECK(testXmlPrintLegacy);
    TEST_CHECK(testXmlPrint);
}
