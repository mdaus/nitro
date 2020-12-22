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

#include "io/StringStream.h"
#include <TestCase.h>

#include "xml/lite/MinidomParser.h"

static const std::string text("TEXT");
static const std::string strXml = "<root><doc><a>" + text + "</a><b/><b/></doc></root>";

struct test_MinidomParser final
{
    xml::lite::MinidomParser xmlParser;
    const xml::lite::Element* getRootElement()
    {
        io::StringStream ss;
        ss.stream() << strXml;

        xmlParser.parse(ss);
        const auto doc = xmlParser.getDocument();
        return doc->getRootElement();    
    }
};

TEST_CASE(test_getRootElement)
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
}

TEST_CASE(test_getElementsByTagName)
{
    test_MinidomParser xmlParser;
    const auto root = xmlParser.getRootElement();
    
    {
        const auto aElements = root->getElementsByTagName("a", true /*recurse*/);
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
        TEST_ASSERT_EQ(aElements.size(), 1);
        const auto& a = *(aElements[0]);

        const auto characterData = a.getCharacterData();
        TEST_ASSERT_EQ(characterData, text);
        const auto pEncoding = a.getEncoding();
        TEST_ASSERT_NULL(pEncoding);
    }
}

TEST_CASE(test_getElementsByTagName_b)
{
    test_MinidomParser xmlParser;
    const auto root = xmlParser.getRootElement();

    {
        const auto bElements = root->getElementsByTagName("b", true /*recurse*/);
        TEST_ASSERT_EQ(bElements.size(), 2);
        const auto& b = *(bElements[0]);

        const auto characterData = b.getCharacterData();
        TEST_ASSERT_TRUE(characterData.empty());
    }

    const auto docElements = root->getElementsByTagName("doc");
    TEST_ASSERT_FALSE(docElements.empty());
    TEST_ASSERT_EQ(docElements.size(), 1);
    {
        const auto bElements = docElements[0]->getElementsByTagName("b");
        TEST_ASSERT_EQ(bElements.size(), 2);
        const auto& b = *(bElements[0]);

        const auto characterData = b.getCharacterData();
        TEST_ASSERT_TRUE(characterData.empty());
    }
}

TEST_CASE(test_getElementByTagName)
{
    test_MinidomParser xmlParser;
    const auto root = xmlParser.getRootElement();

    {
        const auto& a = root->getElementByTagName("a", true /*recurse*/);
        const auto characterData = a.getCharacterData();
        TEST_ASSERT_EQ(characterData, text);
    }

    const auto& doc = root->getElementByTagName("doc");
    {
        const auto& a = doc.getElementByTagName("a");
        const auto characterData = a.getCharacterData();
        TEST_ASSERT_EQ(characterData, text);
    }
}

TEST_CASE(test_getElementByTagName_nothrow)
{
    test_MinidomParser xmlParser;
    const auto root = xmlParser.getRootElement();

    {
        const auto pNotFound = root->getElementByTagName(std::nothrow, "not_found", true /*recurse*/);
        TEST_ASSERT_NULL(pNotFound);
    }

    const auto& doc = root->getElementByTagName("doc");
    {
        const auto pNotFound = doc.getElementByTagName(std::nothrow, "not_found");
        TEST_ASSERT_NULL(pNotFound);
    }
}

TEST_CASE(test_getElementByTagName_b)
{
    test_MinidomParser xmlParser;
    const auto root = xmlParser.getRootElement();
    
    TEST_SPECIFIC_EXCEPTION(root->getElementByTagName("b", true /*recurse*/), xml::lite::XMLException);

    const auto& doc = root->getElementByTagName("doc");
    TEST_SPECIFIC_EXCEPTION(doc.getElementByTagName("b"), xml::lite::XMLException);
}

int main(int, char**)
{
    TEST_CHECK(test_getRootElement);
    TEST_CHECK(test_getElementsByTagName);
    TEST_CHECK(test_getElementsByTagName_b);
    TEST_CHECK(test_getElementByTagName);
    TEST_CHECK(test_getElementByTagName_nothrow);
    TEST_CHECK(test_getElementByTagName_b);
}
