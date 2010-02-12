/* =========================================================================
 * This file is part of xml.lite-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * xml.lite-c++ is free software; you can redistribute it and/or modify
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

#include "xml/lite/Element.h"

/*==================================================*/
/* This marks the start of tree element code        */
/*==================================================*/

xml::lite::Element::Element(const xml::lite::Element & node)
{
    // Assign each member
    mName = node.mName;
    mCharacterData = node.mCharacterData;
    mAttributes = node.mAttributes;
    mChildren = node.mChildren;

}

xml::lite::Element & xml::lite::Element::operator=(const xml::lite::Element & node)
{
    if (this != &node)
    {
        mName = node.mName;
        mCharacterData = node.mCharacterData;
        mAttributes = node.mAttributes;
        mChildren = node.mChildren;
    }
    return *this;
}

void xml::lite::Element::clone(const xml::lite::Element & node)
{
    mName = node.mName;
    mCharacterData = node.mCharacterData;
    mAttributes = node.mAttributes;

    std::vector<xml::lite::Element *>::const_iterator iter;
    iter = node.getChildren().begin();
    for (; iter != node.getChildren().end(); ++iter)
    {
        xml::lite::Element *child = new xml::lite::Element();
        child->clone(**iter);
        this->addChild(child);
    }
}

bool xml::lite::Element::hasElement(const std::string & uri,
                                    const std::string & localName) const
{

    for (unsigned int i = 0; i < mChildren.size(); i++)
    {
        if (mChildren[i]->getUri() == uri && mChildren[i]->getLocalName()
                == localName)
            return true;
    }
    return false;
}

bool xml::lite::Element::hasElement(const std::string & localName) const
{

    for (unsigned int i = 0; i < mChildren.size(); i++)
    {
        if (mChildren[i]->getLocalName() == localName)
            return true;
    }
    return false;
}

void xml::lite::Element::getElementsByTagName(const std::string & uri,
                                              const std::string & localName,
                                              std::vector<Element *>&elements)
{
    elements.clear();
    for (unsigned int i = 0; i < mChildren.size(); i++)
    {
        if (mChildren[i]->getUri() == uri && mChildren[i]->getLocalName()
                == localName)
            elements.push_back(mChildren[i]);
    }
}

void xml::lite::Element::getElementsByTagName(const std::string & localName,
                                              std::vector<Element *>&elements)
{
    elements.clear();
    for (unsigned int i = 0; i < mChildren.size(); i++)
    {
        if (mChildren[i]->getLocalName() == localName)
            elements.push_back(mChildren[i]);
    }
}

void xml::lite::Element::getElementsByTagNameNS(const std::string & qname,
                                                std::vector<Element *>&elements)
{
    elements.clear();
    for (unsigned int i = 0; i < mChildren.size(); i++)
    {
        if (mChildren[i]->mName.toString() == qname)
            elements.push_back(mChildren[i]);
    }
}

void xml::lite::Element::destroyChildren()
{
    // While something is in vector
    while (mChildren.size())
    {
        // Get the last thing out
        xml::lite::Element * childAtBack = mChildren.back();
        // Pop it off
        mChildren.pop_back();
        // Delete it
        //EVAL(childAtBack->name);


        /* Added this line back in for 0.1.1 */
        //std::cout << "Deleting child at back" << std::endl;
        delete childAtBack;
    }
}

void xml::lite::Element::print(io::OutputStream & stream)
{
    depthPrint(stream, 0, "");
}

void xml::lite::Element::prettyPrint(io::OutputStream & stream,
                                     std::string formatter)
{
    depthPrint(stream, 0, formatter);
}

void xml::lite::Element::depthPrint(io::OutputStream & stream,
                                    int depth,
                                    std::string formatter)
{
    std::string prefix = "";
    for (int i = 0; i < depth; ++i)
        prefix += formatter;

    // Printing in XML form, recursively
    std::string lBrack = "<";
    std::string rBrack = ">";

    std::string acc = prefix + lBrack + mName.toString();

    for (int i = 0; i < mAttributes.getLength(); i++)
    {
        acc += std::string(" ");
        acc += mAttributes.getQName(i);
        acc += std::string("=\"");
        acc += mAttributes.getValue(i);
        acc += std::string("\"");
    }

    if (mCharacterData.empty() && mChildren.empty())
    {
        //simple type - just end it here
        stream.write(acc + "/" + rBrack);
    }
    else
    {
        stream.write(acc + rBrack);
        stream.write(mCharacterData);

        for (unsigned int i = 0; i < mChildren.size(); i++)
        {
            if (!formatter.empty())
                stream.write("\n");
            mChildren[i]->depthPrint(stream, depth + 1, formatter);
        }

        if (!mChildren.empty() && !formatter.empty())
        {
            stream.write("\n" + prefix);
        }

        lBrack += "/";
        stream.write(lBrack + mName.toString() + rBrack);
    }
}

void xml::lite::Element::addChild(xml::lite::Element * node)
{
    //       std::string(node->characterData.getData(),
    //                   node->characterData.getDataSize()) << std::endl;
    // End temp code
    mChildren.push_back(node);
}

void xml::lite::Element::changePrefix(Element* element, const std::pair<
        std::string, std::string> & prefixAndUri)
{

    if (element->mName.getAssociatedUri() == prefixAndUri.second)
    {
        //std::cout << "Got here" << std::endl;
        element->mName.setPrefix(prefixAndUri.first);

        for (int i = 0; i < mAttributes.getLength(); i++)
        {
            if (mAttributes[i].getUri() == prefixAndUri.second)
            {
                //std::cout << "Rewriting prefix in Atts!" << std::endl;
                mAttributes[i].setPrefix(prefixAndUri.first);
            }
        }
        for (unsigned int i = 0; i < mChildren.size(); i++)
        {
            changePrefix(element->mChildren[i], prefixAndUri);
        }
    }
}

void xml::lite::Element::changeUri(Element* element, const std::pair<
        std::string, std::string> & prefixAndUri)
{

    if (element->mName.getPrefix() == prefixAndUri.first)
    {
        //std::cout << "Got here" << std::endl;
        element->mName.setAssociatedUri(prefixAndUri.second);
        for (unsigned int i = 0; i < mChildren.size(); i++)
        {
            changeUri(element->mChildren[i], prefixAndUri);
            break;
        }
    }
}

void xml::lite::Element::rewriteNamespacePrefix(const std::pair<std::string,
        std::string> & prefixAndUri)
{
    for (int i = 0; i < mAttributes.getLength(); i++)
    {
        if (mAttributes[i].getValue() == prefixAndUri.second)
        {
            mAttributes[i].setLocalName(prefixAndUri.first);
            break;
        }
    }
    changePrefix(this, prefixAndUri);
}

void xml::lite::Element::rewriteNamespaceUri(const std::pair<std::string,
        std::string> & prefixAndUri)
{
    for (int i = 0; i < mAttributes.getLength(); i++)
    {
        if (mAttributes[i].getLocalName() == prefixAndUri.first)
        {
            std::cout << "Got here" << std::endl;
            mAttributes[i].setValue(prefixAndUri.second);
            break;
        }
    }
    changeUri(this, prefixAndUri);
}

