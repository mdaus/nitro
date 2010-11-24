/* =========================================================================
 * This file is part of cli-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2010, General Dynamics - Advanced Information Systems
 *
 * cli-c++ is free software; you can redistribute it and/or modify
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

#include "cli/Argument.h"
#include <iterator>

cli::Argument::Argument(std::string nameOrFlags, cli::ArgumentParser* parser) :
    mAction(cli::STORE), mMinArgs(0), mMaxArgs(1), mDefaultValue(NULL),
            mOwnDefault(false), mConstValue(NULL), mOwnConst(false),
            mParser(parser)
{
    std::vector < std::string > vars = str::split(nameOrFlags, " ");
    if (vars.size() == 1 && !str::startsWith(vars[0], "-"))
        mName = vars[0];
    else
    {
        for (std::vector<std::string>::iterator it = vars.begin(); it
                != vars.end(); ++it)
        {
            addFlag(*it);
        }
    }
}

cli::Argument::~Argument()
{
    if (mOwnDefault && mDefaultValue)
        delete mDefaultValue;
    if (mOwnConst && mConstValue)
        delete mConstValue;
}

cli::Argument* cli::Argument::addFlag(std::string flag)
{
    if (str::startsWith(flag, "--") && flag.size() > 2 && flag[2] != '-')
        mLongFlags.push_back(flag.substr(2));
    else if (str::startsWith(flag, "-") && flag.size() > 1 && flag[1] != '-')
        mShortFlags.push_back(flag.substr(1));
    return this;
}
cli::Argument* cli::Argument::setAction(cli::Action action)
{
    mAction = action;
    if (action == cli::STORE_TRUE || action == cli::STORE_FALSE || action
            == cli::STORE_CONST)
    {
        setMinArgs(0);
        setMaxArgs(0);
    }
    return this;
}
cli::Argument* cli::Argument::setMinArgs(int num)
{
    mMinArgs = num;
    return this;
}
cli::Argument* cli::Argument::setMaxArgs(int num)
{
    mMaxArgs = num;
    return this;
}
cli::Argument* cli::Argument::setDefault(Value* val, bool own)
{
    mDefaultValue = val;
    mOwnDefault = own;
    return this;
}
cli::Argument* cli::Argument::setChoices(std::vector<std::string> choices)
{
    mChoices.clear();
    mChoices = choices;
    return this;
}
cli::Argument* cli::Argument::addChoice(std::string choice)
{
    mChoices.push_back(choice);
    return this;
}
cli::Argument* cli::Argument::setHelp(std::string help)
{
    mHelp = help;
    return this;
}
cli::Argument* cli::Argument::setMetavar(std::string metavar)
{
    mMetavar = metavar;
    return this;
}
cli::Argument* cli::Argument::setDestination(std::string dest)
{
    mDestination = dest;
    return this;
}
cli::Argument* cli::Argument::setConst(Value* val, bool own)
{
    mConstValue = val;
    mOwnConst = own;
    return this;
}

std::string cli::Argument::getVariable() const
{
    if (!mDestination.empty())
        return mDestination;
    if (!mName.empty())
        return mName;
    if (!mLongFlags.empty())
        return mLongFlags[0];
    return mShortFlags[0];
}

bool cli::Argument::isPositional() const
{
    return mShortFlags.empty() && mLongFlags.empty();
}
