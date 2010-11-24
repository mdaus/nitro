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

#include "cli/ArgumentParser.h"

#define _MAX_ARG_LINE_LEN 21

namespace cli
{
void _writeArgumentHelp(io::OutputStream& out, const std::string heading,
                        size_t maxFlagsWidth,
                        const std::vector<std::string>& flags,
                        const std::vector<std::string>& helps)
{
    std::ostringstream s;
    out.writeln(heading);
    for (size_t i = 0, num = flags.size(); i < num; ++i)
    {
        s.str("");
        std::string flag = flags[i];
        std::string help = helps[i];
        if (flag.size() <= maxFlagsWidth)
        {
            s << "  ";
            s.width(maxFlagsWidth + 1);
            s << std::left << flag;
            s.width(0);
            s << help;
            out.writeln(s.str());
        }
        else
        {
            s << "  ";
            s << flag;
            out.writeln(s.str());
            s.str("");
            s.width(maxFlagsWidth + 3);
            s << " ";
            s.width(0);
            s << help;
            out.writeln(s.str());
        }
    }
}
}

cli::ArgumentParser::ArgumentParser() :
    mHelpEnabled(true), mPrefixChar('-')
{
}

cli::ArgumentParser::~ArgumentParser()
{
    for (std::vector<cli::Argument*>::iterator it = mArgs.begin(); it
            != mArgs.end(); ++it)
        if (*it)
            delete *it;
}

/**
 * Shortcut for adding an argument
 */
cli::Argument* cli::ArgumentParser::addArgument(std::string nameOrFlags,
                                                std::string help,
                                                cli::Action action,
                                                std::string dest,
                                                std::string metavar,
                                                int minArgs, int maxArgs)
{
    cli::Argument *arg = new cli::Argument(nameOrFlags, this);
    arg->setMinArgs(minArgs);
    arg->setMaxArgs(maxArgs);
    // set the action after setting num args, since action can modify args
    arg->setAction(action);
    arg->setDestination(dest);
    arg->setHelp(help);
    arg->setMetavar(metavar);
    mArgs.push_back(arg);

    if (arg->isPositional())
    {
        mPositionalArgs.push_back(arg);
    }
    else
    {
        const std::vector<std::string>& shortFlags = arg->getShortFlags();
        const std::vector<std::string>& longFlags = arg->getLongFlags();
        for (std::vector<std::string>::const_iterator it = shortFlags.begin(); it
                != shortFlags.end(); ++it)
        {
            std::string op = *it;
            if (mShortFlags.find(op) != mShortFlags.end())
                throw except::Exception(Ctxt(FmtX("Conflicting option: %c%s",
                                                  mPrefixChar, op.c_str())));
            mShortFlags[op] = arg;
        }
        for (std::vector<std::string>::const_iterator it = longFlags.begin(); it
                != longFlags.end(); ++it)
        {
            std::string op = *it;
            if (mLongFlags.find(op) != mLongFlags.end())
                throw except::Exception(Ctxt(FmtX("Conflicting option: %c%c%s",
                                                  mPrefixChar, mPrefixChar,
                                                  op.c_str())));
            mLongFlags[op] = arg;
        }
    }
    return arg;
}

cli::ArgumentParser& cli::ArgumentParser::setDescription(const std::string d)
{
    mDescription = d;
    return *this;
}

cli::ArgumentParser& cli::ArgumentParser::setEpilog(const std::string epilog)
{
    mEpilog = epilog;
    return *this;
}

cli::ArgumentParser& cli::ArgumentParser::setUsage(const std::string usage)
{
    mUsage = usage;
    return *this;
}

cli::ArgumentParser& cli::ArgumentParser::enableHelp(bool flag)
{
    mHelpEnabled = flag;
    return *this;
}

cli::ArgumentParser& cli::ArgumentParser::setProgram(const std::string program)
{
    mProgram = program;
    return *this;
}

void cli::ArgumentParser::printHelp(io::OutputStream& out, bool andExit) const
{
    std::vector<std::string> posFlags, opFlags, posHelps, opHelps, opUsage,
            posUsage;
    size_t maxFlagsWidth = 0;

    processFlags(posFlags, opFlags, posHelps, opHelps, opUsage, posUsage,
                 maxFlagsWidth);

    std::ostringstream s;
    s << "usage: ";
    if (mUsage.empty())
    {
        std::string progName = mProgram;
        s << (progName.empty() ? "program" : progName);
        if (!opUsage.empty())
            s << " " << str::join(opUsage, " ");
        if (!posUsage.empty())
            s << " " << str::join(posUsage, " ");
    }
    else
    {
        s << mUsage;
    }
    out.writeln(s.str());

    if (!mDescription.empty())
    {
        out.writeln("");
        out.writeln(mDescription);
    }

    if (posFlags.size() > 0)
    {
        out.writeln("");
        cli::_writeArgumentHelp(out, "positional arguments:", maxFlagsWidth,
                                posFlags, posHelps);
    }

    if (opFlags.size() > 0)
    {
        out.writeln("");
        cli::_writeArgumentHelp(out, "optional arguments:", maxFlagsWidth,
                                opFlags, opHelps);
    }

    if (!mEpilog.empty())
    {
        out.writeln("");
        out.writeln(mEpilog);
    }

    if (andExit)
    {
        exit(cli::EXIT_USAGE);
    }
}

void cli::ArgumentParser::printHelp(bool andExit) const
{
    io::StandardErrStream err;
    printHelp(err, andExit);
}

cli::Results* cli::ArgumentParser::parse(int argc, const char** argv)
{
    if (mProgram.empty() && argc > 0)
        setProgram(std::string(argv[0]));
    std::vector < std::string > args;
    for (int i = 1; i < argc; ++i)
        args.push_back(std::string(argv[i]));
    return parse(args);
}
cli::Results* cli::ArgumentParser::parse(const std::vector<std::string>& args)
{
    if (mProgram.empty())
        setProgram("program");

    std::vector < std::string > explodedArgs;
    // first, check for combined short options
    for (size_t i = 0, s = args.size(); i < s; ++i)
    {
        std::string argStr = args[i];
        if (argStr.size() > 1 && argStr[0] == mPrefixChar && argStr[1]
                != mPrefixChar)
        {
            std::string flag = argStr.substr(1);
            if (mShortFlags.find(flag) != mShortFlags.end())
            {
                explodedArgs.push_back(argStr);
            }
            else
            {
                //split up each char and make sure only the last can have args
                for (size_t j = 0, n = flag.size(); j < n; ++j)
                {
                    std::string charFlag = flag.substr(j, 1);
                    std::ostringstream oss;
                    oss << mPrefixChar << charFlag;
                    explodedArgs.push_back(oss.str());
                }
            }
        }
        else
        {
            explodedArgs.push_back(argStr);
        }
    }

    cli::Results *results = new Results;
    for (size_t i = 0, s = explodedArgs.size(); i < s; ++i)
    {
        std::string argStr = explodedArgs[i];
        cli::Argument *arg = NULL;
        if (argStr.size() > 2 && argStr[0] == mPrefixChar && argStr[1]
                == mPrefixChar)
        {
            std::string flag = argStr.substr(2);
            if (mLongFlags.find(flag) != mLongFlags.end())
            {
                arg = mLongFlags[flag];
            }
            else if (mHelpEnabled && flag == "help")
            {
                printHelp(true);
            }
            else
            {
                throw except::Exception(Ctxt(FmtX("Invalid option: [%s]",
                                                  argStr.c_str())));
            }
        }
        else if (argStr.size() > 1 && argStr[0] == mPrefixChar && argStr[1]
                != mPrefixChar)
        {
            std::string flag = argStr.substr(1);
            if (mShortFlags.find(flag) != mShortFlags.end())
            {
                arg = mShortFlags[flag];
            }
            else if (mHelpEnabled && flag == "h")
            {
                printHelp(true);
            }
            else
            {
                throw except::Exception(Ctxt(FmtX("Invalid option: [%s]",
                                                  argStr.c_str())));
            }
        }

        if (arg != NULL)
        {
            std::string argVar = arg->getVariable();
            switch (arg->getAction())
            {
            case cli::STORE:
            {
                cli::Value *v =
                        results->exists(argVar) ? results->getValue(argVar)
                                                : new cli::Value;
                int maxArgs = arg->getMaxArgs();
                // risky, I know...
                while (i < s - 1)
                {
                    std::string nextArg(explodedArgs[i + 1]);
                    if (nextArg.size() > 1 && nextArg[0] == mPrefixChar)
                    {
                        //it's another flag, so we have to break out
                        break;
                    }
                    if (maxArgs >= 0 && v->size() >= maxArgs)
                    {
                        parseError(FmtX("too many arguments: [%s]",
                                        argVar.c_str()));
                    }
                    v->add(nextArg);
                    ++i;
                }
                results->put(argVar, v);
                break;
            }
            case cli::STORE_TRUE:
                results->put(argVar, new cli::Value(true));
                break;
            case cli::STORE_FALSE:
                results->put(argVar, new cli::Value(false));
                break;
            case cli::STORE_CONST:
            {
                const Value* constVal = arg->getConst();
                results->put(argVar, constVal ? constVal->clone() : NULL);
                break;
            }
            case cli::VERSION:
                //TODO
                break;
            }
        }
        else
        {
            // it's a positional argument
            cli::Value *lastPosVal = NULL;
            for (std::vector<cli::Argument*>::iterator it =
                    mPositionalArgs.begin(); it != mPositionalArgs.end(); ++it)
            {
                cli::Argument *posArg = *it;
                std::string argVar = posArg->getVariable();
                int maxArgs = posArg->getMaxArgs();
                if (results->exists(argVar))
                {
                    cli::Value *posVal = lastPosVal = results->getValue(argVar);
                    if (posVal->size() >= maxArgs)
                        continue;
                    break;
                }
                else if (maxArgs != 0)
                {
                    lastPosVal = new cli::Value;
                    results->put(argVar, lastPosVal);
                    break;
                }
            }
            if (lastPosVal)
                lastPosVal->add(argStr);
            else
                parseError("too many arguments");
        }
    }

    // add the defaults
    for (std::vector<cli::Argument*>::const_iterator it = mArgs.begin(); it
            != mArgs.end(); ++it)
    {
        cli::Argument *arg = *it;
        std::string argVar = arg->getVariable();
        const Value* defaultVal = arg->getDefault();

        // also validate minArgs
        int minArgs = arg->getMinArgs();

        if (!results->exists(argVar))
        {
            if (defaultVal != NULL)
                results->put(argVar, defaultVal->clone());
            else if (arg->getAction() == cli::STORE_FALSE)
                results->put(argVar, new cli::Value(true));
            else if (arg->getAction() == cli::STORE_TRUE)
                results->put(argVar, new cli::Value(false));
        }

        //TODO validate choices

        if (minArgs > 0)
        {
            size_t numGiven =
                    results->exists(argVar) ? results->getValue(argVar)->size()
                                            : 0;
            if (numGiven < minArgs)
                parseError(FmtX("too few arguments: [%s]", argVar.c_str()));
        }
    }

    return results;
}

void cli::ArgumentParser::parseError(const std::string& msg)
{
    std::ostringstream s;
    s << "usage: ";
    if (mUsage.empty())
    {
        std::vector<std::string> posFlags, opFlags, posHelps, opHelps, opUsage,
                posUsage;
        size_t maxFlagsWidth = 0;

        processFlags(posFlags, opFlags, posHelps, opHelps, opUsage, posUsage,
                     maxFlagsWidth);

        std::string progName = mProgram;
        s << (progName.empty() ? "program" : progName);
        if (!opUsage.empty())
            s << " " << str::join(opUsage, " ");
        if (!posUsage.empty())
            s << " " << str::join(posUsage, " ");
    }
    else
        s << mUsage;
    s << "\n" << msg;
    throw except::ParseException(s.str());
}

void cli::ArgumentParser::processFlags(std::vector<std::string>& posFlags,
                                       std::vector<std::string>& opFlags,
                                       std::vector<std::string>&posHelps,
                                       std::vector<std::string>&opHelps,
                                       std::vector<std::string>&opUsage,
                                       std::vector<std::string>&posUsage,
                                       size_t& maxFlagsWidth) const
{
    std::ostringstream s;

    if (mHelpEnabled)
    {
        std::string helpMsg = FmtX("%ch, %c%chelp", mPrefixChar, mPrefixChar,
                                   mPrefixChar);
        maxFlagsWidth = std::max<size_t>(helpMsg.size(), maxFlagsWidth);
        opFlags.push_back(helpMsg);
        opHelps.push_back("show this help message and exit");
    }

    for (std::vector<cli::Argument*>::const_iterator it = mArgs.begin(); it
            != mArgs.end(); ++it)
    {
        cli::Argument *arg = *it;
        const std::string& argName = arg->getName();
        const std::vector<std::string>& argChoices = arg->getChoices();
        const std::string& argMetavar = arg->getMetavar();
        const std::string& argHelp = arg->getHelp();
        const cli::Value* defaultVal = arg->getDefault();

        s.str("");
        s << argHelp;
        if (defaultVal)
            s << " (default: " << defaultVal->toString() << ")";
        std::string helpMsg = s.str();

        s.str("");
        if (!argMetavar.empty())
            s << argMetavar;
        else if (!argChoices.empty())
            s << "{" << str::join(argChoices, ",") << "}";

        if (arg->isPositional())
        {
            std::string op = s.str();
            //positional argument
            if (op.empty())
                s << argName;
            op = s.str();
            maxFlagsWidth = std::max<size_t>(op.size(), maxFlagsWidth);
            posFlags.push_back(op);
            posHelps.push_back(helpMsg);
            posUsage.push_back(op);
        }
        std::string meta = s.str();
        std::vector < std::string > ops;
        const std::vector<std::string>& argShortFlags = arg->getShortFlags();
        const std::vector<std::string>& argLongFlags = arg->getLongFlags();
        for (size_t i = 0, n = argShortFlags.size(); i < n; ++i)
        {
            s.str("");
            s << mPrefixChar << argShortFlags[i];
            if (!meta.empty())
                s << " " << meta;
            ops.push_back(s.str());
        }
        for (size_t i = 0, n = argShortFlags.size(); i < n; ++i)
        {
            s.str("");
            s << mPrefixChar << mPrefixChar << argLongFlags[i];
            if (!meta.empty())
                s << " " << meta;
            ops.push_back(s.str());
        }
        if (!ops.empty())
        {
            s.str("");
            s << "[" << ops[0] << "]";
            opUsage.push_back(s.str());

            std::string opMsg = str::join(ops, ", ");
            maxFlagsWidth = std::max<size_t>(opMsg.size(), maxFlagsWidth);
            opFlags.push_back(opMsg);
            opHelps.push_back(helpMsg);
        }
    }
    maxFlagsWidth = std::min<size_t>(maxFlagsWidth, _MAX_ARG_LINE_LEN);
}

