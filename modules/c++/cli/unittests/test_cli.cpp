/* =========================================================================
 * This file is part of cli-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
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

#include <import/cli.h>
#include <import/mem.h>
#include "TestCase.h"
#include <sstream>
#include <fstream>
#include <stdio.h>

TEST_CASE(testValue)
{
    cli::Value v("data");
    TEST_ASSERT_EQ("data", v.get<std::string>());

    v.set(3.14f);
    TEST_ASSERT_ALMOST_EQ(3.14f, v.get<float>());
    TEST_ASSERT_EQ(3, v.get<int>());

    std::vector<float> floats;
    std::vector<std::string> strings;
    for(int i = 0; i < 10; ++i)
    {
        floats.push_back(10.0f * i);
        strings.push_back(str::toString(i));
    }

    // floats
    v.setContainer(floats);
    for(int i = 0; i < 10; ++i)
    {
        TEST_ASSERT_ALMOST_EQ(v.at<float>(i), 10.0f * i);
    }
    TEST_ASSERT_EQ(v.size(), 10);

    // strings
    v.setContainer(strings);
    for(int i = 0; i < 10; ++i)
    {
        TEST_ASSERT_EQ(v.at<std::string>(i), str::toString(i));
    }
    TEST_ASSERT_EQ(v.size(), 10);
}

TEST_CASE(testChoices)
{
    cli::ArgumentParser parser;
    parser.setProgram("tester");
    parser.addArgument("-v --verbose", "Toggle verbose", cli::STORE_TRUE);
    parser.addArgument("-t --type", "Specify a type to use", cli::STORE)->addChoice(
            "type1")->addChoice("type2")->addChoice("type3");
    parser.addArgument("-m --many", "Specify a type to use", cli::STORE, "choices", "CHOICES", 0)->addChoice(
            "type1")->addChoice("type2")->addChoice("type3");
    parser.addArgument("images", "Input images", cli::STORE);
    parser.setDescription("This program is kind of pointless, but have fun!");
    parser.setProlog("========= (c) COPYRIGHT BANNER ========= ");
    parser.setEpilog("And that's the usage of the program!");
    std::ostringstream buf;
    parser.printHelp(buf);

    mem::auto_ptr<cli::Results> results(parser.parse(str::split("-v", " ")));
    TEST_ASSERT(results->hasValue("verbose"));
    TEST_ASSERT(results->get<bool>("verbose", 0));

    results.reset(parser.parse(str::split("", " ")));
    TEST_ASSERT_EQ(results->get<bool>("verbose", 0), false);

    results.reset(parser.parse(str::split("-t type2", " ")));
    TEST_ASSERT_EQ(results->get<std::string>("type", 0), std::string("type2"));

    try
    {
        results.reset(parser.parse(str::split("-t type2 -t type1", " ")));
        TEST_FAIL("Shouldn't allow multiple types");
    }
    catch(except::Exception&)
    {
    }
    results.reset(parser.parse(str::split("-t type2", " ")));

    results.reset(parser.parse(str::split("-m type2 --many type1 -m type3", " ")));
}

TEST_CASE(testMultiple)
{
    cli::ArgumentParser parser;
    parser.setProgram("tester");
    parser.addArgument("-v --verbose --loud -l", "Toggle verbose", cli::STORE_TRUE);

    mem::auto_ptr<cli::Results> results(parser.parse(str::split("-v")));
    TEST_ASSERT(results->hasValue("verbose"));
    TEST_ASSERT(results->get<bool>("verbose"));

    results.reset(parser.parse(str::split("-l")));
    TEST_ASSERT(results->get<bool>("verbose"));
    results.reset(parser.parse(str::split("--loud")));
    TEST_ASSERT(results->get<bool>("verbose"));
    results.reset(parser.parse(str::split("")));
    TEST_ASSERT_FALSE(results->get<bool>("verbose"));
}

TEST_CASE(testSubOptions)
{
    cli::ArgumentParser parser;
    parser.setProgram("tester");
    parser.addArgument("-v --verbose", "Toggle verbose", cli::STORE_TRUE);
    parser.addArgument("-c --config", "Specify a config file", cli::STORE);
    parser.addArgument("-x --extra", "Extra options", cli::SUB_OPTIONS);
    parser.addArgument("-c --config", "Config options", cli::SUB_OPTIONS);
    std::ostringstream buf;
    parser.printHelp(buf);

    mem::auto_ptr<cli::Results> results(parser.parse(str::split("-x:special")));
    TEST_ASSERT(results->hasSubResults("extra"));
    TEST_ASSERT(results->getSubResults("extra")->get<bool>("special"));

    results.reset(parser.parse(str::split("--extra:arg=something -x:arg2 1")));
    TEST_ASSERT(results->hasSubResults("extra"));
    TEST_ASSERT_EQ(results->getSubResults("extra")->get<std::string>("arg"), "something");
    TEST_ASSERT_EQ(results->getSubResults("extra")->get<int>("arg2"), 1);

    results.reset(parser.parse(str::split("--config /path/to/file --config:flag1 -c:flag2=true --config:flag3 false")));
    TEST_ASSERT_EQ(results->get<std::string>("config"), "/path/to/file");
    TEST_ASSERT(results->hasSubResults("config"));
    TEST_ASSERT(results->getSubResults("config")->get<bool>("flag1"));
    TEST_ASSERT(results->getSubResults("config")->get<bool>("flag2"));
    TEST_ASSERT_FALSE(results->getSubResults("config")->get<bool>("flag3"));
}

TEST_CASE(testIterate)
{
    cli::ArgumentParser parser;
    parser.setProgram("tester");
    parser.addArgument("-v --verbose", "Toggle verbose", cli::STORE_TRUE);
    parser.addArgument("-c --config", "Specify a config file", cli::STORE);

    mem::auto_ptr<cli::Results>
            results(parser.parse(str::split("-v -c config.xml")));
    std::vector<std::string> keys;
    for(cli::Results::const_iterator it = results->begin(); it != results->end(); ++it)
        keys.push_back(it->first);
    TEST_ASSERT_EQ(keys.size(), 2);
    // std::map returns keys in alphabetical order...
    TEST_ASSERT_EQ(keys[0], "config");
    TEST_ASSERT_EQ(keys[1], "verbose");
}

TEST_CASE(testRequired)
{
    cli::ArgumentParser parser;
    parser.setProgram("tester");
    parser.addArgument("-v --verbose", "Toggle verbose", cli::STORE_TRUE);
    parser.addArgument("-c --config", "Specify a config file", cli::STORE)->setRequired(true);

    mem::auto_ptr<cli::Results> results;
    TEST_EXCEPTION(results.reset(parser.parse(str::split(""))));
    TEST_EXCEPTION(results.reset(parser.parse(str::split("-c"))));
    results.reset(parser.parse(str::split("-c configFile")));
    TEST_ASSERT_EQ(results->get<std::string>("config"), "configFile");
}

TEST_CASE(testUnknownArgumentsOptions)
{
    cli::ArgumentParser parser(true, &std::cerr);
    parser.setProgram("tester");
    parser.addArgument("-v --verbose", "Toggle verbose", cli::STORE_TRUE);
    parser.addArgument("-x --extra", "Extra options", cli::SUB_OPTIONS);

    // Use a flag that is incorrect
    mem::auto_ptr<cli::Results> results(parser.parse(str::split("-z", " ")));

    TEST_ASSERT_FALSE(results->get<bool>("verbose"));

    // Set the output stream to "/dev/null"
    std::ostringstream outStream("/dev/null");
    parser.setIgnoreUnknownArgumentsOutputStream(&outStream);
    results.reset(parser.parse(str::split("-z", " ")));
    TEST_ASSERT_FALSE(results->get<bool>("verbose"));

    // Test a file
    std::string testFilename = "test_failed_parser_arg.log";
    std::ofstream outFStream(testFilename);
    parser.setIgnoreUnknownArgumentsOutputStream(&outFStream);
    results.reset(parser.parse(str::split("-z", " ")));
    outFStream.close();
    // Open the file and make sure it has the appropriate line
    std::ifstream inFStream(testFilename);
    std::string line;
    if (inFStream.is_open())
    {
        std::getline(inFStream, line);
        TEST_ASSERT(line.compare("Unknown arg: -z") == 0);
    }
    // Close the stream and remove the file.
    inFStream.close();
    if (remove(testFilename.c_str()) != 0)
    {
        std::cerr << "Error deleting file: " << testFilename << std::endl;
    }

    // Test setting flag
    parser.setIgnoreUnknownArgumentsFlag(false);
    TEST_EXCEPTION(results.reset(parser.parse(str::split("-z", " "))));

    // Test default with more complex arguments
    cli::ArgumentParser parser2;
    parser2.setProgram("tester");
    parser2.addArgument("-v --verbose", "Toggle verbose", cli::STORE_TRUE);
    TEST_EXCEPTION(results.reset(parser2.parse(str::split("-f", "C:/Data/File.txt"))));

    // Test using one parameter
    // Note that if only the ostream is given it will be evaluated as
    // true and set the ignore flag as such
    cli::ArgumentParser parser3(&std::cout);
    parser3.setProgram("tester");
    parser3.addArgument("-t --type", "Type", cli::STORE_TRUE);
    results.reset(parser3.parse(str::split("--filename", "C:/Data/File.txt")));
    TEST_ASSERT_FALSE(results->get<bool>("type"));

    cli::ArgumentParser parser4(true);
    parser4.setProgram("tester");
    parser4.addArgument("-t --type", "Type", cli::STORE_TRUE);
    results.reset(
            parser3.parse(str::split("--outputFile", "C:/Data/File.txt")));
    TEST_ASSERT_FALSE(results->get<bool>("type"));

    // Verify that cerr did not get messed up
    std::cerr << "cerr is still working as expected" << std::endl;
}

int main(int, char**)
{
    TEST_CHECK( testValue);
    TEST_CHECK( testChoices);
    TEST_CHECK( testMultiple);
    TEST_CHECK( testSubOptions);
    TEST_CHECK( testIterate);
    TEST_CHECK( testRequired);
    TEST_CHECK( testUnknownArgumentsOptions);
}
