
#include <vector>
#include <iostream>
#include <string>

#include <import/nitf.hpp>

#include "TestCase.h"

static std::string testName;
const std::string output_file = "test_writer_3++.nitf";

namespace fs = std::filesystem;

static std::string argv0;
static fs::path findInputFile(const std::string& name)
{
    const fs::path inputFile = fs::path("modules") / "c++" / "nitf" / "unittests" / name;

    fs::path root;
    if (argv0.empty())
    {
        // running in Visual Studio
        root = fs::current_path().parent_path().parent_path();
    }
    else
    {
        root = fs::absolute(argv0).parent_path().parent_path().parent_path().parent_path();
        root = root.parent_path().parent_path();
    }

    return root / inputFile;
}

TEST_CASE(test_nitf_Record_unmergeTREs_crash)
{
    ::testName = testName;
    const std::string input_file = findInputFile("bug2_crash.ntf");

    nitf_Error error;
    nitf_IOHandle io = nitf_IOHandle_create(input_file.c_str(), NITF_ACCESS_READONLY,
        NITF_OPEN_EXISTING, &error);
    if (NITF_INVALID_HANDLE(io))
    {
        TEST_ASSERT_FALSE(true);
    }

    /*  We need to make a reader so we can parse the NITF */
    nitf_Reader* reader = nitf_Reader_construct(&error);
    TEST_ASSERT_NOT_EQ(nullptr, reader);

    /*  This parses all header data within the NITF  */
    nitf_Record* record = nitf_Reader_read(reader, io, &error);
    if (!record) goto CATCH_ERROR;

    /* Open the output IO Handle */
    nitf_IOHandle output = nitf_IOHandle_create("bug2_crash_out.ntf", NITF_ACCESS_WRITEONLY, NITF_CREATE, &error);
    if (NITF_INVALID_HANDLE(output)) goto CATCH_ERROR;

    nitf_Writer* writer = nitf_Writer_construct(&error);
    if (!writer) goto CATCH_ERROR;

    (void)nitf_Writer_prepare(writer, record, output, &error);

    nitf_IOHandle_close(io);
    nitf_Record_destruct(&record);
    nitf_Reader_destruct(&reader);

    TEST_ASSERT_TRUE(true);
    return;

CATCH_ERROR:
    TEST_ASSERT_FALSE(true);
}

TEST_CASE(test_nitf_Record_unmergeTREs_hangs)
{
    ::testName = testName;
    const std::string input_file = findInputFile("bug6_hangs.ntf");

    nitf_Error error;
    nitf_IOHandle io = nitf_IOHandle_create(input_file.c_str(), NITF_ACCESS_READONLY,
        NITF_OPEN_EXISTING, &error);
    if (NITF_INVALID_HANDLE(io))
    {
        TEST_ASSERT_FALSE(true);
    }

    /*  We need to make a reader so we can parse the NITF */
    nitf_Reader* reader = nitf_Reader_construct(&error);
    TEST_ASSERT_NOT_EQ(nullptr, reader);

    /*  This parses all header data within the NITF  */
    nitf_Record* record = nitf_Reader_read(reader, io, &error);
    if (!record) goto CATCH_ERROR;

    /* Open the output IO Handle */
    nitf_IOHandle output = nitf_IOHandle_create("bug6_hangs_out.ntf", NITF_ACCESS_WRITEONLY, NITF_CREATE, &error);
    if (NITF_INVALID_HANDLE(output)) goto CATCH_ERROR;

    nitf_Writer* writer = nitf_Writer_construct(&error);
    if (!writer) goto CATCH_ERROR;

    (void)nitf_Writer_prepare(writer, record, output, &error);

    nitf_IOHandle_close(io);
    nitf_Record_destruct(&record);
    nitf_Reader_destruct(&reader);

    TEST_ASSERT_TRUE(true);
    return;

CATCH_ERROR:
    TEST_ASSERT_FALSE(true);
}


TEST_CASE(test_readBandInfo_crash)
{
    ::testName = testName;
    const std::string input_file = findInputFile("bug4_crash.ntf");

    nitf_Error error;
    nitf_IOHandle io = nitf_IOHandle_create(input_file.c_str(), NITF_ACCESS_READONLY,
        NITF_OPEN_EXISTING, &error);
    if (NITF_INVALID_HANDLE(io))
    {
        TEST_ASSERT_FALSE(true);
    }

    /*  We need to make a reader so we can parse the NITF */
    nitf_Reader* reader = nitf_Reader_construct(&error);
    TEST_ASSERT_NOT_EQ(nullptr, reader);

    /*  This parses all header data within the NITF  */
    (void) nitf_Reader_read(reader, io, &error);
    TEST_ASSERT_TRUE(true);
}

TEST_MAIN(
    (void)argc;
argv0 = argv[0];

TEST_CHECK(test_nitf_Record_unmergeTREs_crash);
TEST_CHECK(test_nitf_Record_unmergeTREs_hangs);
TEST_CHECK(test_readBandInfo_crash);
)