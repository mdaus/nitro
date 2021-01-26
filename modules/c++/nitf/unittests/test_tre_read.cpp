
#include <vector>
#include <iostream>
#include <string>

#include <import/nitf.hpp>

#include "TestCase.h"

static std::string testName;
const std::string output_file = "test_writer_3++.nitf";

namespace fs = std::filesystem;

static std::string argv0;
static fs::path findInputFile()
{
    const fs::path inputFile = fs::path("modules") / "c++" / "nitf" / "unittests" / "bug2_crash.ntf";

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

static nitf::Record doRead(const std::string& inFile, nitf::Reader& reader)
{
    // Check that wew have a valid NITF
    const auto version = nitf::Reader::getNITFVersion(inFile);
    TEST_ASSERT(version != NITF_VER_UNKNOWN);

    nitf::IOHandle io(inFile);
    nitf::Record record = reader.read(io);

    /*  Set this to the end, so we'll know when we're done!  */
    nitf::ListIterator end = record.getImages().end();
    nitf::ListIterator iter = record.getImages().begin();
    for (int count = 0, numImages = record.getHeader().getNumImages();
        count < numImages && iter != end; ++count, ++iter)
    {
        nitf::ImageSegment imageSegment = *iter;
        nitf::ImageReader deserializer = reader.newImageReader(count);
    }

    return record;
}
TEST_CASE(test_read_tre)
{
    ::testName = testName;
    const auto input_file = findInputFile().string();

    nitf::Reader reader;
    nitf::Record record = doRead(input_file, reader);
}

TEST_MAIN(
    (void)argc;
    argv0 = argv[0];

    TEST_CHECK(test_read_tre);
    )