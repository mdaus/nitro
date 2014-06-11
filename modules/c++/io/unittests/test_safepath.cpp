#include <iostream>
#include <fstream>
#include <import/sys.h>
#include <io/FileOutputStreamOS.h>
#include <import/except.h>
#include <io/SafePath.h>
#include <sys/File.h>

#include <TestCase.h>

namespace
{

TEST_CASE(TempFileTest)
{
    try
    {
        sys::OS os;
        std::string realFileName = "that.txt";
        std::string tempFileName;
        std::string fileContents = "hello, world. please enjoy my test message.";
        {
            io::SafePath temp(realFileName);
            tempFileName = temp.getTempPathname();
            io::FileOutputStreamOS fout(temp.getTempPathname(), sys::File::CREATE);
            fout.write(fileContents);
            fout.close();
            temp.moveFile();
            temp.moveFile();
        }
        TEST_ASSERT(os.exists(realFileName));
        TEST_ASSERT(!os.exists(tempFileName));

        std::ifstream fin;
        fin.open(realFileName.c_str());
        std::string input;
        std::getline(fin, input);
        fin.close();

        TEST_ASSERT_EQ(input, fileContents);

        os.remove(realFileName);
    }
    catch (except::Exception& e)
    {
        std::cout << "Caught exception: " << e.getMessage() << std::endl;
        TEST_ASSERT(false);
    }
    catch (...)
    {
        std::cout << "Caught unnamed exception" << std::endl;
        TEST_ASSERT(false);
    }
}

}

int main(int argc, char* argv[])
{
    TEST_CHECK(TempFileTest);

    return 0;
}
