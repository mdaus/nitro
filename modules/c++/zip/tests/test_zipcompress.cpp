#include <string>
#include <vector>
#include <zip/ZipOutputStream.h>
#include <io/FileInputStream.h>

namespace
{
void usage(const std::string& progname)
{
    std::cerr << "Usage: " << progname
        << " -i <file #1> <zipDir #1> <file#2> <zipDir #2> ..."
        << " [-o Output filename]";
}
}

int main(int argc, char** argv)
{
    try
    {
        std::string outputFile;
        std::vector<std::string> inputFiles;
        std::vector<std::string> zipDirectories;

        size_t index = 1;
        while (index < argc)
        {
            if (!::strcmp(argv[index], "-i"))
            {
                ++index;
                while (index < argc - 1)
                {
                    if ((::strlen(argv[index]) > 0 && argv[index][0] == '-') ||
                            (::strlen(argv[index + 1]) > 0 && argv[index + 1][0] == '-'))
                    {
                        break;
                    }
                    inputFiles.push_back(argv[index++]);
                    zipDirectories.push_back(argv[index++]);
                }
            }
            else if (!::strcmp(argv[index], "-o") && index + 1 < argc)
            {
                outputFile = argv[index + 1];
                index += 2;
            }
            else
            {
                usage(argv[0]);
                return 1;
            }
        }

        zip::ZipOutputStream output(outputFile);
        for (size_t ii = 0; ii < inputFiles.size(); ++ii)
        {
            output.write(inputFiles[ii], zipDirectories[ii]);
        }
        output.close();
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Caught std::exception: " << ex.what() << std::endl;
        return 1;
    }
    catch (const except::Exception& ex)
    {
        std::cerr << "Caught except::exception: " << ex.getMessage()
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Caught unknown exception\n";
        return 1;
    }
}
