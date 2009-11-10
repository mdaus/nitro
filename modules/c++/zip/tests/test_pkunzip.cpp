#include <import/sys.h>
#include <import/io.h>
#include "zip/ZipFile.h"

int main(int argc, char** argv)
{
    if (argc != 3)
        die_printf("Usage: %s <gz-file> <out-file>\n", argv[0]);

    try
    {
        std::string inputName(argv[1]);
        std::string outputName(argv[2]);

        std::cout << "Attempting to unzip: " 
                  << std::endl << "\tInput: " << inputName << std::endl
                  << "\tTarget: " << outputName << std::endl;;

	io::FileInputStream input(inputName);
	zip::ZipFile zipFile(&input);

	std::cout << zipFile << std::endl;
// 	for (zip::ZipFile::Iterator p = zipFile.begin();
// 	     p != zipFile.end(); ++p)
//         {
// 	    zip::ZipEntry* entry = *p;
// 	    std::cout << "Entry: " << *entry << std::endl;
// 	}

        input.close();
    }
    catch (except::Exception& ex)
    {
        std::cout << ex.getMessage() << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}
