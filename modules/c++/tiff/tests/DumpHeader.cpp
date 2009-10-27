#include <import/except.h>
#include <import/io.h>
#include <import/sys.h>
#include <import/tiff.h>
#include <fstream>

int main(int argc, char **argv)
{
    try
    {
        if (argc < 2)
            throw except::Exception(FmtX("usage: %s <tiff file>", argv[0]));
        
        sys::OS os;
        std::string path = sys::Path::absolutePath(argv[1]);
        if (!os.exists(path))
            throw except::FileNotFoundException(path);
        
        io::StandardOutStream outStream;
        tiff::FileReader reader(path);
        reader.print(outStream);
    }
    catch (except::Throwable& t)
    {
        std::cerr << t.getMessage() << std::endl;
        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Caught unnamed exception" << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}
