#ifndef __ZIP_ZIP_OUTPUT_STREAM_H__
#define __ZIP_ZIP_OUTPUT_STREAM_H__

#include <string>
#include <zip.h>
#include <sys/Conf.h>
#include <io/OutputStream.h>

namespace zip
{
/*
 *  \class ZipOutputStream
 *  \brief Creates a zip file which can hold a variable amount of files
 *         in a user defined directory structure.
 */
class ZipOutputStream: public io::OutputStream
{
public:
    /*
     *  \func Constructor
     *  \brief Sets up the internal structure of the class.
     *
     *  \param file The path and filename of the zip.
     */
    ZipOutputStream(const std::string& file);

    virtual ~ZipOutputStream()
    {
    }

    /*
     *  \func createFileInZip
     *  \brief Creates a new file within the zip which can be written to.
     *
     *  \filename The directory path and filename as it will appear in the zip.
     *  \comment An optional comment.
     *  \password An optional password for the file.
     */
    void createFileInZip(const std::string& filename,
                         const std::string& comment = "",
                         const std::string& password = "");

    /*
     *  \func closeFileInZip
     *  \brief Closes a file that was openned by createFileInZip. This will
     *         throw if a file was not created.
     */
    void closeFileInZip();

    /*
     *  \func write
     *  \brief Convience function which will create, write, and close a file.
     *
     *  \inputFile The path to the file that you want added to the zip.
     *  \zipFilepath The path and file name you want the inputFile to appear
     *               as in the zip file.
     */
    void write(const std::string& inputFile,
               const std::string& zipFilepath);

    virtual void write(const sys::byte* b, sys::Size_T len);

    virtual void close();

private:
    zipFile mZip;
};

}

#endif
