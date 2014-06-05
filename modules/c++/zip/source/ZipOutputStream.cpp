#include <zip/ZipOutputStream.h>
#include <io/FileInputStream.h>
#include <except/Exception.h>

namespace zip
{
ZipOutputStream::ZipOutputStream(const std::string& file)
{
    mZip = zipOpen64(file.c_str(), APPEND_STATUS_CREATE);
    if (mZip == NULL)
        throw except::IOException(Ctxt(FmtX("Failed to open zip stream [%s].",
                file.c_str())));

}

void ZipOutputStream::createFileInZip(const std::string& filename,
                                      const std::string& comment,
                                      const std::string& password)
{
    sys::Int32_T results = 0;
    zip_fileinfo zipFileInfo;

    // Add the file
    results = zipOpenNewFileInZip3_64(
            mZip,
            filename.c_str(),
            &zipFileInfo,
            NULL,
            0,
            NULL,
            0,
            comment.empty() ? NULL : comment.c_str(),
            Z_DEFLATED,
            Z_DEFAULT_COMPRESSION,
            0,
            -MAX_WBITS,
            DEF_MEM_LEVEL,
            Z_DEFAULT_STRATEGY,
            password.empty() ? NULL : password.c_str(),
            0,
            0);

    if (results != Z_OK)
         throw except::IOException(Ctxt(FmtX("Failed to create file [%s].",
                filename.c_str())));
}

void ZipOutputStream::closeFileInZip()
{
    sys::Int32_T results = zipCloseFileInZip(mZip);
    if (results != Z_OK)
         throw except::IOException(Ctxt("Failed to close file at zip location."));
}

void ZipOutputStream::write(const std::string& inputFile,
                            const std::string& zipFilepath)
{
    io::FileInputStream input(inputFile);

    createFileInZip(zipFilepath);
    input.streamTo(*this);
    closeFileInZip();
}

void ZipOutputStream::write(const sys::byte* b, sys::Size_T len)
{
    // Write the contents to the location
    sys::Int32_T results = zipWriteInFileInZip(mZip, b, len);

    if (results != Z_OK)
         throw except::IOException(Ctxt("Failed to write file to zip location."));
}

void ZipOutputStream::close()
{
    sys::Int32_T results = 0;
    results = zipClose(mZip, NULL);
    if (results != Z_OK)
        throw except::IOException(Ctxt("Failed to save zip file."));
}
}
