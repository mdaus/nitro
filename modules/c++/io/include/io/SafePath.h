#ifndef __IO_SAFEPATH_H__
#define __IO_SAFEPATH_H__

#include <string>

namespace io
{

/*!
 * \brief TempFile class
 *
 * Class to protect against writing a broken file which
 * appears to be complete
 */
class SafePath
{
public:

    /*!
     * Create a TempFile object and generate a unique, temporary name
     */
    SafePath(const std::string& realPathname);

    /*!
     * Destruct TempFile object
     * This either changes the name of the file back to realName or deletes
     * the temp file depending on whether or not the moved flag has been set
     */
    ~SafePath();

    /*!
     * Return the temporary filename
     * Throws if file has already been moved
     */
    std::string getTempPathname() const;

    /*!
     * Tell the object that file processing was successful
     * and that it should rename the file.
     */
    void moveFile();

private:

    const std::string mRealPathname;
    const std::string mTempPathname;
    bool moved;
};

}

#endif /* __IO_SAFEPATH_H__ */
