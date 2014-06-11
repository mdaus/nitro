#include <io/SafePath.h>
#include <unique/UUID.hpp>
#include <sys/OS.h>

namespace io
{

SafePath::SafePath(const std::string& realPathname) :
    mRealPathname(realPathname),
    mTempPathname(unique::generateUUID() + ".tmp"),
    moved(false)
{
}

SafePath::~SafePath()
{
    try
    {
        sys::OS os;
        moveFile();
        if(!moved)
        {
            os.remove(mTempPathname);
        }
    }
    catch(...){}
}

std::string SafePath::getTempPathname() const
{
    if(moved)
    {
        throw except::Exception(Ctxt(
                "File has already been moved, use of getTemp() is invalid."));
    }
    return mTempPathname;
}

void SafePath::moveFile()
{
    if(!moved)
    {
        sys::OS os;
        if(os.move(mTempPathname, mRealPathname))
        {
            moved = true;
        }
        else
        {
            throw except::Exception(Ctxt(
                           "Error renaming file from " + mTempPathname + " to " 
                           + mRealPathname + " in moveFile()."));
        }
    }
}

}
