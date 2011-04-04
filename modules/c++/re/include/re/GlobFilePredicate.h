#ifndef __RE_GLOB_FILE_PREDICATE_H__
#define __RE_GLOB_FILE_PREDICATE_H__

#include "sys/FileFinder.h"
#include "re/PCRE.h"

namespace re
{

struct GlobFilePredicate : sys::FilePredicate
{
    GlobFilePredicate(std::string match)
    {
        mRegex.compile(match);
    }
    
    bool operator()(const std::string& filename)
    {
        return mRegex.matches(filename);
    }
    private:
    re::PCRE mRegex;
    

};

}
#endif
