/* =========================================================================
 * This file is part of sys-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * sys-c++ is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; If not,
 * see <http://www.gnu.org/licenses/>.
 *
 */

#include <iterator>
#include "sys/Glob.h"
#include "sys/DirectoryEntry.h"
#include "sys/Path.h"

bool sys::ExistsPredicate::operator()(const std::string& entry)
{
    return sys::Path(entry).exists();
}

bool sys::FileOnlyPredicate::operator()(const std::string& entry)
{
    return sys::Path(entry).isFile();
}

bool sys::DirectoryOnlyPredicate::operator()(const std::string& entry)
{
    return sys::Path(entry).isDirectory();
}

sys::FragmentPredicate::FragmentPredicate(const std::string& fragment,
                                          bool ignoreCase) :
    mFragment(fragment), mIgnoreCase(ignoreCase)
{
}

bool sys::FragmentPredicate::operator()(const std::string& entry)
{
    if (mIgnoreCase)
    {
        std::string base = entry;
        str::lower(base);

        std::string match = mFragment;
        str::lower(match);

        return str::contains(base, match);
    }
    else
        return str::contains(entry, mFragment);
}


sys::ExtensionPredicate::ExtensionPredicate(const std::string& ext, 
                                            bool ignoreCase) :
    mExt(ext), mIgnoreCase(ignoreCase)
{
}

bool sys::ExtensionPredicate::operator()(const std::string& filename)
{
    if (!sys::FileOnlyPredicate::operator()(filename))
        return false;

    std::string ext = sys::Path::splitExt(filename).second;
    if (mIgnoreCase)
    {
        std::string matchExt = mExt;
        str::lower(matchExt);
        str::lower(ext);
        return ext == matchExt;
    }
    return ext == mExt;
}

sys::MultiGlobPredicate::MultiGlobPredicate(bool orOperator) :
    mOrOperator(orOperator)
{
}

sys::MultiGlobPredicate::~MultiGlobPredicate()
{
    for (size_t i = 0, n = mPredicates.size(); i < n; ++i)
    {
        sys::MultiGlobPredicate::PredicatePair& p = mPredicates[i];
        if (p.first && p.second)
            delete p.first;
    }
}

bool sys::MultiGlobPredicate::operator()(const std::string& entry)
{
    bool ok = !mOrOperator;
    for (size_t i = 0, n = mPredicates.size(); i < n && ok != mOrOperator; ++i)
    {
        sys::MultiGlobPredicate::PredicatePair& p = mPredicates[i];
        if (mOrOperator)
            ok |= (p.first && (*p.first)(entry));
        else
            ok &= (p.first && (*p.first)(entry));
    }
    return ok;
}

sys::MultiGlobPredicate& sys::MultiGlobPredicate::addPredicate(
    GlobPredicate* filter,
    bool ownIt)
{
    mPredicates.push_back(
        sys::MultiGlobPredicate::PredicatePair(
            filter, ownIt));
    return *this;
}

sys::Glob::Glob(const std::vector<std::string>& searchPaths) :
    mPaths(searchPaths)
{
}

sys::Glob::~Glob()
{
    for (size_t i = 0; i < mPredicates.size(); ++i)
    {
        sys::Glob::PredicatePair& p = mPredicates[i];
        if (p.first && p.second)
            delete p.first;
    }
}

sys::Glob& sys::Glob::addSearchPath(std::string path)
{
    mPaths.push_back(path);
    return *this;
}

sys::Glob& sys::Glob::addPredicate(sys::GlobPredicate* filter,
                                   bool ownIt)
{
    mPredicates.push_back(sys::Glob::PredicatePair(filter, ownIt));
    return *this;
}

std::vector<std::string> sys::Glob::search(bool recursive) const
{
    // turn it into a list so we can queue additional entries
    std::list < std::string > paths;
    std::copy(mPaths.begin(), mPaths.end(), std::back_inserter(paths));

    std::vector < std::string > files;
    size_t numInputPaths = mPaths.size();
    for (size_t pathIdx = 0; !paths.empty(); ++pathIdx)
    {
        sys::Path path(paths.front());
        paths.pop_front();

        //! check if it exists
        if (path.exists())
        {
            // check it against all predicates
            for (size_t i = 0; i < mPredicates.size(); ++i)
            {
                const sys::Glob::PredicatePair& p = mPredicates[i];
                if (p.first)
                {
                    // check if this meets the criteria -- 
                    // we only need one to add it
                    if ((*p.first)(path.getPath()))
                    {
                        files.push_back(path.getPath());
                        break;
                    }
                }
            }

            // if it's a directory we need to search its contents
            if (path.isDirectory())
            {
                // If its an original directory or we are recursively searching
                if (pathIdx < numInputPaths || recursive)
                {
                    sys::DirectoryEntry d(path.getPath());
                    for (sys::DirectoryEntry::Iterator p = d.begin(); 
                         p != d.end(); ++p)
                    {
                        std::string fname(*p);
                        if (fname != "." && fname != "..")
                        {
                            // add it to the list
                            paths.push_back(sys::Path::joinPaths(path.getPath(),
                                                                 fname));
                        }
                    }
                }
            }

        }
    }
    return files;
}

