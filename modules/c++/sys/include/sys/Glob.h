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

#ifndef __SYS_GLOB_H__
#define __SYS_GLOB_H__

#include <functional>
#include <vector>

namespace sys
{

/**
 * Predicate interface for all entries
 */
struct GlobPredicate : std::unary_function<std::string, bool>
{
    virtual ~GlobPredicate() {}
    virtual bool operator()(const std::string& entry) = 0;
};

/**
 * Predicate interface for existance
 */
struct ExistsPredicate : GlobPredicate
{
    virtual ~ExistsPredicate() {}
    virtual bool operator()(const std::string& entry);
};

/**
 * Predicate that matches files only (no directories)
 */
struct FileOnlyPredicate: public GlobPredicate
{
    virtual ~FileOnlyPredicate() {}
    virtual bool operator()(const std::string& entry);
};

/**
 * Predicate that matches directories only (no files)
 */
struct DirectoryOnlyPredicate: public GlobPredicate
{
    virtual ~DirectoryOnlyPredicate() {}
    virtual bool operator()(const std::string& entry);
};

/**
 * Predicate that matches directories only (no files)
 */
struct FragmentPredicate : public GlobPredicate
{
public:
    FragmentPredicate(const std::string& ext, bool ignoreCase = true);
    bool operator()(const std::string& filename);

private:
    std::string mFragment;
    bool mIgnoreCase;

};


/**
 * Predicate interface for filtering files with a specific extension
 * This method will not match '.xxx.yyy' type patterns, since the 
 * splitting routines will only find '.yyy'.  See re::GlobFilePredicate
 * for a more useful finder.
 */
class ExtensionPredicate: public FileOnlyPredicate
{
public:
    ExtensionPredicate(const std::string& ext, bool ignoreCase = true);
    bool operator()(const std::string& filename);

private:
    std::string mExt;
    bool mIgnoreCase;
};

class MultiGlobPredicate : public GlobPredicate
{
public:
    MultiGlobPredicate(bool orOperator = true);
    virtual ~MultiGlobPredicate();

    virtual bool operator()(const std::string& entry);
    MultiGlobPredicate& addPredicate(GlobPredicate* filter, bool ownIt = false);

protected:
    bool mOrOperator;

    typedef std::pair<GlobPredicate*, bool> PredicatePair;
    std::vector<PredicatePair> mPredicates;
};


/**
 * \class Glob
 *
 *  The Glob class allows you to search for 
 *  files/directories in a clean way.
 */
class Glob
{
public:
    Glob(){}
    Glob(const std::vector<std::string>& searchPaths);
    ~Glob();

    /**
     * Add a search path
     */
    Glob& addSearchPath(std::string path);

    /**
     * Add a predicate/filter to use when searching
     */
    Glob& addPredicate(GlobPredicate* filter, bool ownIt = false);

    /**
     * Perform the search
     * \return a std::vector<std::string> of paths that match
     */
    std::vector<std::string> search(bool recursive = false) const;

protected:
    std::vector<std::string> mPaths;

    typedef std::pair<GlobPredicate*, bool> PredicatePair;
    std::vector<PredicatePair> mPredicates;
};

}

#endif
