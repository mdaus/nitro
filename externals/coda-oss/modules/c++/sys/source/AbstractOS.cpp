/* =========================================================================
 * This file is part of sys-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2016, MDA Information Systems LLC
 * (C) Copyright 2021, Maxar Technologies, Inc.
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
#include <sys/AbstractOS.h>

#include <assert.h>

#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <iterator>
#include <algorithm>

#include <config/compiler_extensions.h>
#include <import/str.h>
#include <sys/Path.h>
#include <sys/DirectoryEntry.h>
#include <sys/DateTime.h>
#include <sys/Dbg.h>

#include <sys/filesystem.h>
namespace fs = coda_oss::filesystem;

namespace sys
{
AbstractOS::AbstractOS() = default;

AbstractOS::~AbstractOS() = default;

std::vector<std::string>
AbstractOS::search(const std::vector<std::string>& searchPaths,
                   const std::string& fragment,
                   const std::string& extension,
                   bool recursive) const
{
    std::vector<std::string> elementsFound;

    // add the search criteria
    if (!fragment.empty() && !extension.empty())
    {
        sys::ExtensionPredicate extPred(extension);
        sys::FragmentPredicate fragPred(fragment);

        sys::LogicalPredicate logicPred(false);
        logicPred.addPredicate(&extPred);
        logicPred.addPredicate(&fragPred);

        elementsFound = sys::FileFinder::search(logicPred,
                                                searchPaths,
                                                recursive);
    }
    else if (!extension.empty())
    {
        sys::ExtensionPredicate extPred(extension);
        elementsFound = sys::FileFinder::search(extPred,
                                                searchPaths,
                                                recursive);
    }
    else if (!fragment.empty())
    {
        sys::FragmentPredicate fragPred(fragment);
        elementsFound = sys::FileFinder::search(fragPred,
                                                searchPaths,
                                                recursive);
    }
    return elementsFound;
}

std::vector<coda_oss::filesystem::path> AbstractOS::search(
        const std::vector<coda_oss::filesystem::path>& searchPaths,
        const std::string& fragment,
        const std::string& extension,
        bool recursive) const
{
    const auto results = search(convertPaths(searchPaths), fragment, extension, recursive);
    return convertPaths(results);
}

void AbstractOS::remove(const std::string& path) const
{
    if (isDirectory(path))
    {
        // Iterate through each entry in the directory and remove it too
        DirectoryEntry dirEntry(path);
        for (DirectoryEntry::Iterator iter = dirEntry.begin();
                iter != dirEntry.end();
                ++iter)
        {
            const std::string filename(*iter);
            if (filename != "." && filename != "..")
            {
                remove(sys::Path::joinPaths(path, filename));
            }
        }

        // Directory should be empty, so remove it
        removeDirectory(path);
    }
    else
    {
        removeFile(path);
    }
}

bool AbstractOS::getEnvIfSet(const std::string& envVar, std::string& value, bool includeSpecial) const
{
    if (isEnvSet(envVar))
    {
        value = getEnv(envVar);
        return true;
    }

    if (includeSpecial && isSpecialEnv(envVar))
    {
        value = getSpecialEnv(envVar);
        return true;
    }

    return false;
}

std::string s_argvPathname;
void AbstractOS::setArgvPathname(const std::string& argvPathname)
{
    if (argvPathname.empty())
    {
        throw std::invalid_argument("argvPathname is empty");
    }
    s_argvPathname = argvPathname;
}
std::string AbstractOS::getArgvPathname(const std::string& argvPathname) const
{
    return argvPathname.empty() ? s_argvPathname : argvPathname;
}

std::string AbstractOS::getCurrentExecutable(
        const std::string& argvPathname) const
{
    if (argvPathname.empty())
    {
        // If the OS-specific overrides can't find the name,
        // and we don't have an argv[0] to look at,
        // there's nothing we can do.
        return "";
    }

    if (sys::Path::isAbsolutePath(argvPathname))
    {
        return argvPathname;
    }

    std::string candidatePathname = sys::Path::joinPaths(
            getCurrentWorkingDirectory(), argvPathname);
    if (exists(candidatePathname))
    {
        return candidatePathname;
    }

    // Look for it in PATH
    std::vector<std::string> pathDirs;
    if (splitEnv("PATH", pathDirs))
    {
        for (const auto& pathDir : pathDirs)
        {
            candidatePathname =
                    sys::Path::joinPaths(sys::Path::absolutePath(pathDir),
                                         argvPathname);
            if (exists(candidatePathname))
            {
                return candidatePathname;
            }
        }
    }

    return "";
}

// A variable like PATH is often several directories, return each one that exists.
static bool splitEnv_(const AbstractOS& os, const std::string& envVar, std::vector<std::string>& result, fs::file_type* pType = nullptr)
{
    std::string value;
    if (!os.getEnvIfSet(envVar, value))
    {
        return false;
    }
    const auto vals = str::split(value, sys::Path::separator());
    for (const auto& val : vals)
    {
        const fs::path val_(val);
        bool matches = true;
        if (pType != nullptr)
        {
            const auto isFile = (*pType == fs::file_type::regular) && is_regular_file(val_);
            const auto isDirectory = (*pType == fs::file_type::directory) && is_directory(val_);
            matches = isFile || isDirectory;
        }
        if (exists(val_) && matches)
        {
            result.push_back(val);
        }
    }
    return !result.empty(); // false for no matches
}
bool AbstractOS::splitEnv(const std::string& envVar, std::vector<std::string>& result, fs::file_type type) const
{
    return splitEnv_(*this, envVar, result, &type);
}
bool AbstractOS::splitEnv(const std::string& envVar, std::vector<std::string>& result) const
{
    return splitEnv_(*this, envVar, result);
}

static void modifyEnv(AbstractOS& os, const std::string& envVar, bool overwrite,
                      const std::vector<std::string>& prepend, const std::vector<std::string>& append)
{
    std::vector<std::string> values;
    auto splitResult = os.splitEnv(envVar, values);
    if (splitResult && !overwrite)
    {
        // envVar already exists and we can't overwrite it
        return;
    }

    values.insert(values.begin(), prepend.begin(), prepend.end()); // prepend
    values.insert(values.end(), append.begin(), append.end());  // append

    std::string val;
    if (!values.empty()) // don't let size()-1 wrap-around
    {
        for (size_t i = 0; i<values.size(); i++)
        {
            val += values[i];
            if (i < values.size()-1)
            {
                val += Path::separator();  // ':' or ';'
            }
        }
    }

    os.setEnv(envVar, val, overwrite);
}
void AbstractOS::prependEnv(const std::string& envVar, const std::vector<std::string>& values, bool overwrite)
{
    static const std::vector<std::string> empty;
    modifyEnv(*this, envVar, overwrite, values, empty);
}
void AbstractOS::appendEnv(const std::string& envVar, const std::vector<std::string>& values, bool overwrite)
{
    static const std::vector<std::string> empty;
    modifyEnv(*this, envVar, overwrite, empty, values);
}

static std::string getSpecialEnv_PID(const AbstractOS& os, const std::string& envVar)
{
    assert((envVar == "$") || (envVar == "PID"));
    CODA_OSS_mark_symbol_unused(envVar);
    const auto pid = os.getProcessId();
    return std::to_string(pid);
}

static std::string getSpecialEnv_USER(const AbstractOS& os, const std::string& envVar)
{
    // $USER on *nix, %USERNAME% on Windows; make it so either one always works
    assert((envVar == "USER") || (envVar == "USERNAME"));
    CODA_OSS_mark_symbol_unused(envVar);
    #if _WIN32
    return os.getEnv("USERNAME");
    #else
    return os.getEnv("USER");
    #endif
}

static std::string getSpecialEnv_HOME(const AbstractOS& os, const std::string& envVar)
{
    // $HOME on *nix, %USERPROFILE% on Windows; make it so either one always works
    assert((envVar == "HOME") || (envVar == "USERPROFILE"));

    CODA_OSS_mark_symbol_unused(envVar);
    #ifdef _WIN32
    constexpr auto home = "USERPROFILE";
    #else  // assuming *nix
    // Is there a better way to support ~ on *nix than $HOME ?
    constexpr auto home = "HOME";
    #endif

    std::vector<std::string> paths;
    if (!os.splitEnv(home, paths, fs::file_type::directory))
    {
        // something is horribly wrong
        throw except::FileNotFoundException(Ctxt(home));
    }

    if (paths.size() != 1)
    {
        // somebody set HOME to multiple directories ... why?
        throw except::FileNotFoundException(Ctxt(home));
    }
    return paths[0];
}

static std::string getSpecialEnv_Configuration(const AbstractOS&, const std::string& envVar)
{
    assert(envVar == "Configuration");
    CODA_OSS_mark_symbol_unused(envVar);
    // in Visual Studio, by default this is usually "Debug" and "Release"
    return sys::debug_build() ? "Debug" : "Release";
}
static std::string getSpecialEnv_Platform(const AbstractOS&, const std::string& envVar)
{
    assert((envVar == "Platform") || (envVar == "HOSTTYPE"));

    // in Visual Studio, this is "Win32" (maybe "x86") or "x64"
    CODA_OSS_mark_symbol_unused(envVar);
    #ifdef _WIN32
        #ifdef _WIN64
        return "x64";
        #else
        return "Win32"; // "x86" ?
       #endif
    #else // assume 64-bit *nix
    return "x86_64";
    #endif
}

// https://stackoverflow.com/questions/13794130/visual-studio-how-to-check-used-c-platform-toolset-programmatically
static std::string getSpecialEnv_PlatformToolset(const AbstractOS&, const std::string& envVar)
{
    assert(envVar == "PlatformToolset");
    CODA_OSS_mark_symbol_unused(envVar);

#ifdef _WIN32
	// https://docs.microsoft.com/en-us/cpp/build/how-to-modify-the-target-framework-and-platform-toolset?view=msvc-160
	// https://learn.microsoft.com/en-us/cpp/preprocessor/predefined-macros?view=msvc-170
	#if _MSC_VER >= 1930
		return "v143"; // Visual Studio 2022
	#elif _MSC_VER >= 1920
		return "v142"; // Visual Studio 2019
    #elif _MSC_VER >= 1910
        return "v141";  // Visual Studio 2017
    #elif _MSC_VER >= 1900
        return "v140";  // Visual Studio 2015
	#else
		#error "Don't know $(PlatformToolset) value.'"
	#endif
#else 
	// Linux
	return "";
#endif
}

static std::string getSpecialEnv_SECONDS_()
{
    // https://en.cppreference.com/w/cpp/chrono/c/difftime
    static const auto start = std::time(nullptr);
    const auto diff = static_cast<int64_t>(std::difftime(std::time(nullptr), start));
    return std::to_string(diff);
}
static std::string getSpecialEnv_SECONDS(const AbstractOS&, const std::string& envVar)
{
    // https://www.gnu.org/software/bash/manual/html_node/Bash-Variables.html
    // "This variable expands to the number of seconds since the shell was started. ..."
    assert(envVar == "SECONDS");
    CODA_OSS_mark_symbol_unused(envVar);
    return getSpecialEnv_SECONDS_();
}

CODA_OSS_disable_warning_push
#if _MSC_VER
#pragma warning(disable: 26426) // Global initializer calls a non-constexpr function '...' (i.22).
#endif
static std::string strUnusedSeconds = getSpecialEnv_SECONDS_(); // "start" the "shell"
CODA_OSS_disable_warning_pop

// See https://www.gnu.org/software/bash/manual/html_node/Bash-Variables.html
// and https://wiki.bash-hackers.org/syntax/shellvars
typedef std::string (*get_env_fp)(const AbstractOS&, const std::string&);
// For some special variables, a separate function may be needed;
// others can be done in-line.
static const std::map<std::string, get_env_fp> s_get_env{
                                                    {"0", nullptr}, {"ARGV0", nullptr},
                                                    {"$", getSpecialEnv_PID}, {"PID", getSpecialEnv_PID},
                                                    {"PWD", nullptr},
                                                    {"USER", getSpecialEnv_USER}, {"USERNAME", getSpecialEnv_USER},
                                                    {"HOME", getSpecialEnv_HOME}, {"USERPROFILE", getSpecialEnv_HOME},
                                                    {"EPOCHSECONDS", nullptr},
                                                    {"HOSTNAME", nullptr},
                                                    {"HOSTTYPE", getSpecialEnv_Platform}, // x86_64
                                                    {"MACHTYPE", nullptr}, // x86_64-pc-linux-gnu
                                                    {"OSTYPE", nullptr}, // linux-gnu
                                                    {"SECONDS", getSpecialEnv_SECONDS},
                                                    // c.f., Visual Studio
                                                    {"Configuration", getSpecialEnv_Configuration},
                                                    {"Platform", getSpecialEnv_Platform},
                                                    {"PlatformToolset", getSpecialEnv_PlatformToolset},
};
bool AbstractOS::isSpecialEnv(const std::string& envVar) const
{
    const auto it = s_get_env.find(envVar);
    return it != s_get_env.end();
}

std::string AbstractOS::getSpecialEnv(const std::string& envVar) const
{
    const auto it = s_get_env.find(envVar);
    if (it == s_get_env.end())
    {
        // see sys::OSUnix::getEnv()
        throw sys::SystemException(Ctxt("Unable to get special environment variable: " + envVar));
    }

    // call the function if there is one
    auto f = it->second;
    if (f != nullptr)
    {
        return f(*this, envVar);
    }

    if ((envVar == "0") || (envVar == "ARGV0")) 
    {
        return getCurrentExecutable(); // $0
    }

    if (envVar == "PWD")
    {
        return getCurrentWorkingDirectory();
    }

    if (envVar == "HOSTNAME")
    {
        return getNodeName();
    }

    if (envVar == "EPOCHSECONDS")
    {
        return std::to_string(sys::DateTime::getEpochSeconds());
    }

    if (envVar == "OSTYPE")
    {
        return sys::platformName<sys::Platform>();
    }
    
    // should explicitly handle all env. vars in some way    
    throw sys::SystemException(Ctxt("Unable to determine value for special environment variable: " + envVar));
}

}

