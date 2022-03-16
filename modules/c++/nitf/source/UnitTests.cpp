/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2022, Maxar Technologies, Inc.
 *
 * NITRO is free software; you can redistribute it and/or modify
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
 * License along with this program; if not, If not,
 * see <http://www.gnu.org/licenses/>.
 *
 */

#include "nitf/UnitTests.hpp"

#include <std/filesystem>
#include <import/sys.h>

namespace fs = std::filesystem;

static const sys::OS os;
static std::string Configuration() // "Configuration" is typically "Debug" or "Release"
{
	return os.getSpecialEnv("Configuration");
}
static std::string Platform()
{
	return os.getSpecialEnv("Platform");
}

// https://stackoverflow.com/questions/13794130/visual-studio-how-to-check-used-c-platform-toolset-programmatically
static std::string PlatformToolset()
{
	// https://docs.microsoft.com/en-us/cpp/build/how-to-modify-the-target-framework-and-platform-toolset?view=msvc-160
#ifdef _WIN32
#if _MSC_FULL_VER >= 190000000
	return "v142";
#else
#error "Don't know $(PlatformToolset) value.'"
#endif
#else 
	// Linux
	return "";
#endif
}

static fs::path make_waf_install(const fs::path& p)
{
	// just "install" on Linux; install-Debug-x64.v142 on Windows
#ifdef _WIN32
	const auto configuration_and_platform = Configuration() + "-" + Platform() + "." + PlatformToolset();
	return p / ("install-" + configuration_and_platform);
#else
	// Linux
	return "install";
#endif
}

static fs::path make_cmake_install(const fs::path& p)
{
	auto out = p;
	fs::path configuration_and_platform;
	fs::path build;
	while (out.stem() != "out")
	{
		configuration_and_platform = build; // "...\out\build\x64-Debug"
		build = out; // "...\out\build"
		out = out.parent_path(); // "...\out"
	}

	const auto install = out / "install"; // "...\out\install"
	return install / configuration_and_platform.stem(); // "...\out\install\x64-Debug"
}

static fs::path findRoot(const fs::path& p)
{
	if (is_regular_file(p / "LICENSE") && is_regular_file(p / "README.md") && is_regular_file(p / "CMakeLists.txt"))
	{
		return p;
	}
	return findRoot(p.parent_path());
}

static fs::path buildDir(const fs::path& path)
{
	const auto cwd = fs::current_path();

	const auto exec = fs::absolute(fs::path(os.getCurrentExecutable()));
	const auto argv0 = exec.filename();

	if (argv0 == "testhost.exe")
	{
		// Running in Visual Studio on Windows
		return cwd / path;
	}

	auto extension = argv0.extension().string();
	str::upper(extension);
	if (extension == ".EXE")
	{
		// stand-alone executable on Windows (ends in .EXE)
		const auto root = findRoot(exec);
		std::string relative_exec = exec.string();
		str::replaceAll(relative_exec, root.string(), "");
		fs::path install;
		if (str::starts_with(relative_exec, "/out") || str::starts_with(relative_exec, "\\out"))
		{
			install = make_cmake_install(exec);
		}
		else
		{
			install = make_waf_install(root);
		}
		
		return install / path;
	}

	//fprintf(stderr, "cwd = %s\n", cwd.c_str());
	//fprintf(stderr, "exec = %s\n", exec.c_str());

	// running a CTest from CMake
	const auto nitro_out = cwd.parent_path().parent_path().parent_path().parent_path().parent_path();
	const auto install = nitro_out / "install" / (Platform() + "-" + Configuration()); // e.g., "x64-Debug"
	return install / path;
	// return cwd;
}

std::string nitf::Test::buildPluginsDir()
{
	const auto plugins = buildDir(fs::path("share") / "nitf" / "plugins");
	return plugins.string();
}