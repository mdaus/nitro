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
#if _MSC_FULL_VER >= 190000000
	return "v142";
#else
#error "Don't know $(PlatformToolset) value.'"
#endif
}

static bool is_x64_Configuration(const fs::path& path) // "Configuration" is typically "Debug" or "Release"
{
	static const std::string build_configuration = Configuration();
	const auto Configuration = path.filename();
	const auto path_parent_path = path.parent_path();
	const auto x64 = path_parent_path.filename();
	return (Configuration == build_configuration) && (x64 == Platform());
}

static bool is_install_unittests(const fs::path& path)
{
	const auto unittests = path.filename();
	const auto path_parent_path = path.parent_path();
	const auto install = path_parent_path.filename();
	return (unittests == "unittests") && (install == "install");
}
static bool is_install_tests(const fs::path& path)
{
	const auto tests = path.filename();
	const auto path_parent_path = path.parent_path();
	const auto install = path_parent_path.filename();
	return (tests == "tests") && (install == "install");
}

static fs::path make_waf_install(const fs::path& p)
{
	// just "install" on Linux; install-Debug-x64.v142 on Windows
	const auto configuration_and_platform = Configuration() + "-" + Platform() + "." + PlatformToolset();
	return p / ("install-" + configuration_and_platform);
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
	if ((argv0 == "Test++.exe") || (argv0 == "testhost.exe"))
	{
		// Running GTest unit-tests in Visual Studio on Windows
		if (is_x64_Configuration(cwd))
		{
			//const auto root = cwd.parent_path().parent_path();
			//const auto install = "install-" + Configuration() + "-" + Platform() + "." + PlatformToolset();
			//return root / install / path;
			return cwd / path;
		}
	}

	auto extension = argv0.extension().string();
	str::upper(extension);
	if (extension == ".EXE")
	{
		// stand-alone executable on Windows (ends in .EXE)
		const auto parent_path = exec.parent_path();
		if (is_x64_Configuration(parent_path))
		{
			const auto parent_path_ = parent_path.parent_path().parent_path();
			return parent_path_ / "dev" / "tests" / "images";
		}

		const auto root = findRoot(exec);
		const auto install = make_waf_install(root);
		return install / path;
	}

	// stand-alone unit-test on Linux
	const auto exec_dir = exec.parent_path();
	if (is_install_unittests(exec_dir))
	{
		const auto install = exec_dir.parent_path();
		return install / "unittests" / "data";
	}
	if (is_install_tests(exec_dir))
	{
		const auto install = exec_dir.parent_path();
		return install / "unittests" / "data";
	}

	if (argv0 == "unittests")
	{
		// stand-alone unittest executable on Linux
		const auto bin = exec.parent_path();
		if (bin.filename() == "bin")
		{
			const auto unittests = bin.parent_path();
			return unittests / "unittests" / "data";
		}
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