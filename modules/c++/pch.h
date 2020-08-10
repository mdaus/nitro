#pragma once

#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include <ctype.h>
#include <time.h>
#include <stdint.h>

#include <string>
#include <numeric>
#include <limits>
#include <memory>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <ios>
#include <vector>
#include <map>
#include <utility>
#include <functional>
#include <mutex>

#include <sys/File.h>

#include <windows.h>
#undef min
#undef max

#include <import/sys.h>
#include <import/mem.h>
#include <import/except.h>
#include <import/io.h>
#include <import/types.h>
#include <import/mt.h>

#include "import/nitf.h"

#include "nitf/ImageIO.h"
#include "nitf/System.h"
#include "nitf/Field.h"
#include "nitf/Types.h"


#include "nitf/Object.hpp"
#include "nitf/NITFException.hpp"
#include "nitf/TRE.hpp"
#include "nitf/Record.hpp"

#pragma comment(lib, "str-c++")
#pragma comment(lib, "mt-c++")
#pragma comment(lib, "except-c++")
#pragma comment(lib, "sys-c++")

#pragma comment(lib, "ws2_32")