/*
 * =========================================================================
 * This file is part of coda_logging-python 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2015, MDA Information Systems LLC
 *
 * coda_logging-python is free software; you can redistribute it and/or modify
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
 */

%module coda_logging

%feature("autodoc", "1");

%{
  #include "import/logging.h"

  using namespace logging;
%}

%import "std_string.i"

%include "logging/Formatter.h"
%include "logging/StandardFormatter.h"
%include "logging/Filterer.h"

// Classes in the following includes are assumed to take ownership and handle
// the deletion of logging::Formatter* arguments named "formatter" passed to
// any of their methods.  If a SWIG object (Python object wrapping a C++
// object) is passed this way, the following directive stops SWIG from garbage
// collecting the C++ part of the object and causing a segfault on the second
// deletion attempt.
%apply SWIGTYPE *DISOWN { logging::Formatter* formatter };

%include "logging/Handler.h"
%include "logging/StreamHandler.h"

// Clear the typemap applied above
%clear logging::Formatter* formatter;

%include "logging/Filter.h"
%include "logging/Logger.h"
%include "logging/NullLogger.h"

%ignore LoggerManager::LoggerManager();
%ignore LoggerManager::getLoggerSharedPtr(const std::string& name);
%ignore LoggerManager::getLogger(const std::string& name);
%ignore logging::setLogLevel(LogLevel level);
%ignore logging::getLogger(const std::string& name);
%ignore logging::getLoggerSharedPtr(const std::string& name);
%include "logging/LoggerFactory.h"

