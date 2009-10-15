# =========================================================================
# This file is part of CODA
# =========================================================================
# 
# (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
#
# CODA is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public 
# License along with this program; If not, 
# see <http://www.gnu.org/licenses/>.
#


#----------------------------------------
# Project specific
#----------------------------------------
include ../build/project.make

TARGET = i686-pc-linux-gnu
LANGUAGE = c++
#----------------------------------------
# Compile flags
#----------------------------------------
CXX =  g++
CXXDEFINES = -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D_REENTRANT -D__POSIX -DUSE_PCRE  -DHAVE_SYS_TIME_H $(MY_CXXDEFINES)
CXXFLAGS = -Wl,-E     -fPIC -v -g -D__DEBUG -Wall $(MY_CXXFLAGS)
INCLUDES = -I../include   -I/home/dpressel/coda-oss/trunk/modules/c++/re/include  $(MY_INCLUDES)
COMPILE = $(CXX) $(CXXFLAGS) $(CXXDEFINES) $(INCLUDES)

#----------------------------------------
# Library flags
#----------------------------------------
ARCHIVE = ar -ru
ARCHNAME = $(PACKAGE)-$(LANGUAGE)
LIBNAME = lib$(ARCHNAME).a
LIBDIR = ../lib/$(TARGET)/gnu
DSONAME = lib$(ARCHNAME).so
LIBS = -l$(ARCHNAME) $(MY_LIBS) -ldl -lnsl -lpthread -lpcre  

#----------------------------------------
# Link
#----------------------------------------
LIBPATH = -L$(LIBDIR) $(MY_LIBPATH)   -L/home/dpressel/coda-oss/trunk/modules/c++/re/lib/i686-pc-linux-gnu/gnu 
LINK = $(LIBPATH) $(LFLAGS) $(LIBS)
LFLAGS =  $(MY_LFLAGS)
#----------------------------------------
# DSO flags
#----------------------------------------
DLLFLAGS =  -fPIC -shared 
DSODIR = ../plugins/$(TARGET)/gnu
DSO    = ${PLUGINS:.cpp=.so}

#----------------------------------------
# Test flags
#----------------------------------------
TESTDIR = ../bin/$(TARGET)

#----------------------------------------
# Misc.
#----------------------------------------
DSOEXT  = .so
LIBEXT  = .a
