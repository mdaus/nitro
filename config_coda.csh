#!/bin/tcsh

set build_type="RelWithDebInfo"  # Debug|Release|RelWithDebInfo|MinSizeRel
set build_lib_type="Static"      # Static|Shared
#set build_lib_type="Shared"      # Static|Shared

set builddir_basename="target"
set installdir_basename="install"

set source_root=`pwd`
set system_type="`uname`-`arch`" # e.g. "Linux_x86_64"
set config_name="${system_type}-${build_type}-${build_lib_type}"
set build_root="${builddir_basename}/${config_name}"
set install_root="${cwd}/${installdir_basename}/${config_name}"

# lower-case compare
set build_lib_type_lc=`echo ${build_lib_type} | tr "[:upper:]" "[:lower:]"`
if (${build_lib_type_lc} == "shared") then
    set build_shared_libs="ON"
else
    set build_shared_libs="OFF"
endif

set extra_args=""

# To generate dependency graph information under target
#set extra_args="--graphviz=graph/graph.dot"

mkdir -p "${build_root}"
set pushdsilent
pushd "${build_root}"
cmake -DCMAKE_INSTALL_PREFIX:PATH=${install_root} -DCMAKE_BUILD_TYPE:STRING=${build_type} -DBUILD_SHARED_LIBS:BOOL=${build_shared_libs} "${source_root}" "${extra_args}" $argv:q
popd
unset pushdsilent

set build_script_name="build_coda.csh"
echo '#\!/bin/tcsh' >! "${build_script_name}"
echo "pushd ${build_root}" >> "${build_script_name}"
echo 'make $argv:q' >> "${build_script_name}"
echo 'popd' >> "${build_script_name}"
chmod u+x "${build_script_name}"
echo "CODA configuration ${config_name} set."
echo "Enter "\""${build_script_name}"\"" to build CODA."
