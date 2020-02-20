#!/bin/sh

build_type="RelWithDebInfo"  # Debug|Release|RelWithDebInfo|MinSizeRel
build_lib_type="Static"      # Static|Shared

builddir_basename="target"
installdir_basename="install"

source_root=`pwd`
system_type="`uname`-`arch`" # e.g. "Linux_x86_64"
config_name="${system_type}-${build_type}-${build_lib_type}"
build_root="${builddir_basename}/${config_name}"
install_root="${source_root}/${installdir_basename}/${config_name}"

# lower-case compare
build_lib_type_lc=`echo ${build_lib_type} | tr "[:upper:]" "[:lower:]"`
if [ ${build_lib_type_lc} = "shared" ];
then
    build_shared_libs="ON"
else
    build_shared_libs="OFF"
fi

extra_args=""
echo "${build_root}"
# To generate dependency graph information under target
#set extra_args="--graphviz=graph/graph.dot"

mkdir -p "${build_root}"
cd "${build_root}"
cmake -DCMAKE_INSTALL_PREFIX:PATH=${install_root} -DCMAKE_BUILD_TYPE:STRING=${build_type} -DBUILD_SHARED_LIBS:BOOL=${build_shared_libs} "${extra_args}" "$@" "${source_root}"
cd -

build_script_name="build_coda.sh"
echo '#\!/bin/sh' > "${build_script_name}"
echo "cd ${build_root}" >> "${build_script_name}"
echo 'make "$@"' >> "${build_script_name}"
echo 'cd -' >> "${build_script_name}"
chmod u+x "${build_script_name}"
echo "CODA configuration ${config_name} set."
echo "Enter "\""${build_script_name}"\"" to build CODA."
