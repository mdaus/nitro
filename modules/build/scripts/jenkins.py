import zipfile, glob, sys
import os, platform, shutil
from subprocess import check_call
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-p", "--package", dest="package_name", help="Package name")
parser.add_option("-b", "--build-dir", dest="build_dir", help="Build Directory", default=".")
parser.add_option("-c", "--config-options", dest="config_options", help="Configure Options", default="--require-ant,--require-java")

(options, args) = parser.parse_args()

if not options.package_name or not options.build_dir:
    print 'You must specify package name and build directory'
    sys.exit(1)

install_suffix = ''

package_name = options.package_name
build_dir = options.build_dir
config_options = options.config_options.split(',')

print 'Package Name: %s' % package_name
print 'Build Dir: %s' % build_dir
print 'Config Options: %s' % config_options

if 'studio11' in os.environ.get('JOB_NAME'):
    os.environ['PATH'] += os.pathsep + '/var/studio11/SUNWspro/bin'
    os.environ['LD_LIBRARY_PATH'] += os.pathsep + '/var/studio11/SUNWspro/lib'
    install_suffix = 'sparc-sun-solaris2.10-64-studio11'
elif 'studio12' in os.environ.get('JOB_NAME'):
    os.environ['PATH'] += os.pathsep + '/var/studio12/SUNWspro/bin'
    os.environ['LD_LIBRARY_PATH'] += os.pathsep + '/var/studio12/SUNWspro/lib'
    install_suffix = 'sparc-sun-solaris2.10-64-studio12'
elif 'linux' in os.environ.get('JOB_NAME'):
    os.environ['PATH'] += os.pathsep + '/opt/python/v2.7.3/bin'
    install_suffix = 'x86_64-linux-gnu-64'
elif 'win32' in os.environ.get('JOB_NAME'):
    install_suffix = 'win32'
elif 'win64' in os.environ.get('JOB_NAME'):
    install_suffix = 'win64'
if '-md' in os.environ.get('JOB_NAME'):
    install_suffix += '-md'
    config_options += ["--with-crt=MD"]

print 'Job: %s' % os.environ.get('JOB_NAME', '')
print "Revision: %s" % os.environ.get('SVN_REVISION', '')
print "LD_LIBRARY_PATH: %s" % os.environ.get('LD_LIBRARY_PATH','')
install_path = "%s-%s-r%s" % (package_name,install_suffix,os.environ.get('SVN_REVISION', ''))

os.chdir(build_dir)
for f in glob.glob('%s-*' % package_name):
    if os.path.isdir(f):
        shutil.rmtree(f, True)
    else:
        os.remove(f)

check_call(["python", "waf", "distclean"])
check_call(["python", "waf", "configure", "--prefix=%s" % install_path] + config_options)
check_call(["python", "waf", "build"])
check_call(["python", "waf", "install"])

if os.path.isdir(install_path):
    shutil.make_archive(install_path, "zip", install_path)
