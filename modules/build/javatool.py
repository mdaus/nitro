import os
from os.path import join, isdir, abspath, dirname
from waflib import Options, Utils, Logs, TaskGen, Errors
from waflib.Errors import ConfigurationError
from waflib.TaskGen import task_gen, feature, after, before

def options(opt):
    opt.load('java')
    opt.add_option('--disable-java', action='store_false', dest='java',
                   help='Disable java', default=True)
    opt.add_option('--with-java-home', action='store', dest='java_home',
                   help='Specify the location of the java home')
    opt.add_option('--require-java', action='store_true', dest='force_java',
               help='Require Java (configure option)', default=False)
    opt.add_option('--require-jni', action='store_true', dest='force_jni',
               help='Require Java lib/headers (configure option)', default=False)
    opt.add_option('--require-ant', action='store_true', dest='force_ant',
               help='Require Ant (configure option)', default=False)
    opt.add_option('--with-ant-home', action='store', dest='ant_home',
                    help='Specify the Apache Ant Home - where Ant is installed')    

def configure(self):
    if not Options.options.java:
        return
    
    from build import recursiveGlob
    
    ant_home = Options.options.ant_home or self.environ.get('ANT_HOME', None)
    if ant_home is not None:
        ant_paths = [join(self.environ['ANT_HOME'], 'bin'), self.environ['ANT_HOME']]
    else:
        ant_paths = []

    env = self.env
    env['HAVE_ANT'] = self.find_program('ant', var='ANT', path_list=ant_paths, mandatory=False)

    if not env['ANT'] and  Options.options.force_ant:
        raise Errors.WafError('Cannot find ant!')

    if Options.options.java_home:
        self.environ['JAVA_HOME'] = Options.options.java_home 
    
    try:
        self.load('java')
    except Exception, e:
        if Options.options.force_java:
            raise e
        else:
            return

    if not self.env.CC_NAME and not self.env.CXX_NAME:
        self.fatal('load a compiler first (gcc, g++, ..)')

    try:
        if not self.env.JAVA_HOME:
            self.fatal('set JAVA_HOME in the system environment')
    
        # jni requires the jvm
        javaHome = abspath(self.env['JAVA_HOME'][0])
        
        if not isdir(javaHome):
            self.fatal('could not find JAVA_HOME directory %r (see config.log)' % javaHome)
    
        incDir = abspath(join(javaHome, 'include'))
        if not isdir(incDir):
            self.fatal('could not find include directory in %r (see config.log)' % javaHome)
        
        incDirs = list(set(map(lambda x: dirname(x),
                      recursiveGlob(incDir, ['jni.h', 'jni_md.h']))))
        libDirs = list(set(map(lambda x: dirname(x),
                      recursiveGlob(javaHome, ['*jvm.a', '*jvm.lib']))))
        if not libDirs:
            libDirs = list(set(map(lambda x: dirname(x),
                          recursiveGlob(javaHome, ['*jvm.so', '*jvm.dll']))))
    
        if not self.check(header_name='jni.h', define_name='HAVE_JNI_H', lib='jvm',
                    libpath=libDirs, includes=incDirs, uselib_store='JAVA', uselib='JAVA',
                    function_name='JNI_GetCreatedJavaVMs'):
            if Options.options.force_jni:
                self.fatal('could not find lib jvm in %r (see config.log)' % libDirs)
    except ConfigurationError, ex:
        err = str(ex).strip()
        if err.startswith('error: '):
            err = err[7:]
        if Options.options.force_java:
            self.fatal(err)
        else:
            self.msg('Java lib/headers', err, color='YELLOW')

# Used to call ant. Assumes the ant script respects a target property.
@task_gen
@feature('ant')
def ant(self):
    if not hasattr(self, 'defines'):
        self.defines = []
    if isinstance(self.defines, str):
        self.defines = [self.defines]
    self.env.ant_defines = map(lambda x: '-D%s' % x, self.defines)
    self.rule = ant_exec

def ant_exec(tsk):
    # Source file is build.xml
    cmd = ['\"' + tsk.env['ANT'] + '\"', '-file', tsk.inputs[0].abspath(), '-Dtarget=' + tsk.outputs[0].abspath()] + tsk.env.ant_defines
    return tsk.generator.bld.exec_command(cmd)

# Tell waf to ignore any build.xml files, the 'ant' feature will take care of them.
TaskGen.extension('build.xml')(Utils.nada)

