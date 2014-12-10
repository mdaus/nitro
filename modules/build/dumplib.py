from waflib.Build import BuildContext

class dumplib(BuildContext):
    '''dumps the libs connected to the targets'''
    cmd = 'dumplib'
    fun = 'build'

    def execute(self):

        # Recurse through all the wscripts to find all available tasks
        self.restore()
        if not self.all_envs:
            self.load_envs()

        self.recurse([self.run_dir])
        
        targets = self.targets.split(',')
        libs = []
        
        for target in targets:
            # Find the target
            tsk = self.get_tgen_by_name(target)

            # Actually create the task object
            tsk.post()

            # Now we can grab his libs
            libs += tsk.env.STLIB
                
        # Now run again but add all the targets we found
        # This resolves running with multiple targets
        moduleDeps = ''
        for lib in libs:
            moduleDeps += str(lib).split('-')[0] + ' '

        # Add in the original targets again
        for target in targets:
            moduleDeps += str(target).split('-')[0] + ' '

        # Now run with the new module deps
        modArgs = globals()
        modArgs['NAME'] = 'dumplib'
        modArgs['MODULE_DEPS'] = moduleDeps
        
        # We need a source file here so it doesn't think it is headers only
        modArgs['SOURCE_DIR'] = 'build'
        modArgs['SOURCE_EXT'] = 'pyc'
        self.module(**modArgs)

        self.recurse([self.run_dir])
        
        libs = []
        tsk = self.get_tgen_by_name('dumplib-c++')
        tsk.post()
        libs = tsk.env.STLIB
        
        for uselib in tsk.uselib:
            if tsk.env['LIB_' + uselib]:
                libs += tsk.env['LIB_' + uselib]
        
        if len(libs) == 0:
            # If we found nothing print that we found nothing
            # otherwise it looks like the command failed.
            print 'No dependencies.'
        else:
            ret = ''
            for lib in libs:
                if ret:
                    ret += ' '
                ret += tsk.env.STLIB_ST % lib
            print ret
