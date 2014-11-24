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

            # Now we can grab his defines
            libs += tsk.env.STLIB
            
            # Add use libs
            for uselib in tsk.uselib:
                if tsk.env['LIB_' + uselib]:
                    libs += tsk.env['LIB_' + uselib]

        # Sort and uniquify it, then print them all out
        libs = sorted(list(set(libs)))
        
        if len(libs) == 0:
            # If we found nothing print that we found nothing
            # otherwise it looks like the command failed.
            print 'No dependencies.'
        else:
            str = ''
            for lib in libs:
                if str:
                    str += ' '
                str += tsk.env.STLIB_ST % lib
            print str
