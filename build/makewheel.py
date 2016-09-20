from waflib.Build import BuildContext

import glob
import os
import shutil
import subprocess


class makewheel(BuildContext):
    '''builds a wheel for easy installation'''
    cmd='makewheel'
    fun='build'

    def execute(self):
        self.restore()
        if not self.all_envs:
            self.load_envs()

        targetsStr = '--targets=' + self.targets

        shutil.copyfile(os.path.join(self.getBuildDir(), 'six', 'conf',
            'setup.py'), 'setup.py')
        self.to_log('Creating wheel\n')
        subprocess.call(['pip', 'wheel', '.', '--wheel-dir', '.', '--no-deps'])
        os.remove('setup.py')
        numpyWheels = glob.glob('numpy*whl')
        for wheel in numpyWheels:
            os.remove(wheel)


