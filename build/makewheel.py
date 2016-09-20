from waflib.Build import BuildContext
from waflib import Errors

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

        if not os.path.exists('setup.py'):
            raise Errors.WafError('Could not make wheel. setup.py not found.')
        self.to_log('Creating wheel\n')
        subprocess.call(['pip', 'wheel', '.', '--wheel-dir', '.', '--no-deps'])
        wheels = glob.glob('*.whl')
        for wheel in wheels:
            shutil.move(wheel, os.path.join(self.env['install_bindir'], wheel))


