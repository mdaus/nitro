from waflib.Build import BuildContext

import glob
import os
import shutil
import subprocess
from zipfile import ZipFile

class package(BuildContext):
    '''Creates a zip file of installation dir, and any wheels'''
    cmd='package'
    fun='build'

    def execute(self):
        self.restore()
        if not self.all_envs:
            self.load_envs()

        targetsStr = '--targets=' + self.targets

        self.to_log('Zipping installation\n')
        wheels = glob.glob('*.whl')
        installDir = os.path.dirname(self.env['install_bindir'])

        for wheel in wheels:
            shutil.copy(wheel, os.path.join(installDir, wheel))

        shutil.make_archive('install', 'zip', None, installDir)

        for wheel in wheels:
            os.remove(os.path.join(installDir, wheel))

