#!/usr/bin/env python

import os
import sys
import subprocess

sys.dont_write_bytecode = True
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jar_path_util import get_jar_path
bin_path = get_jar_path()

subprocess.call(['java', '-cp', bin_path, 'org.janelia.stitching.TilesToN5Converter'] + sys.argv[1:])