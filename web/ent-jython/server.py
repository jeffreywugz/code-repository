#!/usr/bin/env python

import sys
sys.path.extend(["cherrypy.zip", "mako.zip"])
import msite

msite.server_start()
