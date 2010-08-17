#!/usr/bin/python

import re

comp_items=(
    ('halt', 0, 'sudo halt'),
    ('reboot', 0, 'sudo reboot'),
    ('hibernate', 0, 'sudo pm-hibernate'),
    ('stardict', 0, 'stardict'),
    ('ario', 0, 'mpd; ario'),
    ('sakura', 0, 'sakura'),
    ('dia', 0, 'dia'),
    ('swriter', 0, 'swriter'),
    ('rox', -1, 'rox'),
    ('lxtask', 0, 'lxtask'),
    ('qq', 0, 'qq'),
    ('gvim', 0, 'gvim'),
    ('audacious', 0, 'audacious'),
    ('devhelp', 0, 'devhelp'),
    ('glade', -1, 'glade-3'),
    ('gtk-demo', 0, 'gtk-demo'),
    ('pygtk-demo', 0, 'pygtk-demo'),
)
def get_candidates(prefix):
    candidates=[item for item in comp_items if re.match(prefix, item[0])]
    candidates.sort(key=lambda x:x[1])
    return candidates
