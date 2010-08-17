#!/usr/bin/python

import subprocess
import os.path
import re
import time
from Xlib import X, XK, display
from Xlib.ext import xtest

dpy = display.Display()
screen = dpy.screen()
root = screen.root

class Action:
    def __init__(self, action):
        self.action = action

    def doAction(self):
        subprocess.Popen(self.action, cwd=os.path.expanduser("~"), shell=True)
        
class KeyRecorder:
    def __init__(self, keys):
        self.keys = keys
        mask = X.LockMask
        mode = X.GrabModeAsync
        for type, modifier, key in self.keys:
            if type == 'key':
                root.grab_key(key, modifier, 0, mode, mode)
                root.grab_key(key, modifier|mask, 0, mode, mode)
            elif type == 'button':
                root.grab_button(key, modifier, root, X.ButtonPressMask, mode, mode, 0, 0)
                root.grab_button(key, modifier|mask, root, X.ButtonPressMask, mode, mode, 0, 0)

    def handleEvent(self, event):
        if event.type == X.KeyPress:
            type = 'key'
        elif event.type == X.ButtonPress:
            type = 'button'
        else:
            return False
        allModifierMask = X.ControlMask | X.Mod1Mask | X.ShiftMask
        mask = event.state & allModifierMask
        keycode = event.detail                    
        return (type, mask, keycode)

class KeyTrigger:
    def __init__(self, keyTriggerItems):
        self.keyRecorder = KeyRecorder(keyTriggerItems.keys())
        self.keyTriggerItems = keyTriggerItems

    def mainLoop(self):
        while True:
            event = dpy.next_event()
            self.handleEvent(event)
        
    def handleEvent(self, event):
        trigger = self.keyRecorder.handleEvent(event)
        if not trigger:
            return False
        if self.keyTriggerItems.has_key(trigger):
            self.keyTriggerItems[trigger].doAction()
            return True
        else:
            return False
    
keyTriggerItems = {
    ('key', X.Mod1Mask, dpy.keysym_to_keycode(XK.XK_Escape)): Action('~/prj/wmutil/launcher/launcher.py'),
    ('key', X.Mod1Mask, dpy.keysym_to_keycode(XK.XK_t)): Action('sakura'),
    }

if __name__ == '__main__':
    keyTrigger = KeyTrigger(keyTriggerItems)
    keyTrigger.mainLoop()
