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
        
class EdgeRecorder:
    POSITIONS = {'topleft': (0, 0), 'top': (1, 0), 'topright': (2, 0),
                 'left': (0, 1), 'right': (2, 1),
                 'bottomleft': (0, 2), 'bottom': (1, 2), 'bottomright': (2, 2),
                 }
    def __init__(self, edgeDelay=0.1, cornerGap=10):
        self.screenWidth = screen.width_in_pixels
        self.screenHeight = screen.height_in_pixels
        self.edgeDelay = edgeDelay
        self.cornerGap = cornerGap
        self.topWindow = self.createWindow(0, 0, self.screenWidth, 1)
        self.bottomWindow = self.createWindow(0, self.screenHeight-1, self.screenWidth, 1)
        self.leftWindow = self.createWindow(0, 0, 1, self.screenHeight)
        self.rightWindow = self.createWindow(self.screenWidth-1, 0, 1, self.screenHeight)
        self.edgeTriggerItems = edgeTriggerItems

    def createWindow(self, x, y, width, height):
        window = root.create_window(
            x, y, width, height, 0,
            X.CopyFromParent,
            X.InputOutput,
            X.CopyFromParent,
            override_redirect=True,
            event_mask = (X.ButtonPressMask | X.ButtonReleaseMask | X.PointerMotionMask),
            )
        window.map()
        return window

    def grab(self):
        self.topWindow.map()
        self.bottomWindow.map()
        self.leftWindow.map()
        self.rightWindow.map()

    def ungrab(self):
        self.topWindow.unmap()
        self.bottomWindow.unmap()
        self.leftWindow.unmap()
        self.rightWindow.unmap()

    def deliverClick(self, button):
        self.ungrab()
        dpy.sync()
        xtest.fake_input(dpy, X.ButtonPress, button)
        xtest.fake_input(dpy, X.ButtonRelease, button)
        dpy.sync()
        time.sleep(0.1)
        self.grab()
        
    @staticmethod
    def discretization(section, x):
        i = 0
        for boundary in section:
            if x < boundary:
                return i
            i += 1
        return i

    def position(self, x, y):
        x = self.discretization([self.cornerGap, self.screenWidth-self.cornerGap], x)
        y = self.discretization([self.cornerGap, self.screenHeight-self.cornerGap], y)
        for pos, xy in self.POSITIONS.items():
            if xy == (x, y):
                return pos

    def handleEvent(self, event):
        x, y = event.root_x, event.root_y
        if event.type == X.ButtonPress:
            button = event.detail
        elif event.type == X.ButtonRelease:
            self.deliverClick(event.detail)
            return None
        else:
            button = None
        pos = self.position(x, y)
        return pos, button

class StrokeRecorder:
    def __init__(self, button):
        self.button = button
        self.state='not started'
        self.grab()

    def grab(self):
        mode = X.GrabModeAsync
        root.grab_button(self.button, X.AnyModifier, root, X.ButtonMotionMask | X.ButtonReleaseMask, mode, mode, 0, 0)

    def ungrab(self):
        root.ungrab_button(self.button, X.AnyModifier)

    def handleEvent(self, event):
        if not (event.type == X.MotionNotify
                or ((event.type == X.ButtonPress or event.type == X.ButtonRelease)
                    and event.detail == self.button)):
            return False
        if self.state == "not started":
            if event.type == X.ButtonPress:
                self.points = []
                self.state = "started"
            elif event.type == X.ButtonRelease:
                return None
        elif self.state == "started":
            if event.type == X.ButtonRelease:
                self.state = "not started"
                return self.points
            elif event.type == X.MotionNotify:
                self.points.append((event.root_x, event.root_y))

class StrokeTrigger:
    def __init__(self, strokeTriggerItems):
        self.strokeTriggerItems = strokeTriggerItems
        self.strokeRecorder = StrokeRecorder(X.Button3)

    def pointsToStroke(self, points):
        print 'points:', points

    def ungrab(self):
        self.strokeRecorder.ungrab()

    def grab(self):
        self.strokeRecorder.grab()
        
    def handleEvent(self, event):
        points = self.strokeRecorder.handleEvent(event)
        if not points:
            return False
        stroke = self.pointsToStroke(points)
        if self.strokeTriggerItems.has_key(stroke):
            self.strokeTriggerItems[stroke].doAction()
        return True
            
keyTriggerItems = {
    ('key', X.Mod1Mask, dpy.keysym_to_keycode(XK.XK_Escape)): Action('~/bin/launcher/launcher.py'),
    ('key', X.Mod1Mask, dpy.keysym_to_keycode(XK.XK_t)): Action('sakura'),
    }

edgeTriggerItems={
    ('left', X.Button3): Action('rox'),
    ('topleft', X.Button3): Action('sakura'),
    }

strokeTriggerItems={
    ('left',): Action('sakura'),
    ('right',): Action('rox'),
    }


class Trigger:
    def __init__(self, keyTriggerItems, edgeTriggerItems, strokeTriggerItems):
        self.keyTriggerItems = keyTriggerItems
        self.edgeTriggerItems = edgeTriggerItems
        self.strokeTriggerItems = strokeTriggerItems

    def mainLoop(self):
        while True:
            event = dpy.next_event()
            print event.type
            self.handleEvent(event)
        
    def handleEvent(self, event):
        if not trigger:
            return False
        if self.triggerItems.has_key(trigger):
            self.triggerItems[trigger].doAction()
            return True
        else:
            return False

def deliverClick(self, event):
    button = event.detail
    dpy.sync()
    xtest.fake_input(dpy, X.ButtonPress, button)
    time.sleep(0.1)

if __name__ == '__main__':
    # keyTrigger = Trigger(KeyRecorder(keyTriggerItems.keys()), keyTriggerItems)
    # keyTrigger.mainLoop()
    edgeTrigger = Trigger(EdgeRecorder(), edgeTriggerItems)
    edgeTrigger.mainLoop()
