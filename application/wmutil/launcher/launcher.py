#!/usr/bin/python

import sys
import os, os.path
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))
         
import gtk
import gobject
import subprocess
import complete
         
glade_file = 'ui.glade'

class Completion(gobject.GObject):
    def __init__(self):
        gobject.GObject.__init__(self)
        self.items=[]
        self.selected=0

    def set_items(self, items):
        self.items=items
        self.selected=0
        self.emit('refresh')

    def get_text(self):
        items=[item[0] for item in self.items]
        if not items:
            text='<span foreground="red">no candidates!</span>'
        else:
            text='%s <span foreground="green" background="black">%s</span> %s'%(
                ' '.join(items[:self.selected]),
                items[self.selected],
                ' '.join(items[self.selected+1:]))
        return text

    def get_active_item(self):
        if self.items:
            return self.items[self.selected][2]
        else:
            return None
    
    def next_item(self):
        self.selected = (self.selected+1)%len(self.items)
        self.emit('refresh')
        
    def prev_item(self):
        self.selected = (self.selected+len(self.items)-1)%len(self.items)
        self.emit('refresh')

gobject.signal_new('refresh', Completion, gobject.SIGNAL_RUN_FIRST, gobject.TYPE_NONE, ())
completion=Completion()
builder=gtk.Builder()
builder.add_from_file(glade_file)
win=builder.get_object('win')
win.set_keep_above(True)
win.set_size_request(gtk.gdk.screen_width(), -1)
win.move(0, 0)
label=builder.get_object('label')

def on_complete_refresh(comp):
    text=comp.get_text()
    label.set_markup(text)

def on_entry_changed(widget):
    prefix=widget.get_text()
    items=complete.get_candidates(prefix)
    completion.set_items(items)
    
def on_entry_key_press_event(widget, event):
    if event.state & gtk.gdk.CONTROL_MASK:
        if gtk.gdk.keyval_name(event.keyval) == "s":
            completion.next_item()
            return True
        if gtk.gdk.keyval_name(event.keyval) == "r":
            completion.prev_item()
            return True

    if event.keyval == gtk.keysyms.Escape:
        gtk.main_quit()
    return False

def on_entry_activate(widget):
    cmd=completion.get_active_item()
    subprocess.Popen(cmd, cwd=os.path.expanduser("~"), shell=True)

def on_entry_focus_in(widget, event):
    widget.emit('changed')
    
signal_map={
    'win_exit': gtk.main_quit,
    'on_entry_changed': on_entry_changed,
    'on_entry_key_press_event': on_entry_key_press_event,
    'on_entry_activate': on_entry_activate,
    'on_entry_focus_in_event':  on_entry_focus_in,
    'on_entry_focus_out_event': gtk.main_quit,
}

completion.connect("refresh", on_complete_refresh)
builder.connect_signals(signal_map)
gtk.main()

