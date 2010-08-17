#!/usr/bin/python

import gtk
import os

builder=gtk.Builder()
builder.add_from_file('demo.glade')
win=builder.get_object('win')

signal_map={
    'win_exit':gtk.main_quit,
}
builder.connect_signals(signal_map)
gtk.main()
