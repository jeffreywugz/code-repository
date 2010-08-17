#!/usr/bin/env python
#-*- encoding:utf-8 -*-

import gtk, gtk.glade
import threading,os
import solver
solveFunction = solver.cppSolve
path=None
state=[
    1,2,3,4,
    5,7,16,8,
    9,6,10,12,
    13,14,11,15
]
initialState=[
    1,2,3,4,
    5,6,7,8,
    9,10,11,12,
    13,14,15,16
]
gladeFile = 'ui.glade'
fname='bg.jpg'
win= gtk.glade.XML(gladeFile,'win')
drawingarea=win.get_widget("drawingarea")

msgbox=gtk.MessageDialog(parent=None, flags=gtk.DIALOG_MODAL,
                type=gtk.MESSAGE_INFO, buttons=gtk.BUTTONS_OK)
def response(w,e):
    w.hide()
    return True
msgbox.connect('response',response)

def init():
    global state,path
    state=initialState[:]
    path=None

def load(fname):
    global image,width,height,xlen,ylen
    try:
        im=gtk.gdk.pixbuf_new_from_file(fname)
    except Exception,e:
        return
    image=im
    width,height=image.get_width(),image.get_height()
    xlen,ylen=int(width/4),int(height/4)
    width,height=xlen*4,ylen*4

def redraw():
    drawingarea.set_size_request(width,height)
    pixmap=gtk.gdk.Pixmap(drawingarea.window,width,height)
    for i in range(16):
        j=state[i]-1
        if j==15:
            pos=i
        pixmap.draw_pixbuf(None,image,(j%4)*xlen,(j/4)*ylen,
                (i%4)*xlen,(i/4)*ylen,xlen,ylen)

    gc=pixmap.new_gc()
    gc.line_width=2
    gc.foreground=pixmap.get_colormap().alloc_color("blue")
    for x in range(5):
        pixmap.draw_line(gc,x*xlen,0,x*xlen,height)
    for y in range(5):
        pixmap.draw_line(gc,0,y*ylen,width,y*ylen)
        
    gc.line_width=3
    gc.foreground=pixmap.get_colormap().alloc_color("red")
    pixmap.draw_rectangle(gc,False,(pos%4)*xlen,(pos/4)*ylen,xlen,ylen)
    gc=drawingarea.style.bg_gc[gtk.STATE_NORMAL]
    drawingarea.window.draw_drawable(gc,pixmap,0,0,0,0,width,height)

def move(widget,event):
    global state,path
    if event.button!=1:return
    x,y=event.x,event.y
    j=int(x/xlen)+int(y/ylen)*4
    for i in range(16):
        if state[i]==16:break
    state[i],state[j]=state[j],state[i]
    path=None
    drawingarea.queue_draw_area(0,0,width,height)

def showMsg(str):
    msgbox.set_property("text",str)
    msgbox.show()

def stop():
    global finished
    if not finished:
        os.system("pkill -9 jigsaw")
        showMsg("time out!")
        reset()

def forward():
    global iter,path,state,finished
    if not path:
        timer=threading.Timer(3.0,stop)
        timer.start()
        finished=False
        success,path=solveFunction(state)
        finished=True
        if not success:
           showMsg("can't solve!")
           reset()
           return
        iter=0
    i=len(path)-1
    if iter>=i:return
    iter+=1
    state=path[iter]
    drawingarea.queue_draw_area(0,0,width,height)

def back():
    global path,iter,state
    if not path:return
    if iter<=0:return
    iter-=1
    state=path[iter]
    drawingarea.queue_draw_area(0,0,width,height)

def set_fname(widget):
    global fname
    fname=widget.get_filename()
    load(fname)
    reset()

def reset():
    init()
    drawingarea.queue_draw_area(0,0,width,height)

signal_dict = {
    'on_win_destroy': gtk.main_quit,
    'on_drawingarea_expose_event': lambda w,e:redraw(),
    'on_drawingarea_button_release_event': move,
    'on_back_button_clicked':lambda w:back(),
    'on_forward_button_clicked':lambda w:forward(),
    'on_clear_button_clicked':lambda w:reset(),
    'on_filechooserbutton_file_set': set_fname,
}

load(fname)
init()
drawingarea.set_size_request(width,height)
win.signal_autoconnect(signal_dict)
gtk.main()
