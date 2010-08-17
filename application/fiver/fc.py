#!/usr/bin/env python
#-*- encoding:utf-8 -*-

import gtk, gtk.glade
from config import *
import util,ai
import locale
print locale.getdefaultlocale()

fcboard=[[none for j in range(YCOUNTS)] for i in range(XCOUNTS)]
steps=[]

gladeFile = 'fc.glade'
window = gtk.glade.XML(gladeFile,'window')
drawingarea=window.get_widget("drawingarea")
board=gtk.gdk.pixbuf_new_from_file(BOARDIMAGE)
blackchess=gtk.gdk.pixbuf_new_from_file(BLACKCHESSIMAGE)
whitechess=gtk.gdk.pixbuf_new_from_file(WHITECHESSIMAGE)
drawingarea.set_size_request(board.get_width(),board.get_height())

def redraw(widget,event):
    boardpixmap=gtk.gdk.Pixmap(widget.window,BOARDWIDTH,BOARDHEIGHT)
    boardpixmap.draw_pixbuf(None,board,0,0,0,0)
    for x in  range(XCOUNTS):
        for y in range(YCOUNTS):
            chess=fcboard[x][y]
            if chess==black:
                chessimage=blackchess
            elif chess==white:
                chessimage=whitechess
            else:
                continue
            boardpixmap.draw_pixbuf(None,chessimage,
                    0,0,
                    x*BOARDXL+BOARDX0-BOARDXL/2,
                    y*BOARDYL+BOARDY0-BOARDYL/2)

    gc=widget.style.bg_gc[gtk.STATE_NORMAL]
    widget.window.draw_drawable(gc,boardpixmap,0,0,0,0,BOARDWIDTH,BOARDHEIGHT)
    return True

def check_win(fcboard,x,y):
    color=fcboard[x][y]
    for L in util.get_fclists(fcboard,x,y):
        for i in range(5):
            for j in L[i:i+5]:
                if j!=color:
                    break
            else:return color
    return none

def show_win(party):
    if party==none:
        return False
    if party==black:
        msg='black win'
    elif party==white:
        msg='white win'
    else:
        msg='internal error'

    msgbox=gtk.MessageDialog(parent=None, flags=gtk.DIALOG_MODAL,
            type=gtk.MESSAGE_INFO, buttons=gtk.BUTTONS_OK,
            message_format=msg)
    def response(w,e):
        w.destroy()
        start()
        return True
    msgbox.connect('response',response)
    msgbox.show()
    return True

def select(event):
    x=event.x-BOARDX0+BOARDXL/2
    y=event.y-BOARDY0+BOARDYL/2
    x=int(x/BOARDXL)
    y=int(y/BOARDYL)
    if fcboard[x][y]!=none:
        return
    fcboard[x][y]=black
    steps.append((x,y))
    drawingarea.queue_draw_area(0,0,BOARDWIDTH,BOARDHEIGHT)
    if show_win(check_win(fcboard,x,y)):
        return
    x,y=ai.get_move(fcboard)
    fcboard[x][y]=white
    steps.append((x,y))
    drawingarea.queue_draw_area(0,0,BOARDWIDTH,BOARDHEIGHT)
    if show_win(check_win(fcboard,x,y)):
        return

def start():
    for x in  range(XCOUNTS):
        for y in range(YCOUNTS):
            fcboard[x][y]=none
    global steps,first_party
    steps=[]
    if first_party==white:
        fcboard[MIDX][MIDY]=white
    drawingarea.queue_draw_area(0,0,BOARDWIDTH,BOARDHEIGHT)

def back():
    if not steps:
        return
    x,y=steps.pop()
    fcboard[x][y]=none
    x,y=steps.pop()
    fcboard[x][y]=none
    drawingarea.queue_draw_area(0,0,BOARDWIDTH,BOARDHEIGHT)

def on_button_release(widget,event):
    if event.button==1:
        select(event)

def on_radiobutton_human_toggled(widget):
    global first_party
    if widget.get_active():
        first_party=black
    else:
        first_party=white

signal_dict = {
    'on_window_destroy': gtk.main_quit,
    'on_drawingarea_button_release_event':on_button_release,
    'on_drawingarea_expose_event':redraw,
    'on_radiobutton_human_toggled':on_radiobutton_human_toggled,
    'on_toolbutton_new_clicked':lambda w:start(),
    'on_toolbutton_undo_clicked':lambda w:back(),
    'on_toolbutton_quit_clicked':gtk.main_quit,
}

window.signal_autoconnect(signal_dict)
start()
gtk.main()
