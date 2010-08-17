#include <stdio.h>
#include <assert.h>
#include <gtk/gtk.h>

GdkPixmap *frame;
GdkPixbuf *background, *firefox;
GtkDrawingArea *drawing_area;
int width, height, obj_width, obj_height;
/* Note: if FRAME_DELAY is two short, this program will not work! */
int FRAME_DELAY=300;

static gboolean redraw(GtkWidget *widget,GdkEventExpose *event,gpointer data)
{
        gdk_draw_drawable(widget->window, widget->style->black_gc, frame,
                          0, 0, 0, 0, width, height);

        return TRUE;
}

static gint timeout(gpointer arg)
{
        static int x=0, y=0, dx=10, dy=10;
        gdk_draw_pixbuf(frame, NULL, background,
                        0, 0, 0, 0, width, height,
                        GDK_RGB_DITHER_NORMAL, 0, 0);

        x += dx; y += dy;
        if(x > width-obj_width || x < 0){
                dx = -dx;
                x += dx;
        }
        if(y > height-obj_height || y < 0){
                dy = -dy;
                y += dy;
        }
        gdk_draw_pixbuf(frame, NULL, firefox,
                        0, 0, x, y, obj_width, obj_height,
                        GDK_RGB_DITHER_NORMAL, 0, 0);
        gtk_widget_queue_draw(drawing_area);
        return TRUE;
}

void load_pixbuf()
{
        GError *error=NULL;
        firefox = gdk_pixbuf_new_from_file("firefox.png", &error);
        background = gdk_pixbuf_new_from_file("background.jpg", &error);
        width = gdk_pixbuf_get_width(background);
        height = gdk_pixbuf_get_height(background);
        obj_width = gdk_pixbuf_get_width(firefox);
        obj_height = gdk_pixbuf_get_height(firefox);
}

int main(int argc, char *argv[])
{
        GtkWidget *win;

        gtk_init(&argc, &argv);
        win=gtk_window_new(GTK_WINDOW_TOPLEVEL);
        gtk_window_set_title(win, "drawing");
        g_signal_connect(win, "destroy", gtk_main_quit, NULL);
        
        load_pixbuf();
        drawing_area = gtk_drawing_area_new();
        gtk_widget_set_size_request(drawing_area, width, height);
        g_signal_connect(drawing_area, "expose-event", redraw, NULL);
        gtk_container_add(win, drawing_area);
        gtk_widget_show_all(win);
        frame = gdk_pixmap_new(((GtkWidget*)drawing_area)->window, width, height, -1);
        g_timeout_add(FRAME_DELAY, timeout, NULL);

        gtk_main();
        return 0;
}
