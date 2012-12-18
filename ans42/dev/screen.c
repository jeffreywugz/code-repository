#include <phi/kernel.h>
#include <string.h>
#include <screen.h>

#define VIDEO_MEM_BASE  0xc00b8000

/* These define our textpointer, our background and foreground
 *  colors (attributes), and x and y cursor coordinates */
struct screen_t {
        int x, y;
        int attr;
        u16 *text;
};
struct screen_t *screen;

/* Scrolls the screen */
void scroll()
{
        unsigned blank, temp;

        /* A blank is defined as a space... we need to give it
         *  backcolor too */
        blank = 0x20 | (screen->attr << 8);

        /* Row 25 is the end, this means we need to scroll up */
        if(screen->y >= 25)
        {
                /* Move the current text chunk that makes up the screen
                 *  back in the buffer by a line */
                temp = screen->y - 25 + 1;
                memcpy (screen->text, screen->text + temp*80, (25-temp)*80*2);

                /* Finally, we set the chunk of memory that occupies
                 *  the last line of text to our 'blank' character */
                memsetw (screen->text + (25 - temp) * 80, blank, 80);
                screen->y = 25 - 1;
        }
}

/* Updates the hardware cursor: the little blinking line
 *  on the screen under the last character pressed! */
void move_csr(void)
{
        unsigned temp;

        /* The equation for finding the index in a linear
         *  chunk of memory can be represented by:
         *  Index = [(y * width) + x] */
        temp = screen->y * 80 + screen->x;

        /* This sends a command to indicies 14 and 15 in the
         *  CRT Control Register of the VGA controller. These
         *  are the high and low bytes of the index that show
         *  where the hardware cursor is to be 'blinking'. To
         *  learn more, you should look up some VGA specific
         *  programming documents. A great start to graphics:
         *  http://www.brackeen.com/home/vga */
        outb(14, 0x3D4);
        outb(temp >> 8, 0x3D5);
        outb(15, 0x3D4);
        outb(temp, 0x3D5);
}

/* Clears the screen */
void cls()
{
        unsigned blank;
        int i;

        /* Again, we need the 'short' that will be used to
         *  represent a space with color */
        blank = 0x20 | (screen->attr << 8);

        /* Sets the entire screen to spaces in our current
         *  color */
        for(i = 0; i < 25; i++)
                memsetw (screen->text + i * 80, blank, 80);

        /* Update out virtual cursor, and then move the
         *  hardware cursor */
        screen->x = 0;
        screen->y = 0;
        move_csr();
}

/* Puts a single character on the screen */
void putch(const char c)
{
        unsigned short *where;
        unsigned att = screen->attr << 8;

        /* Handle a backspace, by moving the cursor back one space */
        if(c == 0x08)
        {
                if(screen->x != 0) screen->x--;
        }
        /* Handles a tab by incrementing the cursor's x, but only
         *  to a point that will make it divisible by 8 */
        else if(c == 0x09)
        {
                screen->x = (screen->x + 8) & ~(8 - 1);
        }
        /* Handles a 'Carriage Return', which simply brings the
         *  cursor back to the margin */
        else if(c == '\r')
        {
                screen->x = 0;
        }
        /* We handle our newlines the way DOS and the BIOS do: we
         *  treat it as if a 'CR' was also there, so we bring the
         *  cursor to the margin and we increment the 'y' value */
        else if(c == '\n')
        {
                screen->x = 0;
                screen->y++;
        }
        /* Any character greater than and including a space, is a
         *  printable character. The equation for finding the index
         *  in a linear chunk of memory can be represented by:
         *  Index = [(y * width) + x] */
        else if(c >= ' ')
        {
                where = screen->text + (screen->y * 80 + screen->x);
                *where = c | att;	/* Character AND screen->attrutes: color */
                screen->x++;
        }

        /* If the cursor has reached the edge of the screen's width, we
         *  insert a new line in there */
        if(screen->x >= 80)
        {
                screen->x = 0;
                screen->y++;
        }

        /* Scroll the screen if needed, and finally move the cursor */
        scroll();
        move_csr();
}

/* Uses the above routine to output a string... */
void puts(const char *text)
{
        int i;

        for (i = 0; i < strlen(text); i++)
        {
                putch(text[i]);
        }
}

/* Sets the forecolor and backcolor that we will use */
void settextcolor(unsigned char forecolor, unsigned char backcolor)
{
        /* Top 4 bytes are the background, bottom 4 bytes
         *  are the foreground color */
        screen->attr = (backcolor << 4) | (forecolor & 0x0F);
}

void screen_init()
{
        static struct screen_t gscreen;
        screen=&gscreen;
        screen->attr=0x0F;
        screen->x=0;
        screen->y=0;
        screen->text = (unsigned short *)VIDEO_MEM_BASE;
        settextcolor(GREEN, BLACK);
        cls();
}
