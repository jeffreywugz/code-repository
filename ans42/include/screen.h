#ifndef SCREEN_H
#define SCREEN_H

#define BLACK 0
#define BLUE 1
#define GREEN 2
#define CYAN 3
#define RED 4
#define MAGENTA 5
#define BROWN 6
#define LGREY 7
#define DGREY 8
#define LBLUE 9
#define LGREEN 10
#define LCYAN 11
#define LRED 12
#define LMAGENTA 13
#define LBROWN 14
#define WHITE 15

void cls();
void putch(const char c);
void puts(const char *str);
void settextcolor(unsigned char forecolor, unsigned char backcolor);
void init_screen();

                      
#endif //SCREEN_H
