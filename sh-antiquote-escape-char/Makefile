all: test
echo: echo.c
	gcc -o echo $^ --std=c99
test: echo
	for i in `ls --color *`; do ./echo $$i; done