all: test

test: pthread_key
	./$<
pthread_key: pthread_key.c
	gcc --std=c99 -o $@ $< -lpthread
