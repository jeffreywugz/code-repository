#include <stdio.h>

int main ()
{
	const int N = 10;
	int i, j, l, sp;
	nt A[2][N];
	for (i = 0, l = 0, sp = N - 1; i < N; i++, l ^= 1, sp--) {
		A[l][0] = 1;
		for (j = 1; j < i; j++)
			A[l][j] = A[l ^ 1][j - 1] + A[l ^ 1][j];
		A[l][i] = 1;
		for (j = 0; j < sp; j++)
			printf ("\t");
		for (j = 0; j <= i; j++)
			printf ("%d\t\t", A[l][j]);
		putchar ('\n');
	}
	return 0;
}
