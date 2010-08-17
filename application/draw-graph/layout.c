#include <complex.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "layout.h"
#define INF (1<<30)
#define MAX_T 100000
#define MIN_T (MAX_T/3)
#define ADJUST 1.0
#define squre(x) ((x)*(x))
#define swap(x,y) {typeof(x) t=x;x=y; y=t;}

static int max_of(int* array, int n)
{
	int i, max=-INF;
	for(i=0; i<n; i++)
		if(max<array[i])max=array[i];
	return max;
}

double rand_f()
{
	return 1.0*rand()/RAND_MAX - 0.5;
}

static void rand_generate(double* pos, int n)
{
	int i;
	for(i=0; i<2*n; i++)
		pos[i]=rand_f();
}

static void rand_change(int n, double* src, double* dest)
{
	int k;
	memcpy(dest, src, sizeof(double)*2*n);
	k=rand()%(2*n);
	dest[k] = src[k]+rand_f()*ADJUST/n;
	dest[k] = dest[k]>.5? .5: (dest[k]<-.5?-.5:dest[k]);
}

//must be positive
static double eval(double* pos, int n, int (*edge)[MAX_NUM_VERTEXS], int m)
{
	int i, j;
	double L=1.0/n;
	double e=0.0, d;
	for(i=0; i<n; i++)
		for(j=i+1; j<n; j++){
			d=squre(pos[2*i]-pos[2*j])+squre(pos[2*i+1]-pos[2*j+1]);
			d+=0.000000001; //make d != 0
			e+=L/sqrt(d);
			if(edge[i][j])e+=d/sqrt(L);
		}
	return e;
}

int layout(int* edge_seq, int m, double* pos_seq)
{
	int T, n, i;
	double pos[2][MAX_NUM_VERTEXS*2], *pos_cur, *pos_new;
	double t, C, K;
	int edge[MAX_NUM_VERTEXS][MAX_NUM_VERTEXS];
	n = max_of(edge_seq, 2*m)+1;
	memset(edge, 0, sizeof(edge));
	for(i=0; i<2*m; i+=2){
		edge[edge_seq[i]][edge_seq[i+1]]=1;
		edge[edge_seq[i+1]][edge_seq[i]]=1;
	}

	srand(time(NULL));
	pos_cur=pos[0]; pos_new=pos[1];
	rand_generate(pos_cur, n);
	//C is here to let average(t*(MAX_T-T)/C) == 1
	C=0.00001;//make sure C != 0 
	K=1.0;
	for(T=MAX_T-1; T>MIN_T; T--){
		rand_change(n, pos_cur, pos_new);
		t=eval(pos_new,n,edge,m)-eval(pos_cur,n,edge,m);
		C+=fabs(t);
		if(rand_f()+0.5<exp(-t*((MAX_T-T)/C)*K*MAX_T/T))
			swap(pos_cur, pos_new);
	}

	for(i=0; i<2*n; i++)
		if(fabs(pos_cur[i])<0.0001)pos_cur[i]=0;
	memcpy(pos_seq, pos_cur, sizeof(double)*2*n);
	return 2*n;
}
