#define TARGET 1
#define N_X 4 
#define N_Y 4
#define N_GRID (N_X*N_Y)
#define N_NODES N_GRID*2

#define COST_TURN 1
#define COST_BACK 2

int base_cost[2][2]={
	{1, 1+COST_TURN}, 
	{1+COST_TURN, 1},
};

bool is_target(int* grid, int node)
{
	return grid[node/2]==TARGET;
}

int find_next(int* cost, int* gmask)
{
	int cost_min=INFINITY, min;
	for(i=0; i<N_NODES; i++){
		if(gmask[i])continue;
		if(cost[i]<cost_min){
			cost_min=cost[i];
			min=i;
		}
	}
	gmask[min]=true;
	return min;
}

void update(int* cost, int* last, int min)
{
	for(i=0, k=graph_adjoin[min][i]; k!=-1; i++, k=graph_adjoin[min][i]){
		cost_cur=cost[min]+graph_cost[min][i];
		if(cost_cur<cost[k]){
			cost[k]=cost_cur;
			last[k]=min;
		}
	}

}

void gen_adjoin_cost(int *grid, int graph_adjoin[][8], int graph_cost[][8])
{
	for(i=0; i<N_NODES; i++){
		gi=i/2; x=gi%X_N; y=gi/X_N;
		k=0;
		
#define add_adjoin(dx, dy, j) { \
		x_=x+dx; y_=y+dy; \
		if(x_>=0 && x_<X_N && y_>=0 && y_<Y_N){ \
			gi_=x_+y_*X_N; \
			graph_adjoin[i][k]=gi_*4+j; \
			graph_cost[i][k]=base_cost[j][i%4];\
			k++; \
		}
		add_adjoin_cost(0, -1, 0);
		add_adjoin_cost(0, +1, 0);
		add_adjoin_cost(-1, 0, 1);
		add_adjoin_cost(+1, 0, 1);
		graph_adjoin[i][k]=-1;
	}
}


int find_path(int* grid, int start)
{
	int min;
	gen_adjoin_cost(grid, graph_adjoin, graph_cost);
	for(i=0; i<N_NODES; i++){
		if(is_target(grid, min))
			return min;
		min=find_next(cost, gmask);
		update(cost, last, min);
	}
	return -1;
}

