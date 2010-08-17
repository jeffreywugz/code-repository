#include <stdlib.h>
#include "util.h"
#include "prio_queue.h"
#include "hfm_tree.h"

static void create_hfm_leaf(struct HfmTree* tree[], unsigned short* weights)
{
	int i;
	for(i=0; i<N_HFM_LEAF; i++){
		tree[i]=malloc(sizeof(struct HfmTree));
		if(tree[i]==NULL)fatal("malloc error", MEM_ERR);
		tree[i]->lchild=tree[i]->rchild=NULL;
		tree[i]->weight=weights[i];
		tree[i]->c=i;
	}
}

static void merge_hfm_tree(struct HfmTree** root, struct HfmTree* child1, 
		struct HfmTree* child2)
{
	*root=malloc(sizeof(struct HfmTree));
	if(*root==NULL)fatal("malloc error", MEM_ERR);
	(*root)->lchild=child1;
	(*root)->rchild=child2;
	(*root)->weight=child1->weight+child2->weight;
}

struct HfmTree* create_hfm_tree(unsigned short* weights)
{
	struct HfmTree *tree[N_HFM_LEAF], *child1, *child2, *root=NULL;
	struct PrioQueue prio_queue;
	create_hfm_leaf(tree, weights);
	init_prio_queue(&prio_queue, tree, N_HFM_LEAF);
	while(1){
		de_prio_queue(&prio_queue, &root);
		if(root->weight!=0)break;
	}
	en_prio_queue(&prio_queue, root);//care of prio_queue.len==1
	while(prio_queue.len>1){
		de_prio_queue(&prio_queue, &child1);
		de_prio_queue(&prio_queue, &child2);
		merge_hfm_tree(&root, child1, child2);
		en_prio_queue(&prio_queue, root);
	}
	return root;
}

void destroy_hfm_tree(struct HfmTree* root)
{
	if(root==NULL)return;
	destroy_hfm_tree(root->lchild);
	destroy_hfm_tree(root->rchild);
	free(root);
}

#include <stdio.h>
void print_hfm_tree(struct HfmTree* root, int depth)
{
	int i;
	if(root==NULL)return;
	print_hfm_tree(root->lchild, depth+1);
	for(i=0; i<depth; i++)
		printf("   ");
	printf("%d", root->weight);
	if(root->lchild==NULL) putchar(root->c);
	putchar('\n');
	print_hfm_tree(root->rchild, depth+1);
}
/*
int main()
{
	struct HfmTree* root;
	int weights[N_HFM_LEAF];
	int i;
	for(i=0; i<N_HFM_LEAF; i++)
		weights[i]=i;
	root=create_hfm_tree(weights);
	print_hfm_tree(root, 0);
	destroy_hfm_tree(root);
	return 0;
}
*/
