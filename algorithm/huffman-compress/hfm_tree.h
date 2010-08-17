#ifndef  __HFM_TREE_H__
#define  __HFM_TREE_H__




struct HfmTree {
	short weight;
	unsigned c;
	struct HfmTree *lchild, *rchild;
};

struct HfmTree* create_hfm_tree(unsigned short* weights);
void destroy_hfm_tree(struct HfmTree* root);
void print_hfm_tree(struct HfmTree* root, int depth);

#endif  /*__HFM_TREE_H__*/
#define  N_HFM_LEAF 256
