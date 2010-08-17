#include <stdio.h>
#include <string.h>
#include "util.h"
#include "bits_io.h"
#include "hfm_tree.h"

#define  HC_BLOCK (1<<16)
struct HcHeader {
	int orig_len, store_len;
	unsigned short weight[N_HFM_LEAF];
};
static void hc();
static void uhc();
static void hc_block(struct BitStream* ibits, struct BitStream* obits,
		struct HcHeader* header);
static void uhc_block(struct BitStream* ibits, struct BitStream* obits, 
		struct HcHeader* header);

int main(int argc, char* argv[])
{
	if(argc==1) hc();
	else uhc();
	return 0;
}

void hc()
{
	struct BitStream ibits, obits;
	struct HcHeader header;
	while(1){
		init_bits(&ibits);
		init_bits(&obits);
		if(load_bits(&ibits, stdin, HC_BLOCK)==0)
			break;
		hc_block(&ibits, &obits, &header);
		fwrite(&header, sizeof(header), 1, stdout);
		unload_bits(&obits, stdout);
	}
}

void uhc()
{
	struct BitStream ibits, obits;
	struct HcHeader header;
	int len;
	while(1){
		init_bits(&ibits);
		init_bits(&obits);
		fread(&header, sizeof(header), 1, stdin);
		len=header.store_len;
		if(load_bits(&ibits, stdin, len)==0)break;
		uhc_block(&ibits, &obits, &header);
		unload_bits(&obits, stdout);
	}
}


static void get_weights(unsigned short* weights, unsigned char* buf, int len)
{
	int i;
	memset(weights, 0, sizeof(short)*N_HFM_LEAF);
	for(i=0; i<len; i++)
		weights[buf[i]]++;
}
static void get_hfm_code_recursive(struct Code* code, struct HfmTree* root, 
		int buf, int len)
{
	if(root->lchild==NULL){
		code[root->c].code=buf;
		code[root->c].len=len;
		return;
	}
	get_hfm_code_recursive(code, root->lchild, buf, len+1);
	buf |= 1<<len;
	get_hfm_code_recursive(code, root->rchild, buf, len+1);
}

static void get_hfm_code(struct Code* code, struct HfmTree* root)
{
	if(root==NULL)return;
	get_hfm_code_recursive(code, root, 0, 0);
}

static int hfm_decode(struct BitStream* bits, struct HfmTree* root)
{
	int c;
	while(1){
		if(root->lchild==NULL)return root->c;
		c=get_bit(bits);
		if(c==-1)fatal("decode: eof error", EOF_ERR);
		if(c==0)root=root->lchild;
		else root=root->rchild;
	}
}

void hc_block(struct BitStream* ibits, struct BitStream* obits,
		struct HcHeader* header)
{
	int c;
	struct Code code[N_HFM_LEAF];
	get_weights(header->weight, ibits->bits, (ibits->n+7)/8);
	struct HfmTree* root=create_hfm_tree(header->weight);
	get_hfm_code(code, root);
	destroy_hfm_tree(root);
	while((c=getchar_bits(ibits))!=-1)
		write_bits(obits, code+c);
	header->orig_len = (ibits->n+7)/8; 
	header->store_len = (obits->n+7)/8;
}

void uhc_block(struct BitStream* ibits, struct BitStream* obits,
		struct HcHeader* header)
{
	int c, i;
	struct HfmTree* root=create_hfm_tree(header->weight);
	for(i=0; i<header->orig_len; i++){
		c=hfm_decode(ibits, root);
		putchar_bits(obits, c);
	}
	destroy_hfm_tree(root);
}
