#include <assert.h>
#include <iostream>
#include <iterator>
#include <vector>
#include <queue>
#include <stack>
#include <set>
#include <string.h>

using namespace std;

class State;
class StateTable {
public:
	~StateTable();
	State* create();
private:
	vector<State*> table;
};

class State {
public:
	void setState(int *state,int cost,State* parent)
	{
		this->cost=cost;
		this->parent=parent;
		memcpy(this->state,state,sizeof(this->state));
		int i;
		for(i=0; i<16;i++)
			if(state[i]==16)break;
		pos=i;
	};
	bool canSolve()
	{
		int reverse=0;
		for(int i=0; i<16; i++)
			for(int j=i+1; j<16; j++)
				if(state[i]>state[j])reverse++;
		return (reverse+tab[pos])%2==0;
	}
	bool isAnswer() { return restCost()==0; }
	//int allCost() const{ return cost+restCost(); }
	int allCost() const{ return restCost(); }
	vector<State*>& children(vector<State*>& chs,StateTable& stateTable)
	{
		for(int i=0; move[pos][i]; i++){
			swap(state[pos],state[pos+move[pos][i]]);
			State* p=stateTable.create();
			p->setState(state,cost+1,this);
			chs.push_back(p);
			swap(state[pos],state[pos+move[pos][i]]);
		}
		return chs;
	}
	friend ostream& operator<<(ostream& os,State& s)
	{
		copy(s.state,s.state+16,ostream_iterator<int>(os," "));
		return os;
	}
	State* parent;
	int state[16];
private:
	int restCost() const
	{
		int rCost=0;
		for(int i=0; i<16; i++)
			if(state[i]!=16&&state[i]!=i+1)rCost++;
		return rCost;
	}

	const static int tab[16];
	const static int move[16][5];
	int pos;
	int cost;
};

StateTable::~StateTable()
{
	for(vector<State*>::iterator p=table.begin();
			p!=table.end();p++)
		delete *p;
}
State* StateTable::create()
{
	State* p=new State;
	table.push_back(p);
	return p;
}

struct ltstate
{
	bool operator()(const State* s1, const State* s2) const
	{
		return s1->allCost()>=s2->allCost();
	}
};
struct ltstate1
{
	bool operator()(const State* s1, const State* s2) const
	{
		int i;
		for(i=0; i<16; i++)
			if(s1->state[i]!=s2->state[i])break;
		return i<16?s1->state[i]<s2->state[i]:false;
	}
};

const int State::tab[16]={
    0,1,0,1,
    1,0,1,0,
    0,1,0,1,
    1,0,1,0
};
const int State::move[16][5]={
    {1,4,0}, {1,-1,4,0}, {1,-1,4,0}, {-1,4,0},
    {1,4,-4,0}, {1,-1,4,-4,0}, {1,-1,4,-4,0}, {-1,4,-4,0},
    {1,4,-4,0}, {1,-1,4,-4,0}, {1,-1,4,-4,0}, {-1,4,-4,0},
    {1,-4,0}, {1,-1,-4,0}, {1,-1,-4,0}, {-1,-4,0},
};

bool search(State* root,stack<State*>& path,StateTable& stateTable)
{
	if(!root->canSolve())return false;
	priority_queue<State*,vector<State*>,ltstate> open;
	set<State*,ltstate1> stateSet;
	State* enode;
	open.push(root);
	while(!open.empty()){
		enode=open.top();
		open.pop();
		if(enode->isAnswer())break;
		if(stateSet.find(enode)!=stateSet.end())continue;
		stateSet.insert(enode);
		vector<State*> chs;
		enode->children(chs,stateTable);
		for(vector<State*>::iterator p=chs.begin();p!=chs.end();p++){
			open.push(*p);
		}
	}
	if(!enode->isAnswer())return false;
	for(State* p=enode; p; p=p->parent){
		path.push(p);
	}
	return true;
}

int	main()
{
	StateTable stateTable;
	int s[16];
	copy(istream_iterator<int>(cin),istream_iterator<int>(),s);
	State* root=stateTable.create();
	root->setState(s,0,0);
	stack<State*> path;
	if(!search(root,path,stateTable)){
		cout<<"no answer!"<<endl;
		return -1;
	}
	cout<<"answer:"<<endl;
	while(!path.empty()){
		State* p=path.top();
		path.pop();
		cout<<(*p)<<endl;
	}
	return 0;
}
