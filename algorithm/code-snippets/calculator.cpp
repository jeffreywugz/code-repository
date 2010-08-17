#include <iostream>
#include <cctype>
#include <stack>
#include <map>

using namespace std;

int	calculate(char *s);
int	operate(int a,int b,char op);
void	fatal(char *s);

int	main()
{
	char s[80];
	while(cin.getline(s,80))
		cout<<calculate(s)<<endl;
	return 0;
}

int	calculate(char *s)
{
#define N 7
	int prio[N][N]={
		//+, -, *, /, (, ), #
		{ 1, 1,-1,-1,-1, 1, 1, },
		{ 1, 1,-1,-1,-1, 1, 1, },
		{ 1, 1, 1, 1,-1, 1, 1, },
		{ 1, 1, 1, 1,-1, 1, 1, },
		{-1,-1,-1,-1,-1, 0, 2, },
		{ 1, 1, 1, 1, 2, 1, 1, },
		{-1,-1,-1,-1,-1, 2, 0, },
	};
	map<char,int> id_op;
	id_op['+']=0;id_op['-']=1;
	id_op['*']=2;id_op['/']=3;
	id_op['(']=4;id_op[')']=5;
	id_op['#']=6;
	stack<int> num_stack;
	stack<char> op_stack;
	op_stack.push('#');

	int i=0;
	int a,b;
	char op;
	while(s[i]){
		char c=s[i];
		if(isdigit(c)){
			num_stack.push(c-'0');
			i++;
			continue;
		}
		switch(prio[id_op[op_stack.top()]][id_op[c]]){
			case -1:
				op_stack.push(c);
				i++;
				continue;
			case 0:
				op_stack.pop();
				i++;
				continue;
			case +1:
				op=op_stack.top();
				op_stack.pop();
				if(num_stack.empty())fatal("no operand!\n");
				b=num_stack.top();
				num_stack.pop();
				if(num_stack.empty())fatal("no operand!\n");
				a=num_stack.top();
				num_stack.pop();
				num_stack.push(operate(a,b,op));
				break;
			default:
				fatal("error!\n");
		}
	}
	return num_stack.top();
}

int	operate(int a,int b,char op)
{
	switch(op){
		case '+':return a+b;
		case '-':return a-b;
		case '*':return a*b;
		case '/':return a/b;
		default: fatal("couldn't recognize!\n");
	}
}

void	fatal(char *s)
{
	fprintf(stderr,"%s",s);
}
