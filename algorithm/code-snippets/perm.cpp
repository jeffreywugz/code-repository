#include <iostream>

using namespace std;

void swap(char& a,  char& b)
{
	char c;
	c=a;
	a=b;
	b=c;
}

void output_list(char list[], int n)
{
	for(int i=0; i<n; i++)
		cout<<list[i];
	cout<<endl;
}

void perm_recursive(char list[], int n, int k)
{
	if(k==n-1)
		output_list(list, n);
	for(int i=k; i<n; i++){
		swap(list[k], list[i]);
		perm_recursive(list, n, k+1);
		swap(list[k], list[i]);
	}
}

void perm(char list[], int n)
{
	perm_recursive(list, n, 0);
}

int	main()
{
	char list[]={'a','b','c','d'};
	perm(list, sizeof(list));
	return 0;
}
