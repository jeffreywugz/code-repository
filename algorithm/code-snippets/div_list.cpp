#include  <iostream>
#include  <list>
#include  <cmath>

using	namespace std;

typedef	list<list<int> > div_list;
div_list get_div_list(int n)
{
	list<list<int> >::iterator p;
	list<list<int> >::iterator q;
	div_list ret_ls, ls;
	list<int> tls;
	tls.push_front(n);
	ret_ls.push_front(tls);
	for(int i=2;i<=sqrt(n);i++){
		if(n%i)continue;
		ls=get_div_list(n/i);
		for(p=ls.begin();p!=ls.end();p++){
			if(*p->begin()<i)continue;
			p->push_front(i);
			ret_ls.insert(ret_ls.end(), *p);
		}
	}
	return ret_ls;
}

int	main()
{
	int n;
	cin>>n;
	div_list ls=get_div_list(n);
	list<list<int> >::iterator p;
	list<int>::iterator p1;
	for(p=ls.begin(); p!=ls.end(); p++){
		for(p1=p->begin(); p1!=p->end(); p1++)
			cout<<*p1<<' ';
		cout<<endl;
	}
	list<int>a;
	return 0;
}
