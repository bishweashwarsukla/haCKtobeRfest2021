#include<bits/stdc++.h>
using namespace std;
struct student
{
      int n,s;
      string f,l,t;
}s1;
int main()
{     
      cin>>s1.n;
      cin>>s1.f;
      cin>>s1.l;
      cin>>s1.s;
      s1.t=s1.f+" "+s1.l;
      if(s1.t.size()<50)
            cout<<s1.n<<" "<<s1.f<<" "<<s1.l<<" "<<s1.s;
      return 0;

      
}