#include<bits/stdc++.h>
using namespace std;
class student
{
      string f,l,t;
      int a,c;
      public:
      void set_age(int x)
      {
            a=x;
      }
      int get_age()
      {
            return a;
      }
      void set_fname(string y)
      {     f=y;  }
      string get_fname()
      {return f;}
      void set_lname(string z)
      {     l=z;}
      string get_lname()
      {return l;}
      void set_standard(int ss)
      {
            c=ss;
      }
      int get_standard()
      {
            return c;
      }
}s;
int main()
{
      int age,clas;
      string fname,lname,tt;
      cin>>age;
      cin>>fname;
      cin>>lname;
      cin>>clas;
      s.set_age(age);
      s.set_fname(fname);
      s.set_lname(lname);
      s.set_standard(clas);
      cout<<s.get_age();
      cout<<"\n"<<s.get_lname()<<", "<<s.get_fname();
      cout<<"\n"<<s.get_standard();cout<<"\n\n";
      cout<<s.get_age()<<","<<s.get_fname()<<","<<s.get_lname()<<","<<s.get_standard();
      return 0;
}