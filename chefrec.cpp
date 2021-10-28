#include<bits/stdc++.h>
using namespace std;
int main()
{
    int t,n;
    cin>>t;
    while(t--)
    {
        vector <int> vec,cou;
        vector <int> ::iterator fin;
        cin>>n;
        while(n--)
        {
            int m,counts;
            cin>> m;
            vec.push_back(m); 
        }
        for(int i=0;i<vec.size();i++)
        {
            
            int z=count(vec.begin(),vec.end(),vec[i]);// a[i] is counted throughtout
            for(int j=0;j<vec.size()-1;j++)
            {
                int temp;
                int counts=0;
                temp=vec[j];
                while(temp!=vec[j+1])
                {
                    counts+=1;
                }
                fin=find(cou.begin(),cou.end(),counts);
                if(fin!=cou.end())
                    cout<<"No first"<<endl; 
                else
                    {
                    cou.push_back(counts);
                    if (z==counts)
                        cout<<"Yes"<<endl;
                    else
                        cout<<"No"<<endl;
                    }
            }
        }
    }
    return 0;
}