#include<bits/stdc++.h>
using namespace std;
int main()
{
    int T;
    cin>>T;
    while(T--)
    {
        int N,z;
        cin>>N;
        vector <int> vec,res;
        while(N--)
        {
            cin>> z;
            vec.push_back(z);
        }
        int minn=1,max=0;
        for(int i=0;i<vec.size()-1;i++)
        {   
            minn=1;
            for(int j=i;j<vec.size();j++)
            {
                if( (vec[j+1]-vec[j])<=2 )
                    minn+=1;
            }
            if(max<minn)
                max=minn;
           // res.push_back(minn);
        }
        cout<<minn<<" "<<max<<endl;
    }
    return 0;
}