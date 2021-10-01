#include<bits/stdc++.h>
using namespace std;
int main()
{
    vector <int> nums {3,1,3,4,2};
    vector <int> c;
        for(int i=0;i<nums.size();i++)
        {
            auto it=find(c.begin(),c.end(),nums[i]);
            if(it!=c.end())
                int k=0;
            else
            {
                c.push_back(nums[i]);
                auto ff=count(nums.begin(),nums.end(),nums[i]);
                if( ff>1 )
                {
                    cout<<nums[i]<<" ";
                }
            }

        }
    return 0;
}