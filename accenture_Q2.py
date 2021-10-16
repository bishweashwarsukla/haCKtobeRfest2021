#iterative
n=int(input())
mod = 10**9+7
dp = [[0 for x in range(2)] for y in range((n+1)//2+1)]
dp[1][0]=21
dp[1][1]=5
for i in range(2,(n+1)//2+1):
    dp[i][0]=(((dp[i-1][0]+dp[i-1][1])%mod)*21)%mod;
    dp[i][1]=(dp[i-1][0]*5)%mod;

ans=0
if(n%2==0):
    ans=dp[n//2][0]
else:
    ans=(dp[(n+1)//2][0]+dp[(n+1)//2][1])%mod
print(ans)

#recursive
import sys
sys.setrecursionlimit(10**6)

c = 10**9 + 7
con = {}
vo = {}
    
def vowel(n):
    if n in vo:
        return(vo[n])
    else:
        cases =  5
        if n > 1:
            total_cases = cases * consonant(n-1)
            vo[n] = total_cases % c
            return(vo[n])
        else:
            return(cases)
    
def consonant(n):
    if n in con:
        return(con[n])
    else:
        cases = 21
        if n > 1:
            v_cases = cases * vowel(n-1)
            c_cases = cases * consonant(n-1)
            total_cases = v_cases + c_cases
            con[n] = total_cases % c
            return(con[n])
        else:
            return(cases)

t = int(input())
for _ in range(t):
    n = int(input())
    if n%2==0:
        m = n/2
        print((consonant(m)) % c)
    else:
        m = n//2 + 1
        print((consonant(m) + vowel(m)) % c)    