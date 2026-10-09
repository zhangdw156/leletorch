import bisect


def main(n):
    if n<4:
        return 0
    is_prime=[True]*(n+1)
    is_prime[0]=is_prime[1]=False
    for i in range(2,int(n**0.5)+1):
        if is_prime[i]:
            for j in range(i*i,n+1,i):
                is_prime[j]=False
    primes=[i for i in range(2,n+1) if is_prime[i]]
    ans=0
    if n>=4:
        ans+=1
    half=n//2
    for p in primes:
        if p==2:
            continue
        if p>half:
            break
        left=bisect.bisect_left(primes,p)
        right=bisect.bisect_right(primes,n-p)
        ans+=right-left
    return ans    
if __name__=='__main__':
    n=int(input())
    print(main(n))