


def main():
    n,k=map(int,input().split())
    a=list(map(int,input().split()))
    cur_sum=sum(a[:k])
    max_sum=cur_sum
    cnt=1

    for i in range(k,n):
        cur_sum+=a[i]-a[i-k]
        if cur_sum>max_sum:
            max_sum=cur_sum
            cnt=1
        elif cur_sum==max_sum:
            cnt+=1
    print(max_sum,cnt)


if __name__ == '__main__':
    main()