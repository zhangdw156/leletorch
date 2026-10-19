from collections import defaultdict


def main():
    n=int(input())
    a=map(int,input().split())
    pos_dict=defaultdict(list)
    for idx,num in enumerate(a,start=1):
        pos_dict[num].append(idx)
    count=0
    total=0
    for num,positions in pos_dict.items():
        k=len(positions)
        if k<3:
            continue
        d=positions[1]-positions[0]
        is_stable=True
        for i in range(2,k):
            if positions[i]-positions[i-1]!=d:
                is_stable=False
                break
        if is_stable:
            count+=1
            total+=k
    print(count,total)

if __name__=='__main__':
    main()    