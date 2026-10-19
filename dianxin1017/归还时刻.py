

def main():
    q=int(input())
    in_warehouse={}
    success=0
    reject=0
    current_stock=0
    total_stay=0
    for t in range(1,q+1):
        op,x=input().split()
        x=int(x)
        if op=='IN':
            if x in in_warehouse:
                reject+=1
            else:
                in_warehouse[x]=t
                current_stock+=1
                success+=1
        elif op=='OUT':
            if x not in in_warehouse:
                reject+=1
            else:
                stay=t-in_warehouse[x]
                total_stay+=stay
                del in_warehouse[x]
                success+=1
                current_stock-=1
    print(success,reject,total_stay,current_stock)


if __name__ == '__main__':
    main()