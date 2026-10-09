import sys
import math


if __name__=='__main__':
    size=int(input())
    n=math.pow(2,size)
    res=[]
    binary_str=''
    for _ in  range(int(n)):
        s=input()
        for ch in s:
            if ch=='.':
                binary_str+='0'
            else:
                binary_str+='1'
    for i in range(0,len(binary_str),4):
        res.append(str(int(binary_str[i:i+4],2)))
    print(' '.join(res))