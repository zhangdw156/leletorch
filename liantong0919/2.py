def max_depth(s):
    st=[]
    depth=0
    max_depth=0
    for ch in s:
        if ch in {'(','[','{'}:
            st.append(ch)
            depth+=1
            max_depth=max(max_depth,depth)
        elif ch in {')',']','}'}:
            if not st:
                return -1
            top=st.pop()
            if (top=='(' and ch!=')') or (top=='[' and ch!=']') or (top=='{' and ch!='}'):
                return -1
            depth-=1
    return max_depth if depth==0 else -1
        


if __name__ == "__main__":
    s = input()
    print(max_depth(s))