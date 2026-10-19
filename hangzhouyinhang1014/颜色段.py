#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
# 
# @param c int整型一维数组 
# @return long长整型
#
class Solution:
    def work(self , c: List[int]) -> int:
        mask=0
        count={0:1}
        res=0
        for color in c:
            mask^=(1<<color)
            if mask in count:
                res+=count[mask]
                count[mask]+=1
            else:
                count[mask]=1
        return res