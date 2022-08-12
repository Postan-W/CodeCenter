"""
@Time : 2021/8/31 15:32
@Author : wmingzhu
@Annotation : 
"""



#题目：给定一个序列，求具有最大和的连续子序列

#解法1，滑动窗口,时间复杂度n^3
def sliding_window(source:list):
    beginning = 0
    maximum = source[0]
    length = 1
    for i in range(1,len(source)+1):#该层循环为窗口大小
        max = 0
        local_beginning = 0
        for j in range(len(source)):
            if j + i - 1 > (len(source)-1):
                break
            sum = 0
            for k in range(j,j+i):
                sum += source[k]

            if sum > max:
                max = sum
                local_beginning = j

        if max > maximum:
            maximum = max
            beginning = local_beginning
            length = i

    return (maximum,[source[i] for i in range(beginning,beginning+length)])#返回最大值和子序列

print(sliding_window([-2,1,-3,4,-1,2,1,8,-5,4]))
print(sliding_window([1,2,3,4,5,6,7]))

#使用动态规划的思想
'''使用动态规划的思想解决需要满足两个特性，一是可递推，大问题可转化表示为小问题；二是由小问题的最优解可以得到大问题的最优解。首先表明一个事实：要求的子序列必以原序列中的某个元素结尾。设原序列为P;s[k]为P前k个元素以第k个元素为结尾的最长连续子序的和，那么s[k+1] = max(s[k-1,0) + P[k],其中P[k]为P的第k+1个元素。由此可以求得每个s[k]，k属于[1,n]，其中最大的那个就是原序列要求的目标子序列。'''
def DP_solution(source:list):
    s = [0 for i in range(len(source))]
    s[0] = source[0]
    max_sum = s[0]
    for k in range(1,len(source)):
        s[k] = max(s[k-1],0) + source[k]
        max_sum = max_sum if max_sum > s[k] else s[k]

    return max_sum

print(DP_solution([-2,1,-3,4,-1,2,1,8,-5,4]))
print(DP_solution([1,2,3,4,5,6,7]))

