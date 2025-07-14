from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main() -> None:
    # 구현하세요!
    input = sys.stdin.readline
    T = int(input())
    for _ in range(T) : 
        n,m = map(int, input().split())
        total = n+m 
        tree : SegmentTree[int,int] = SegmentTree(size=total, default=0, func=lambda a,b : a+b)

        pos = [0] * (n+1) 
        for i in range(1, n+1): 
            pos[i] = m+i-1 
            tree.update(pos[i],1)
        
        top = m-1 
        query = list(map(int,input().split()))
        for q in query: 
            idx = pos[q] 
            print(tree.query(0,idx), end= ' ')
            tree.update(idx,0)
            pos[q] = top 
            tree.update(top,1)
            top -= 1


if __name__ == "__main__":
    main()