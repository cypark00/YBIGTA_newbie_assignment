from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable


"""
TODO:
- SegmentTree 구현하기
"""


T = TypeVar("T")
U = TypeVar("U")


class SegmentTree(Generic[T, U]):

    # 구현하세요!
    def __init__(self, size: int, default:T, func: Callable[[T,T],T]) -> None:
        """
        세그먼트 트리 초기화 

        Parameters:
        size : 관리할 데이터 개수 
        default : 트리 초기화 시 사용할 값 
        func : 값을 받아 누적하는 함수 
        """
        self.N =1 
        while self.N < size: 
            self.N *= 2
        self.tree = [default] * (2*self.N)
        self.default = default 
        self.func = func 

    def update(self, index:int, value:T) -> None: 
        """
        세그먼트 트리에서 특정 인덱스 값 갱신, 관련된 부모 노드의 값을 재계산. 
        index : 업데이트할 데이터의 인덱스 
        value : 새로 바꿀 값 
        """
        i = index + self.N 
        self.tree[i] = value  
        while i>1: 
            i//=2
            self.tree[i] = self.func(self.tree[2*i], self.tree[2*i +1])
        
    def query(self, left:int, right:int) -> T: 
        """
        구간에 대한 누적값을 계산 

        Parameters: 
        left : 구간 시작 인덱스 
        right : 구간 끝 인덱스 
        """
        left += self.N 
        right += self.N 
        sum = self.default 

        while left<right: 
            if left % 2 ==1 : 
                sum = self.func(sum, self.tree[left])
                left += 1
            if right % 2 ==1: 
                right -= 1
                sum = self.func(sum, self.tree[right])
            left //=2
            right //=2
        return sum
    
    
    
            


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