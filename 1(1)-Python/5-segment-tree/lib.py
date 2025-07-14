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
    
    
    
            