from __future__ import annotations
import copy
from collections import deque
from collections import defaultdict
from typing import DefaultDict, List


"""
TODO:
- __init__ 구현하기
- add_edge 구현하기
- dfs 구현하기 (재귀 또는 스택 방식 선택)
- bfs 구현하기
"""


class Graph:
    def __init__(self, n: int) -> None:
        """
        그래프 초기화
        n: 정점의 개수 (1번부터 n번까지)
        """
        self.n = n
        self.adj : List[List[int]] = [[] for _ in range(n+1)]
        """
        인접 리스트 초기화
        """
        # 구현하세요!

    
    def add_edge(self, u: int, v: int) -> None:
        """
        양방향 간선 추가
        """
        # 구현하세요!
        self.adj[u].append(v)
        self.adj[v].append(u)
        """
        Parameters: 
        u(int) : 연결할 첫 번째 정점 
        v(int) : 연결할 두 번째 정점 
        """
    
    
    def dfs(self, start: int) -> list[int]:
        """
        깊이 우선 탐색 (DFS)
        
        구현 방법 선택:
        1. 재귀 방식: 함수 내부에서 재귀 함수 정의하여 구현
        2. 스택 방식: 명시적 스택을 사용하여 반복문으로 구현
        """
        # 구현하세요!
        visited = [False] * (self.n +1) 
        result = [] 

        def _dfs(v: int): 
            visited[v] = True 
            result.append(v)
            for neighbor in sorted(self.adj[v]): 
                if not visited[neighbor]: _dfs(neighbor)
        """
        정점 v에서 DFS를 재귀 방식으로 수행
        """
 
        _dfs(start)
        return result

    
    
    
    def bfs(self, start: int) -> list[int]:
        """
        너비 우선 탐색 (BFS)
        큐를 사용하여 구현
        """
        # 구현하세요!
        visited = [False] * (self.n + 1) 
        result = [ ]
        queue = deque([start]) 
        visited[start] = True

        while queue: 
            v = queue.popleft()
            result.append(v) 
            for neighbor in sorted(self.adj[v]):
                if not visited[neighbor]:
                    visited[neighbor] = True 
                    queue.append(neighbor) 

        return result
 

  
    
    def search_and_print(self, start: int) -> None:
        """
        DFS와 BFS 결과를 출력
        """
        dfs_result = self.dfs(start)
        bfs_result = self.bfs(start)
        
        print(' '.join(map(str, dfs_result)))
        print(' '.join(map(str, bfs_result)))
