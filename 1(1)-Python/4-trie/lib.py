from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable


"""
TODO:
- Trie.push 구현하기
- (필요할 경우) Trie에 추가 method 구현하기
"""


T = TypeVar("T")


@dataclass
class TrieNode(Generic[T]):
    body: Optional[T] = None
    children: list[int] = field(default_factory=lambda: [])
    is_end: bool = False


class Trie(list[TrieNode[T]]):
    def __init__(self) -> None:
        super().__init__()
        self.append(TrieNode(body=None))

    def push(self, seq: Iterable[T]) -> None:
        """
        seq: T의 열 (list[int]일 수도 있고 str일 수도 있고 등등...)

        action: trie에 seq을 저장하기
        """
        # 구현하세요!
        start = 0 

        for element in seq: 
            found = False 

            for index in self[start].children:
                if self[index].body == element:
                    start = index 
                    found = True 
                    break 

            if not found: 
                new_node = TrieNode(body = element)
                self.append(new_node)
                self[start].children.append(len(self)-1)
                start = len(self)-1 

        self[start].is_end = True 

    # 구현하세요!