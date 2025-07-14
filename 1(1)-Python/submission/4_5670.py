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


import sys


"""
TODO:
- 일단 Trie부터 구현하기
- count 구현하기
- main 구현하기
"""


def count(trie: Trie, query_seq: str) -> int:
    """
    trie - 이름 그대로 trie
    query_seq - 단어 ("hello", "goodbye", "structures" 등)

    returns: query_seq의 단어를 입력하기 위해 버튼을 눌러야 하는 횟수
    """
    pointer = 0
    cnt = 0

    for element in query_seq:
        if len(trie[pointer].children) > 1 or trie[pointer].is_end:
            cnt += 1

        new_index = None # 구현하세요!
        for index in trie[pointer].children: 
            if trie[index].body == element: 
                new_index = index 
                break
        assert new_index is not None
        pointer = new_index

    return cnt + int(len(trie[0].children) == 1)


def main() -> None:
    """
    입력을 받아 trie를 구성 
    각 단어마다 버튼 입력 횟수를 평균 내 출력 
    """
    line = sys.stdin.read().splitlines()
    index = 0 
    while index < len(line):
        if not line[index].strip(): 
            index += 1
            continue 
        n=int(line[index])
        index += 1

        trie = Trie[str]() 
        words = []

        for _ in range(n): 
            word = line[index].strip()
            index += 1 
            words.append(word)
            trie.push(word)

        total = sum(count(trie,word)for word in words)
        average = total/n 
        print(f"{average:.2f}")
        


if __name__ == "__main__":
    main()