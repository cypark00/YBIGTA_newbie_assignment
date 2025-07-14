from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main() -> None:
    MAX = 1000000

    tree : SegmentTree[int, int] = SegmentTree(size= MAX, default =0, func= lambda a, b: a+b)

    Q = int(input())

    for _ in range(Q):
        query = list(map(int, input().split()))

        if query[0] == 1: 
            k = query[1] 
            taste = tree.find_kth(k)
            print(taste)
            tree.update(taste-1, -1)

        elif query[0] ==2: 
            taste, count = query[1], query[2] 
            tree.update(taste-1, count)


if __name__ == "__main__":
    main()