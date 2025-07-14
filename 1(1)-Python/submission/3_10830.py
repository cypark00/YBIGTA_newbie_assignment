from __future__ import annotations
import copy


"""
TODO:
- __setitem__ 구현하기
- __pow__ 구현하기 (__matmul__을 활용해봅시다)
- __repr__ 구현하기
"""


class Matrix:
    MOD = 1000

    def __init__(self, matrix: list[list[int]]) -> None:
        self.matrix = matrix

    @staticmethod
    def full(n: int, shape: tuple[int, int]) -> Matrix:
        return Matrix([[n] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def zeros(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(0, shape)

    @staticmethod
    def ones(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(1, shape)

    @staticmethod
    def eye(n: int) -> Matrix:
        matrix = Matrix.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1
        return matrix

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.matrix), len(self.matrix[0]))

    def clone(self) -> Matrix:
        return Matrix(copy.deepcopy(self.matrix))

    def __getitem__(self, key: tuple[int, int]) -> int:
        return self.matrix[key[0]][key[1]]

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        # 구현하세요!
        self.matrix[key[0]][key[1]] = value

    def __matmul__(self, matrix: Matrix) -> Matrix:
        x, m = self.shape
        m1, y = matrix.shape
        assert m == m1

        result = self.zeros((x, y))

        for i in range(x):
            for j in range(y):
                for k in range(m):
                    result[i, j] += self[i, k] * matrix[k, j]

        return result

    def __pow__(self, n: int) -> Matrix:
        """
        행렬을 n번 거듭제곱한 결과를 반환
        """
        # 구현하세요!
        assert self.shape[0] == self.shape[1]
        size = self.shape[0]

        def matmul(a:Matrix, b: Matrix) -> Matrix:
            """ 
            두 행렬 a,b를 곱한 결과를 반환 
            누적 곱셈마다 MOD 연산 적용 
            """
            result = Matrix.zeros((size,size))
            for i in range(size): 
                for j in range(size):
                    temp =0 
                    for k in range(size): 
                        temp += a[i,k] * b[k,j]
                        temp %= self.MOD
                    result[i,j] = temp 
            return result

        if n==1:
            result = self.clone() 
            for i in range(size): 
                for j in range(size):
                    result[i,j] %= self.MOD
            return result
        
        half = self ** (n//2)
        half_square = matmul(half,half)
        return half_square if n%2==0 else matmul(half_square, self)
    
    
    def __repr__(self) -> str:
        """
        행렬을 문자열 형태로 반환
        """
        line : list[str] = [] 
        for row in self.matrix: 
            row_str = " ".join(map(str, row))
            line.append(row_str)
        return "\n".join(line)


from typing import Callable
import sys


"""
-아무것도 수정하지 마세요!
"""


def main() -> None:
    intify: Callable[[str], list[int]] = lambda l: [*map(int, l.split())]

    lines: list[str] = sys.stdin.readlines()

    N, B = intify(lines[0])
    matrix: list[list[int]] = [*map(intify, lines[1:])]

    Matrix.MOD = 1000
    modmat = Matrix(matrix)

    print(modmat ** B)


if __name__ == "__main__":
    main()