from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable


"""
TODO:
- SegmentTree 구현하기
"""


T = TypeVar("T")        
U = TypeVar("U")        

@dataclass
class SegmentTree(Generic[T, U]):
    def __init__(
        self,
        data: list[T],
        merge: Callable[[U, U], U],
        default_value: U,
        transform: Callable[[T], U]
    ) -> None:
        """
        :param data: 배열 형태의 입력 데이터
        :param merge: 트리 생성을 위한 병합 함수 (구간합, 최대값, 최소값 등)
        :param default_value: 병합 함수 별 기본값
        """
        self.data = data
        self.merge = merge
        self.default_value = default_value
        self.transform = transform
        self.n = len(data)
        self.tree = [self.default_value] * (4 * self.n) 
        self.build(0, 0, self.n - 1) 

    def build(self, curr_node: int, start_idx: int, end_idx: int) -> None:
        """
        :param node: 현재 노드 인덱스
        :param start: 구간 시작 인덱스
        :param end: 구간 끝 인덱스
        """
        if start_idx == end_idx: 
            self.tree[curr_node] = self.transform(self.data[start_idx])
        else:
            mid = (start_idx + end_idx) // 2 
            left_child = 2 * curr_node + 1
            right_child = 2 * curr_node + 2
            self.build(left_child, start_idx, mid)
            self.build(right_child, mid + 1, end_idx)
            self.tree[curr_node] = self.merge(self.tree[left_child], self.tree[right_child])

    def query(self, query_start: int, query_end: int, curr_node: int, start_idx: int, end_idx: int) -> U:
        """
        특정 구간 [query_start, query_end]의 값을 반환
        :param query_start: 쿼리 시작 인덱스
        :param query_end: 쿼리 끝 인덱스
        :param curr_node: 현재 노드 인덱스
        :param start_idx: 구간 시작 인덱스
        :param end_idx: 구간 끝 인덱스
        :return: 구간 값
        """
        if query_end < start_idx or query_start > end_idx:
            return self.default_value
        if query_start <= start_idx and end_idx <= query_end:
            return self.tree[curr_node]
        mid = (start_idx + end_idx) // 2
        left_child = 2 * curr_node + 1
        right_child = 2 * curr_node + 2
        left_result = self.query(query_start, query_end, left_child, start_idx, mid)
        right_result = self.query(query_start, query_end, right_child, mid + 1, end_idx)
        return self.merge(left_result, right_result)

    def update(self, idx: int, value: T, curr_node: int, start_idx: int, end_idx: int) -> None:
        """
        배열의 특정 인덱스를 업데이트
        :param idx: 업데이트할 인덱스
        :param value: 새로운 값
        :param node: 현재 노드 인덱스
        :param start: 구간 시작 인덱스
        :param end: 구간 끝 인덱스
        """
        if start_idx == end_idx:
            self.tree[curr_node] = self.transform(value)
        else:
            mid = (start_idx + end_idx) // 2
            left_child = 2 * curr_node + 1
            right_child = 2 * curr_node + 2
            if idx <= mid:
                self.update(idx, value, left_child, start_idx, mid)
            else:
                self.update(idx, value, right_child, mid + 1, end_idx)
            self.tree[curr_node] = self.merge(self.tree[left_child], self.tree[right_child])




import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


class Pair(tuple[int, int]):
    """
    힌트: 2243, 3653에서 int에 대한 세그먼트 트리를 만들었다면 여기서는 Pair에 대한 세그먼트 트리를 만들 수 있을지도...?
    """
    def __new__(cls, a: int, b: int) -> 'Pair':
        return super().__new__(cls, (a, b))

    @staticmethod
    def default() -> 'Pair':
        """
        기본값
        이게 왜 필요할까...?
        """
        return Pair(0, 0)

    @staticmethod
    def f_conv(w: int) -> 'Pair':
        """
        원본 수열의 값을 대응되는 Pair 값으로 변환하는 연산
        이게 왜 필요할까...?
        """
        return Pair(w, 0)

    @staticmethod
    def f_merge(a: Pair, b: Pair) -> 'Pair':
        """
        두 Pair를 하나의 Pair로 합치는 연산
        이게 왜 필요할까...?
        """
        return Pair(*sorted([*a, *b], reverse=True)[:2])

    def sum(self) -> int:
        return self[0] + self[1]


def main() -> None:
    # 구현하세요!
    pass


if __name__ == "__main__":
    main()