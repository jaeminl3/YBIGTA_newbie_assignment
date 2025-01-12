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
    body: Optional[T] = None        #타입힌트가 Optional이지만 기본값이 None이다
    children: list[int] = field(default_factory=lambda: [])     #children의 기본값이, 빈 리스트임을 나타냄. 이 때 그냥 빈 리스트로 하면 모든 children이 함께 변동되므로 객체의 독립성을 보장하기 위한 방법
    is_end: bool = False


class Trie(list[TrieNode[T]]):      #리스트의 타입힌트가 TrieNode[T]의 요소라는 것
    def __init__(self) -> None:
        super().__init__()      #부모(리스트)의 객체 메소드 방식을 그대로 물려받는다는 의미, 즉 빈 리스트로 작동동
        self.append(TrieNode(body=None))        #거기에 추가로, body가 None인 노드를 추가로 생성하여 붙힘


    def push(self, seq: Iterable[T]) -> None:                                                  
        
        trie_index = 0

        for char in seq:        
            for child_index in self[trie_index].children:       #for - break - else 구문에서 for가 break에 의해 중단되지 않으면 else가 수행됨. child_body_list가 필요없다.
                if self[child_index].body == char:
                    trie_index = child_index
                    break

            else:
                new_node_index = len(self)
                self.append(TrieNode(body=char))
                self[trie_index].children.append(new_node_index)
                trie_index = new_node_index
        
        self[trie_index].is_end = True

    def find_child_index(self, pointer: int, char: str) -> int:
        for child_index in self[pointer].children:
            if self[child_index].body == char:
                return child_index
        return -999



import sys


"""
TODO:
- 일단 Trie부터 구현하기
- main 구현하기

힌트: 한 글자짜리 자료에도 그냥 str을 쓰기에는 메모리가 아깝다...
"""


def main() -> None:
    MOD = 1000000007
    def factorial_mod(n: int, MOD: int) -> int:
        result = 1
        for i in range(2, n + 1):
            result = (result * i) % MOD
        return result

    lines: list[str] = [line.strip() for line in sys.stdin.readlines()]
    
    N = int(lines[0])
    names = lines[1:]

    trie = Trie[str]()
    for name in names:
        trie.push(name)     #trie구조에 노드 생성 완료

    num_case = 1
    for node in trie:
        node_value = len(node.children) + (1 if node.is_end else 0)
        num_case = (num_case * factorial_mod(node_value,MOD)) % MOD

    print(num_case)

if __name__ == "__main__":
    main()