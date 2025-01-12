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
        
        new_index = trie.find_child_index(pointer, element)

        pointer = new_index

    return cnt + int(len(trie[0].children) == 1)


def main() -> None:
    lines: list[str] = [line.strip() for line in sys.stdin.readlines()]
    datas = []
    avg_typings = []
    curr_index = 0
    next_index = 0


    while next_index < len(lines):
        next_index += int(lines[curr_index]) + 1
        unit_data = lines[curr_index:next_index]
        datas.append(unit_data)
        curr_index = next_index
    
    for data in datas:
        N = int(data[0])
        words = data[1:]
        trie = Trie[str]()
        for word in words:
            trie.push(word)     #한 데이터에 대한 트라이구조 완성

        typing_num = 0
        for word in words:
            typing_num += count(trie, word)
        avg_typings.append(round(typing_num / N, 2))
    
    for avg_typing in avg_typings:
        print(f"{avg_typing:.2f}")

if __name__ == "__main__":
    main()