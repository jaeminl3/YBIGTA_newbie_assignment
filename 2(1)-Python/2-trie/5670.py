from lib import Trie
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