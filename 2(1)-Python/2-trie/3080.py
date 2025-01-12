from lib import Trie
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