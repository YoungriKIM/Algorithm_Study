# Python에서 제공하는 자신만의 반복자를 만드는 모듈
# 반복되는 요소에 대한 처리에 강하다

import numpy as np

# 패키지 임포트
import itertools

# ------------------------------------------
# chain()
# 리스트, 튜플 등을 연결해준다.

letters = ['a', 'b', 'c', 'd', 'e', 'f']
booleans = [1, 0, 1, 0, 0, 1]
decimals = [0.1, 0.7, 0.4, 0.4, 0.5]

chain = list(itertools.chain(letters, booleans, decimals))
# 리스트로 해줘야 리스트 안에 넣어서 출력 됨

print(chain)
# ['a', 'b', 'c', 'd', 'e', 'f', 1, 0, 1, 0, 0, 1, 0.1, 0.7, 0.4, 0.4, 0.5]
print(type(chain))
# <class 'list'>

# ------------------------------------------
