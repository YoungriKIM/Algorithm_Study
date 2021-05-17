# 소스 코드
# 재귀적 구현!!

# n, target = 10, 7
# array = [1,3,5,7,9,11,13,15,17,19]

# 입력 받기
n, target = map(int, input().split())
array = list(map(int, input().split()))

# 함수 정의
def binary_search(array, target, start, end):
    # 스타트가 엔드보다 크다면 none 반환
    if start > end:
        return None
    # mid 지정
    mid = (start + end) // 2
    # 미드가 타겟과 같다면 미드값 반환
    if array[mid] == target:
        return mid
    # 미드가 타겟보다 크면 타켓이 미드 전에 있다는 거니까
    elif array[mid] > target:
        return binary_search(array, target, start, mid-1)
    # 혹인 미드가 타겟보다 작으면 미드 후에 있다는 거니까
    else:
        return binary_search(array, target, mid+1, end)

# 함수 적용
result = binary_search(array, target, 0, n-1)
if result == None:
    print('원소가 존재하지 않습니다.')
else:
    print(result+1)