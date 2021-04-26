# 프로그래머스 카카오 인턴쉽
# https://programmers.co.kr/learn/courses/30/lessons/67256?language=python3

# def solution(numbers, hand):
#     answer = ''
#     return answer

numbers = [7, 0, 8, 2, 8, 3, 1, 5, 7, 6, 2]
hand = 'right'


def solution(hand, numbers):
    answer = ''
    location = [[3,1],[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
    left, right = [3,0], [3,2]
    for i in numbers:
        if i % 3 == 1:
            answer += 'L'
            left = location[i]
        elif i % 3 == 0 and i != 0:
            answer += 'R'
            right = location[i]
        else:
            l = abs(location[i][0] - left[0]) + abs(location[i][1] - left[1])
            r = abs(location[i][0] - right[0]) + abs(location[i][1] - right[1])
            if l < r:
                answer += 'L'
                left = location[i]
            elif l > r:
                answer += 'R'
                right = location[i]
            else:
                answer += hand[0].upper()
                if hand == left :
                    left = location[i]
                else :
                    rifht = location[i]
    print(answer)

solution(hand, numbers)