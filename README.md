알고리즘 문제풀이 정리
======================
## 1. 공부중인 사이트 목록
1. 코딩테스트 연습 | 프로그래머스([https://programmers.co.kr/learn/courses/30](https://programmers.co.kr/learn/courses/30))

## 2. 파이썬 참고자료 목록
### DAY 1 (20.09.01)
1. [파이썬] heapq 모듈 사용법([https://www.daleseo.com/python-heapq/](https://www.daleseo.com/python-heapq/))
2. 순열과 조합 - combinations, permutations([https://programmers.co.kr/learn/courses/4008/lessons/12836](https://programmers.co.kr/learn/courses/4008/lessons/12836))
3. Python에서 리스트 중복 제거하기([https://velog.io/@daybreak/Python%EC%97%90%EC%84%9C-%EB%A6%AC%EC%8A%A4%ED%8A%B8-%EC%A4%91%EB%B3%B5%EC%A0%9C%EA%B1%B0](https://velog.io/@daybreak/Python%EC%97%90%EC%84%9C-%EB%A6%AC%EC%8A%A4%ED%8A%B8-%EC%A4%91%EB%B3%B5%EC%A0%9C%EA%B1%B0))
4. python으로 피보나치 수열 갖고 놀기([https://www.crocus.co.kr/1643](https://www.crocus.co.kr/1643))
5. [Python] 파이썬 특정문자 찾기(find,startswith,endswith)([https://dpdpwl.tistory.com/119](https://dpdpwl.tistory.com/119))

### DAY 2 (20.09.02)
1. [python] 파이썬 절대값(abs)함수 ([https://blockdmask.tistory.com/380](https://blockdmask.tistory.com/380))
2. 파이썬(python) 리스트 중복 요소 개수 찾기([https://infinitt.tistory.com/78](https://infinitt.tistory.com/78))
3. [python] 파이썬 사전 딕셔너리 값 value 로 정렬하는 방법 - lambda 식 응용 - 파이썬으로 단어 수 세기, 텍스트에서 가장 많이 출현한 단어 세기([https://korbillgates.tistory.com/171](https://korbillgates.tistory.com/171))
4. 문자열([https://snakify.org/ko/lessons/strings_str/](https://snakify.org/ko/lessons/strings_str/))

### DAY 3 (20.09.03)
1. [python] 문자열 거꾸로 출력하기 ([https://itholic.github.io/python-reverse-string/](https://itholic.github.io/python-reverse-string/))
2. 파이썬을 사용한 이진 트리와 순회 알고리즘 구현([http://ejklike.github.io/2018/01/09/traversing-a-binary-tree-1.html](http://ejklike.github.io/2018/01/09/traversing-a-binary-tree-1.html))
3. python sorted 에 대해서([http://blog.weirdx.io/post/50236](http://blog.weirdx.io/post/50236))
4. [파이썬(Python)] 회전행렬 / 2차원배열 회전하는 법 구현하기([https://deepwelloper.tistory.com/117](https://deepwelloper.tistory.com/117))
5. [Python] 2차원 리스트 초기화([http://blog.naver.com/PostView.nhn?blogId=ambidext&logNo=221417120233&parentCategoryNo=10&categoryNo=&viewDate=&isShowPopularPosts=true&from=search](http://blog.naver.com/PostView.nhn?blogId=ambidext&logNo=221417120233&parentCategoryNo=10&categoryNo=&viewDate=&isShowPopularPosts=true&from=search))
6. collections 모듈 - deque([https://excelsior-cjh.tistory.com/96](https://excelsior-cjh.tistory.com/96))
7. (Python) 파이썬 약수구하기, Python 약수([https://sun-kakao.tistory.com/96](https://sun-kakao.tistory.com/96))

## 3. 파이썬으로 푼 문제 연습 목록
### DAY 1 (20.09.01)
1. 크레인 인형뽑기 게임([https://programmers.co.kr/learn/courses/30/lessons/64061](https://programmers.co.kr/learn/courses/30/lessons/64061))
```
def solution(board, moves):
    result = []
    cnt = 0
    for i in moves:
        for j in range(0,len(board)):
            if board[j][i-1] != 0:
                if len(result) <= 0:
                    result.append(board[j][i-1])
                else:
                    top = result[-1]
                    if top == board[j][i-1]:
                        result.pop()
                        cnt+=1
                    else:
                        result.append(board[j][i-1])
                board[j][i-1]=0
                break
            else:
                continue
    return cnt*2
```
2. 튜플([https://programmers.co.kr/learn/courses/30/lessons/64065](https://programmers.co.kr/learn/courses/30/lessons/64065))
```
def solution(s):
    ansList = []
    ans = []
    for i in range(1,len(s)):
        numList = []
        numStr = ''
        if s[i] == '{':
            i+=1
            while(s[i] != '}'):
                if s[i] == ',':
                    numList.append(numStr)
                    numStr = ''
                else:
                    numStr += s[i]
                i+=1
            numList.append(numStr)
            numsStr = ''
            ansList.append(numList)
        else:
            pass
    for i in range(1,len(ansList)+1):
        for j in ansList:
            if len(j)==i:
                for k in range(0,i):
                    if j[k] not in ans:
                        ans.append(j[k])
    ans = [int (i) for i in ans]
    return ans
```

3. 스킬트리([https://programmers.co.kr/learn/courses/30/lessons/49993](https://programmers.co.kr/learn/courses/30/lessons/49993))
```
def solution(skill, skill_trees):
    ans = 0
    for skill_tree in skill_trees:
        #print(skill_tree)
        checkList = skill
        flag = True
        for j in skill_tree:
            if j in checkList:
                if checkList[0] == j:
                    checkList = checkList.replace(j,'')
                else:
                    flag = False
                    break
        if flag == True:
            ans+=1
    return ans
```

4. 2 x n 타일링([https://programmers.co.kr/learn/courses/30/lessons/12900](https://programmers.co.kr/learn/courses/30/lessons/12900))
```
def solution(n):
    fibo = [] 
    for x in range(0,n+1): 
        if x < 2: 
            fibo.append(1) 
        else: 
            fibo.append(fibo[x-2]%1000000007 + fibo[x-1]%1000000007)
    answer = fibo[len(fibo)-1]%1000000007
    return answer
```

5. 전화번호 목록([https://programmers.co.kr/learn/courses/30/lessons/42577](https://programmers.co.kr/learn/courses/30/lessons/42577))
```
def solution(phone_book):
    for i in range(0,len(phone_book)):
        for j in range(0,len(phone_book)):
            if phone_book[j].startswith(phone_book[i]) == True and i!=j:
                return False
    return True
```

6. 더 맵게([https://programmers.co.kr/learn/courses/30/lessons/42626](https://programmers.co.kr/learn/courses/30/lessons/42626))
```
import heapq

def solution(scoville, K):
    heap = []
    for num in scoville:
        heapq.heappush(heap, num)
    cnt=0
    while heap[0]<K:
        try:
            heapq.heappush(heap, heapq.heappop(heap) + (heapq.heappop(heap) * 2))
        except:
             return -1
        cnt+=1
    return cnt
```

7. 소수 찾기([https://programmers.co.kr/learn/courses/30/lessons/42839](https://programmers.co.kr/learn/courses/30/lessons/42839))
```
import itertools

def solution(numbers):
    ans=0
    permut_list = []
    for i in range(1,len(numbers)+1):
        permut_list.append(list(map(''.join, itertools.permutations(numbers, i))))
    ans_list = []
    for i in permut_list:
        for j in i:
            ans_list.append(int(j))
    my_set = set(ans_list)
    new_list = list(my_set)
    def isPrime(a):
        if(a<2):
            return False
        for i in range(2,a):
            if(a%i==0):
                return False
        return True
    for i in new_list:
        if(isPrime(i)):
            ans+=1
    return ans
```

8. 체육복([https://programmers.co.kr/learn/courses/30/lessons/42862](https://programmers.co.kr/learn/courses/30/lessons/42862))
```
def solution(n, lost, reserve):
    temp = n - len(lost)
    for i in range(0,len(reserve)):
        for j in range(0,len(lost)):
            if lost[j]==-1 and reserve[i]==-1:
                continue
            if lost[j] == reserve[i]:
                lost[j] = -1
                reserve[i] = -1
                temp+=1      
    for i in range(0,len(lost)):
        for j in range(0,len(reserve)):
            if lost[i]==-1 or reserve[j]==-1:
                continue
            if abs(lost[i]-reserve[j])==1:
                temp+=1
                reserve[j] = -1
                break
    return temp
```

9. 구명보트([https://programmers.co.kr/learn/courses/30/lessons/42885](https://programmers.co.kr/learn/courses/30/lessons/42885))
```
def solution(people, limit):
    people_list = sorted(people)
    i = 0
    j = len(people_list)-1
    ans = 0
    while i <= j:
        if people_list[i] + people_list[j] <= limit:
            i+=1
        j-=1
        ans += 1
    return ans
```

### DAY 2 (20.09.02)
1. 오픈채팅방([https://programmers.co.kr/learn/courses/30/lessons/42888](https://programmers.co.kr/learn/courses/30/lessons/42888))
```
def solution(record):
    name_dict = {}
    record_list = []
    for i in record:
        record_list.append([(i.split(' '))])
    for i in record_list:
        for j in i:
            if j[0] == 'Enter' or j[0] == 'Change':
                name_dict[j[1]] = j[2]
    ans_list = []
    for i in record_list:
        for j in i:
            if j[0] == 'Enter':
                result = str(name_dict[j[1]]) +'님이 들어왔습니다.'
                ans_list.append(result)
            elif j[0] == 'Leave':
                result = str(name_dict[j[1]]) + '님이 나갔습니다.'
                ans_list.append(result)
    return(ans_list)
```

2. 위장([https://programmers.co.kr/learn/courses/30/lessons/42578](https://programmers.co.kr/learn/courses/30/lessons/42578))
```
def solution(clothes):
    count = {}
    for i in clothes:
        try: count[i[1]] += 1
        except: count[i[1]]=1
    sum = 1
    for i in count:
        sum *= (count[i]+1)
    sum -= 1
    return sum
```

3. 실패율([https://programmers.co.kr/learn/courses/30/lessons/42889?language=python3](https://programmers.co.kr/learn/courses/30/lessons/42889?language=python3))
```
def solution(N, stages):
    ans_list = []
    ans_dict = {}
    for i in range(1,N+1):
        f=0
        t=0
        for j in stages:
            if j >= i:
                t +=1
            if j == i:
                f +=1
        if f == 0:
            ans_dict[i] = 0
        elif t == 0:
            ans_dict[i] = 0
        else:
            ans_dict[i] = float(f/t)
    ans_list = (sorted(ans_dict.items(), key=lambda x: x[1], reverse=True))
    res_list = []
    for i in ans_list:
        res_list.append(i[0])
    return res_list
```

4. 괄호 변환([https://programmers.co.kr/learn/courses/30/lessons/60058?language=python3](https://programmers.co.kr/learn/courses/30/lessons/60058?language=python3))
```
def solution(p):
    def hi(p):
        # 입력이 빈 문자열
        if p == '':
            # 빈 문자열 반환
            return p
        else:
            left = 0
            right = 0
            u = ''
            v = ''
            #u, v로 반환
            for i in p:
                if left == right and left != 0:
                    v += i
                else:
                    if i =='(':
                        left += 1
                        u += i
                    else:
                        right += 1
                        u += i
            # u가 올바른 괄호 문자열 체크
            stack = []
            flag = True
            for i in u:
                if i == '(':
                    stack.append(i)
                else:
                    if len(stack) == 0:
                        flag = False
                        break
                    else:
                        stack.pop()
            if len(stack)>0:
                flag = False
            # u가 올바른 괄호 문자열일 경우
            if flag == True:
                return u + hi(v)
            elif flag == False:
                new_str = '('
                new_str += hi(v)
                new_str += ')'
                u = u[1:len(u)-1]
                new_u = ''
                for i in u:
                    if i =='(':
                        new_u += ')'
                    else:
                        new_u += '('
                new_str += new_u
                return new_str
    return(hi(p))
```

### DAY 3 (20.09.03)
1. 124 나라의 숫자([https://programmers.co.kr/learn/courses/30/lessons/12899#](https://programmers.co.kr/learn/courses/30/lessons/12899#))
```
def solution(n):
    s_list = []
    while(n>0):
        nam = n%3
        n = int(n/3)
        if nam == 0:
            nam = 4
            n -= 1
        s_list.append(str(nam))
    s_list.reverse()  
    return(''.join(s_list))  # 거꾸로 뒤집어진 리스트를 연결해서 출력
```
2. 문자열 압축([https://programmers.co.kr/learn/courses/30/lessons/60057#](https://programmers.co.kr/learn/courses/30/lessons/60057#))
```
def solution(s):
    if len(s) == 1:
        return 1
    else:
        ans_list = []
        for j in range(1,int(len(s)/2)+1):
            s_list = []
            for i in range(0,int(len(s)/j)+1):
                s_list.append(s[i*j:i*j+j])
            tmp = ''
            s2_list = []
            for i in s_list:
                if tmp == '':
                    cnt = 1
                    tmp = i
                else:
                    if tmp == i:
                        cnt +=1
                    else:
                        if cnt == 1:
                            str_cnt = tmp
                        else:
                            str_cnt = str(cnt) + tmp
                        s2_list.append(str_cnt)
                        tmp = i
                        cnt = 1
            if tmp != '':
                if cnt == 1:
                    str_cnt = tmp
                else:
                    str_cnt = str(cnt) + tmp
                s2_list.append(str_cnt)
            ans = ''
            for i in s2_list:
                ans += i
            ans_list.append(len(ans))
        return min(ans_list)
```

3. H-Index([https://programmers.co.kr/learn/courses/30/lessons/42747](https://programmers.co.kr/learn/courses/30/lessons/42747))
```
def solution(citations):
    ans_list = []
    for h in range(0,10001):
        y = 0
        n = 0
        for j in citations:
            if j >= h:
                y+=1
        nam = len(citations)-y
        if y>=h and nam<=h:
            ans_list.append(h)
    return max((ans_list))
```

4. 프린터([https://programmers.co.kr/learn/courses/30/lessons/42587](https://programmers.co.kr/learn/courses/30/lessons/42587))
```
from collections import deque

def solution(priorities, location):
    deq_list = []
    for i in range(0,len(priorities)):
        deq_list.append([priorities[i],i])
    cnt = 1
    while(True):
        idx = 0
        max_elem = max(deq_list)
        max_num = max_elem[0]
        for i in deq_list:
            idx+=1
            if i[0] < max_num:
                deq_list.append(i)
            else:
                #loc 체크
                if i[1] == location:
                    return cnt
                else:
                    break
        deq_list = deque(deq_list)
        for i in range(0,idx):
            deq_list.popleft()
        deq_list = list(deq_list)
        cnt+=1
```

5. 카펫([https://programmers.co.kr/learn/courses/30/lessons/42842#](https://programmers.co.kr/learn/courses/30/lessons/42842#))
```
def solution(brown, yellow):
    num = brown+yellow
    ans_list = []
    for i in range(1, num+1):
        if num % i == 0:
            ans_list.append(i)
    if len(ans_list)%2 == 0:
        for i in range(0,int(len(ans_list)/2)):
            row = ans_list[i] - 2
            col = ans_list[len(ans_list)-i-1] - 2
            if row * col == yellow:
                if row > col:
                    return [row+2,col+2]
                else:
                    return [col+2,row+2]
    else:
        for i in range(0,int(len(ans_list)/2)):
            row = ans_list[i] - 2
            col = ans_list[len(ans_list)-i-1] - 2
            if row * col == yellow:
                if row > col:
                    return [row+2,col+2]
                else:
                    return [col+2,row+2]
        row = ans_list[int(len(ans_list)/2)] -2 
        if row * row == yellow:
            return [row+2,row+2]
```