알고리즘 문제풀이 정리
======================
## 1. 공부중인 사이트 목록
### 알고리즘 실습
1. Baekjoon Online Judge(https://www.acmicpc.net/)
2. 프로그래머스(https://programmers.co.kr/learn/courses/30)
3. Codeforces(https://codeforces.com/)
4. AtCoder(https://atcoder.jp/home)

### 알고리즘 이론
1. 박트리의 블로그(https://baactree.tistory.com/notice/16)
2. Ries 마법의 슈퍼마리오(https://blog.naver.com/kks227/220769870195)
3. BaaaaaaaarkingDog(https://blog.encrypted.gg/)
4. 안경잡이개발자(https://blog.naver.com/ndb796/221226794899)

## 2. 파이썬 참고자료 목록
### DAY 1 (20.09.01)
1. heapq 모듈 사용법(https://www.daleseo.com/python-heapq/)
2. 순열과 조합 - combinations, permutations(https://abit.ly/wyxcdu)
3. 리스트 중복 제거하기(https://abit.ly/1ljjxk9)

### DAY 3 (20.09.03)
1. 문자열 거꾸로 출력하기(https://itholic.github.io/python-reverse-string/)
2. 이진 트리와 순회 알고리즘 구현(https://abit.ly/d7sg68)
3. 회전행렬 / 2차원배열 회전하는 법 구현하기(https://deepwelloper.tistory.com/117)

### DAY 4 (20.09.04)
1. 최대공약수, 최소공배수, N개의 최소공배수((https://brownbears.tistory.com/454)
2. 2진수, 8진수, 16진수 다루기(https://www.daleseo.com/python-int-bases/)
3. 행렬의 곱셈(https://brownbears.tistory.com/449)

### DAY 5 (20.09.05)
1. 소수 찾기 - 에라토스테네스의 체(https://abit.ly/xnmgwa)

### DAY 9 (20.09.09)
1. 2차원 list 중복 제거(https://inma.tistory.com/132)

### DAY 10 (20.09.10)
1. 달팽이삼각형(https://machine-geon.tistory.com/64)

### DAY 18 (20.09.18)
1. 정렬, 다중조건(https://dailyheumsi.tistory.com/67)

### DAY 22 (20.09.22)
1. 효율적인 약수의 개수를 찾는 알고리즘(https://hsdevelopment.tistory.com/110)

### DAY 44 (20.10.14)
1. 다익스트라 알고리즘(https://m.blog.naver.com/ndb796/221234424646)
2. 플로이드 와샬 알고리즘(https://m.blog.naver.com/ndb796/221234427842)

### DAY 50 (20.10.20)
1. 최장 공통 부분 수열(https://debuglog.tistory.com/77)
2. 위상 정렬(https://abit.ly/am2qwdt)
3. 파이썬의 Asterisk * 이해하기(https://abit.ly/bvhvib)

### DAY 51 (20.10.21)
1. 유니온-파인드(https://m.blog.naver.com/ndb796/221230967614)

### DAY 59 (20.10.29)
1. 문자열, 배열 입력 받기(https://johnyejin.tistory.com/62)

### DAY 67 (20.11.06)
1. 비트마스크는 무엇인가?(https://mygumi.tistory.com/361)
2. 우선순위 큐(https://lipcoder.tistory.com/100)

### DAY 129 (21.01.07)
1. zip(), 배열 회전(https://abit.ly/xwejxd7)

## 3. 파이썬으로 푼 문제 목록
### DAY 1 (20.09.01)
1. 튜플(https://programmers.co.kr/learn/courses/30/lessons/64065)
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

### DAY 2 (20.09.02)
1. 오픈채팅방(https://programmers.co.kr/learn/courses/30/lessons/42888)
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

2. 괄호 변환(https://programmers.co.kr/learn/courses/30/lessons/60058)
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
1. 문자열 압축(https://programmers.co.kr/learn/courses/30/lessons/60057)
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

2. 프린터(https://programmers.co.kr/learn/courses/30/lessons/42587)
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

### DAY 5 (20.09.05)
1. 소수 만들기(https://programmers.co.kr/learn/courses/30/lessons/12977)
```
from itertools import combinations

def solution(nums):    
    def prime_list(n):
        # 에라토스테네스의 체 초기화: n개 요소에 True 설정(소수로 간주)
        sieve = [True] * n
        # n의 최대 약수가 sqrt(n) 이하이므로 i=sqrt(n)까지 검사
        m = int(n ** 0.5)
        for i in range(2, m + 1):
            if sieve[i] == True:           # i가 소수인 경우 
                for j in range(i+i, n, i): # i이후 i의 배수들을 False 판정
                    sieve[j] = False
        # 소수 목록 산출
        return sieve
    
    prime = prime_list(3000)
    ans_list = list(combinations(nums, 3))
    ans = 0
    for i in ans_list:
        sum = 0
        for j in i:
            sum += j
        if sum in prime:
            ans+=1
    return ans
```

### DAY 7 (20.09.07)
1. 네트워크 (https://programmers.co.kr/learn/courses/30/lessons/43162)
```
def solution(n, computers):
    def dfs(computers,visited,v):
        visited[v]=1
        for i in range(len(computers)):
            if computers[v][i]==1 and visited[i]==0:
                dfs(computers,visited,i)
    visited = [0 for i in range(n)]
    ans = 0
    for i in range(n):
        if visited[i] == 0:
            dfs(computers,visited,i)
            ans+=1
    return ans
```

2. 단어 변환(https://programmers.co.kr/learn/courses/30/lessons/43163)
```
stack = []
visited = []

def solution(begin, target, words):
    def dfs(word,depth):
        if word == target:
            stack.append(depth)
        else:
            for i in range(0,len(words)):
                if visited[i] == 0:
                    diff = 0
                    for j in range(0,len(word)):
                        if word[j] != words[i][j]:
                            diff +=1
                    if diff == 1:
                        visited[i]=1
                        dfs(words[i],depth+1)
                        visited[i]=0
    diff = 0
    visited = [0 for i in range (len(words))]
    dfs(begin,0)
    if len(stack)==0:
        return 0
    else:
        return min(stack)
```

### DAY 9 (20.09.09)
1. 여행경로(https://programmers.co.kr/learn/courses/30/lessons/43164)
```
import copy

def solution(tickets):
    visited = [0 for i in tickets]
    st = []
    ans = []
    def dfs(start,tickets,visited,cnt):
        st.append(start)
        if cnt >= len(tickets):
            ans.append(copy.deepcopy(st))
        for i in range(0,len(tickets)):
            if tickets[i][0] == start and visited[i] == 0:
                visited[i] = 1
                dfs(tickets[i][1],tickets,visited,cnt+1)
                visited[i] = 0
        st.pop()
    dfs("ICN",tickets, visited,0)
    return sorted(ans)[0]
```

2. 불량 사용자(https://programmers.co.kr/learn/courses/30/lessons/64064)
```
import itertools

def solution(user_id, banned_id):
    # 모든 경우의 수에 대한 조합
    ans_list = (list(set(itertools.permutations(user_id,len(banned_id)))))
    ans = []
    for i in range(0,len(ans_list)): # e.g) (frodo, fradi)
        flag = True
        cnt = 0
        for j in range(0,len(ans_list[i])): # e.g.) 1. frodo -> 2. fradi
            if len(ans_list[i][j]) == len(banned_id[j]): # e.g.) frodo 길이 == fr*d* 길이
                for k in range(0,len(ans_list[i][j])): # string 하나씩 비교
                    if ans_list[i][j][k] == banned_id[j][k]:
                        continue
                    else:
                        if banned_id[j][k] == '*': 
                            continue
                        else:
                            flag = False
                            break
                if flag == False:
                    break
                else:
                    cnt +=1
                    # 일치한 개수 == banned_id 개수
                    if cnt == len(banned_id):
                        ans.append(list(ans_list[i]))
            else:
                break
    # 2차원 list 중복 제거
    return(len(set([tuple(set(item)) for item in ans])))
```

### DAY 10 (20.09.10)
1. 땅따먹기(https://programmers.co.kr/learn/courses/30/lessons/12913)
```
def solution(land):
    for i in range(len(land)-1):
        land[i+1][0] += max(land[i][1], land[i][2], land[i][3])
        land[i+1][1] += max(land[i][0], land[i][2], land[i][3])
        land[i+1][2] += max(land[i][0], land[i][1], land[i][3])
        land[i+1][3] += max(land[i][0], land[i][1], land[i][2])
    return(max(land[-1]))
```

2. 프로그래머스 월간 코드 챌린지 시즌1(https://abit.ly/kmwkhzb)

![1](https://user-images.githubusercontent.com/26870568/92739723-2516ba80-f3b8-11ea-9849-32d7225de37d.PNG)

### DAY 11 (20.09.11)
1. 키패드 누르기(https://programmers.co.kr/learn/courses/30/lessons/67256)
```
import math

def solution(numbers, hand):
    ans = ''
    loc = {1: (0,0), 2: (0,1), 3: (0,2),
           4: (1,0), 5: (1,1), 6: (1,2),
           7: (2,0), 8: (2,1), 9: (2,2),
           '*': (3,0), 0: (3,1), '#': (3,2)}
    left = '*'
    right = '#'
    for i in numbers:
        if i == 1 or i == 4 or i == 7:
            left = i
            ans += 'L'
        elif i == 3 or i == 6 or i == 9:
            right = i
            ans += 'R'
        else:
            # 두 점 사이의 거리 공식 X
            lx = abs(loc[left][0] - loc[i][0])
            ly = abs(loc[left][1] - loc[i][1])
            ll = lx+ly
            rx = abs(loc[right][0] - loc[i][0])
            ry = abs(loc[right][1] - loc[i][1])
            rl = rx+ry
            if ll < rl:
                left = i
                ans += 'L'
            elif ll > rl:
                right = i
                ans += 'R'
            else:
                if hand == "right":
                    right = i
                    ans += 'R'
                else:
                    left = i
                    ans += 'L'
    return(ans)
```

2. 덩치(https://www.acmicpc.net/problem/7568)
```
n = int(input())
p_list = []
for i in range(0,n):
    p_info = input().split()
    p_list.append(p_info)
p_rank = []
for i in range(0,n):
    rank = 1
    for j in range(0,n):
        if i!=j and (p_list[i][0] < p_list[j][0] and p_list[i][1] < p_list[j][1]):
            rank+=1
    print(rank)
```

### DAY 14 (20.09.14)
1. N진수 게임(https://programmers.co.kr/learn/courses/30/lessons/17687)
```
def solution(n, t, m, p):
    s = "0"
    def convert(number, base):
        T = "0123456789ABCDEF"
        q, r = divmod(number, base)
        if q == 0:
            return T[r]
        else:
            return convert(q, base) + T[r]
    for i in range(1,t*m):
        s += convert(i,n)
    q = p-1
    result = ""
    for i in range(0,t):
        result += s[q]
        q += m
    return(result)
```

### DAY 16 (20.09.16)
1. 파일명 정렬(https://programmers.co.kr/learn/courses/30/lessons/17686)
```
import re

def solution(files):
    answer = []
    p = re.compile("([a-z\-\s]+)([0-9]+)(.*)")
    s_list = []
    for temp in files:
        g = p.search(temp.lower())
        head = g.group(1)
        num = int(g.group(2))
        
        s_list.append((head,num,temp))
    
    s_list = sorted(s_list,key=lambda x:(x[0],x[1]))
    for i in s_list:
        answer.append(i[2])
    return(answer)
```

### DAY 18 (20.09.18)
1. 베스트앨범(https://programmers.co.kr/learn/courses/30/lessons/42579)
```
def solution(genres, plays):
    answer = []
    g_dict = {}
    g_t_play = {}
    
    for i in range(len(genres)):
        if genres[i] not in g_dict:
            g_dict[genres[i]] = [(plays[i],i)]
            g_t_play[genres[i]] = plays[i]
        else:
            g_dict[genres[i]].append((plays[i],i))
            g_t_play[genres[i]] = g_t_play[genres[i]] + plays[i]
    # value값으로 정렬
    sorted_t_play = sorted(g_t_play.items(),key=lambda x:x[1], reverse=True)
    
    for key in sorted_t_play:
        play_list = g_dict[key[0]]
        
        # 오름차순
        play_list = sorted(play_list,key=lambda x:(-x[0],x[1]))
        
        for i in range(len(play_list)):
            if i == 2:
                break
            answer.append(play_list[i][1])
    return(answer)
```

### DAY 22 (20.09.22)
1. 문자열 폭발(https://www.acmicpc.net/problem/9935)
```
n=input()
k=input()
ans = []
for i in range(len(n)):
    ans.append(n[i])
    if len(ans) >= len(k):
        flag = True
        for j in range(1, len(k)+1):
            if ans[-j] != k[-j]:
                flag = False
                break
        if flag:
            for j in range(len(k)):
                ans.pop()
if len(ans) == 0:
    print("FRULA")
else:
    print("".join(ans))
```

### DAY 23 (20.09.23)
1. 부분합(https://www.acmicpc.net/problem/1806)
```
import sys

n,s = map(int,sys.stdin.readline().split())
arr = list(map(int,sys.stdin.readline().split()))

result,Sum,start,end = 100001,0,0,0

while True:
    if Sum >= s:
        Sum -= arr[start]
        result = min(result,(end-start))
        start+=1
    elif end == n:
        break
    else:
        Sum += arr[end]
        end+=1

if result == 100001:
    print(0)
else:
    print(result)
```

### DAY 26 (20.09.26)
1. 입국심사(https://programmers.co.kr/learn/courses/30/lessons/43238)
```
def solution(n, times):
    left = 0
    right = max(times)*n
    ans = right
    
    while left <= right:
        temp = 0
        mid = (left+right)//2
        for i in times:
            temp += mid//i
        if temp < n:
            left = mid + 1
        else:
            if mid <= ans:
                ans = mid
            right = mid - 1
    return ans
```

### DAY 35 (20.10.05)
1. 단지번호붙이기(https://www.acmicpc.net/problem/2667)
```
from collections import deque

dx = [0,0,1,-1]
dy = [1,-1,0,0]

n = int(input())
board = [list(map(int, list(input()))) for _ in range(n)]
dq = deque()
visited = [[False]*n for _ in range(n)]

ans = 0
st = []
for i in range(n):
    for j in range(n):
        if visited[i][j] == False and board[i][j] == 1:
            visited[i][j] = True
            dq.append((i,j))
            cnt = 1
            ans+=1
            while dq:
                x,y = dq.popleft()
                for k in range(4):
                    nx, ny = x+dx[k], y+dy[k]
                    if 0 <= nx < n and 0 <= ny < n:
                        if visited[nx][ny] == False and board[nx][ny] == 1:
                            dq.append((nx,ny))
                            cnt+=1
                            visited[nx][ny] = True
            st.append(cnt)
print(ans)
st = sorted(st)
for i in st:
    print(i)
```

2. DFS와 BFS(https://www.acmicpc.net/problem/1260)
```
from collections import deque

n,m,v = map(int, input().split())
board = [[0]*(n+1) for i in range(n+1)]
visited = [0 for i in range(n+1)]

for i in range(m):
    x,y = map(int, input().split())
    board[x][y] = 1
    board[y][x] = 1

def dfs(v):
    print(v,end=' ')
    visited[v] = 1
    for i in range(1,n+1):
        if visited[i] == 0 and board[v][i] == 1:
            dfs(i)
dfs(v)
print()

def bfs(v):
    dq = deque()
    dq.append(v)
    visited[v] = 0
    while dq:
        v = dq.popleft()
        print(v,end= ' ')
        for i in range(1,n+1):
            if visited[i] == 1 and board[v][i] == 1:
                dq.append(i)
                visited[i] = 0
bfs(v)
```

### DAY 38 (20.10.08)
1. 프로그래머스 월간 코드 챌린지 시즌1(https://abit.ly/ibnzqvf)

![1](https://user-images.githubusercontent.com/26870568/95467986-0ad0fc00-09b9-11eb-888e-cd01dc679f80.PNG)

### DAY 39 (20.10.09)
1. 트리 순회(https://www.acmicpc.net/problem/1991)
```
class Node:
    def __init__(self,item,left,right):
        self.item = item
        self.left = left
        self.right = right

def preorder(node):
    print(node.item, end='')
    if node.left != '.':
        preorder(tree[node.left])
    if node.right != '.':
        preorder(tree[node.right])

def inorder(node):
    if node.left != '.':
        inorder(tree[node.left])
    print(node.item, end='')
    if node.right != '.':
        inorder(tree[node.right])
        
def postorder(node):
    if node.left != '.':
        postorder(tree[node.left])
    if node.right != '.':
        postorder(tree[node.right])
    print(node.item, end='')
        
N = int(input())
tree = {}
    
for _ in range(N):
    node,left,right = map(str,input().split())
    tree[node] = Node(item=node,left=left,right=right)
    
preorder(tree['A'])
print()
inorder(tree['A'])
print()
postorder(tree['A'])
```

2. 트리의 부모 찾기(https://www.acmicpc.net/problem/11725)
```
import sys
sys.setrecursionlimit(10 ** 9)

n = int(sys.stdin.readline())
tree = [[] for _ in range(n+1)]

for _ in range(n-1):
    i,j = map(int,sys.stdin.readline().split())
    tree[i].append(j)
    tree[j].append(i)

parents = [0 for _ in range(n+1)]

def dfs(start,tree,parents):
    for i in tree[start]:
        if parents[i] == 0:
            parents[i] = start
            dfs(i,tree,parents)

dfs(1,tree,parents)

for i in range(2,n+1):
    print(parents[i])
```

3. 가장 먼 노드(https://programmers.co.kr/learn/courses/30/lessons/49189)
```
import sys
sys.setrecursionlimit(10 ** 9)
from collections import deque

def solution(n, edge):
    graph = [[] for _ in range(n+1)]
    for i in range(len(edge)):
        s,e = edge[i][0],edge[i][1]
        graph[s].append(e)
        graph[e].append(s)
        
    visited = [0 for _ in range(n+1)]
    dq = deque([[1,0]])
    
    while dq:
        cur_node, dist = dq.popleft()
        
        for next_node in graph[cur_node]:
            if visited[next_node] == 0 and next_node!=1:
                dq.append([next_node,dist+1])
                visited[next_node] = dist+1
                
    return(visited.count(max(visited)))
```

### DAY 43 (20.10.13)
1. 촌수계산(https://www.acmicpc.net/problem/2644)
```
import sys

n = int(sys.stdin.readline())
board = [[0]*(n+1) for i in range(n+1)]
visited = [0 for i in range(n+1)]

ex,ey = map(int,sys.stdin.readline().split())
m = int(sys.stdin.readline())

for i in range(m):
    x,y = map(int,sys.stdin.readline().split())
    board[y][x] = 1
    board[x][y] = 1

def dfs(v,cnt):
    visited[v] = 1
    if v == ey:
        print(cnt)
        return
    for i in range(1,n+1):
        if visited[i] == 0 and board[v][i] == 1:
            dfs(i,cnt+1)
dfs(ex,0)

if visited[ey] != 1:
    print(-1)
```

### DAY 44 (20.10.14)
1. RGB거리(https://www.acmicpc.net/problem/1149)
```
import sys

n = int(input())
cost = []
dp = [[0 for _ in range(3)] for _ in range(n)]

for i in range(n):
    cost.append(list(map(int,sys.stdin.readline().split())))
for i in range(n):
    if i == 0:
        dp[i][0] = cost[i][0]
        dp[i][1] = cost[i][1]
        dp[i][2] = cost[i][2]
    else:
        dp[i][0] = cost[i][0] + min(dp[i-1][1],dp[i-1][2])
        dp[i][1] = cost[i][1] + min(dp[i-1][0],dp[i-1][2])
        dp[i][2] = cost[i][2] + min(dp[i-1][0],dp[i-1][1])
print(min(dp[n-1][0],dp[n-1][1],dp[n-1][2]))
```

2. 섬의 개수(https://www.acmicpc.net/problem/4963)
```
import sys
sys.setrecursionlimit(10**5)

dx = [-1,-1,-1,0,0,1,1,1]
dy = [-1,0,1,-1,1,-1,0,1]
w = h = 0

def dfs(x,y):
    visited[x][y] = 1

    for i in range(8):
        nx = x + dx[i]
        ny = y + dy[i]

        if 0<=nx<h and 0<=ny<w and board[nx][ny]==1 and visited[nx][ny]==0:
            dfs(nx,ny)

while True:
    w,h = map(int,sys.stdin.readline().split())
    if w == 0 and h == 0:
        break
    else:
        board = [[0 for col in range(w)] for row in range(h)]

        for i in range(h):
            arr = list(map(int,sys.stdin.readline().split()))
            for j in range(w):
                board[i][j] = arr[j]

        visited = [[0]*w for _ in range(h)]

        cnt = 0

        for i in range(h):
            for j in range(w):
                if visited[i][j]==0 and board[i][j]==1:
                    dfs(i,j)
                    cnt+=1
        print(cnt)
```

3. 경로 찾기(https://www.acmicpc.net/problem/11403)
```
import sys

n = int(input())
graph = []

for i in range(n):
    graph.append(list(map(int,sys.stdin.readline().split())))

for k in range(n):
    for i in range(n):
        for j in range(n):
            if graph[i][j] == 1 or (graph[i][k] == 1 and graph[k][j] == 1):
                graph[i][j] = 1

for row in graph:
    for col in row:
        print(col, end=' ')
    print()
```

4. 플로이드(https://www.acmicpc.net/problem/11404)
```
import sys
INF=2**32

n = int(input())
m = int(input())

cost = [[INF for _ in range(100)] for _ in range(100)]

for i in range(n):
    cost[i][i] = 0

for _ in range(m):
    a,b,c = map(int,sys.stdin.readline().split())
    cost[a-1][b-1] = min(cost[a-1][b-1],c)

def floyd():
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i==j or j==k or k==i:
                    continue
                elif cost[i][j] > cost[i][k] + cost[k][j]:
                    cost[i][j] = cost[i][k] + cost[k][j]

floyd()

for i in range(n):
    for j in range(n):
        if cost[i][j] == INF:
            cost[i][j] = 0

for i in range(n):
    for j in range(n):
        if i == j:
            print(0,end=' ')
        else:
            print(cost[i][j],end=' ')
    print()
```

5. 최단경로(https://www.acmicpc.net/problem/1753)
```
import sys
from heapq import heappush, heappop

input = sys.stdin.readline
INF=sys.maxsize

v,e = map(int,input().split())
k = int(input())
graph = [[] for _ in range(v+1)]
dp = [INF] * (v+1)
heap = []

for i in range(e):
    u,v,w = map(int,input().split())
    # 방향 그래프
    graph[u].append([v,w])

def dijkstra(start):
    dp[start] = 0
    heappush(heap,[0,start])
    while heap:
        wei,now = heappop(heap)
        for next_node, w in graph[now]:
            next_wei = w + wei
            if dp[now] < wei:
                continue
            if next_wei < dp[next_node]:
                dp[next_node] = next_wei
                heappush(heap,[next_wei,next_node])
dijkstra(k)
for i in dp[1:]:
    if i != INF:
        print(i)
    else:
        print('INF')
```

6. 숫자 카드(https://www.acmicpc.net/problem/10815)
```
import sys

n = int(input())
arr = list(map(int,sys.stdin.readline().split()))
arr.sort()

m = int(input())
arr2 = list(map(int,sys.stdin.readline().split()))

def binary_search(val):
    first, last = 0, n-1
    while first<=last:
        mid = (first+last)//2
        if arr[mid] == val: return 1
        if arr[mid] > val: last = mid-1
        else: first = mid+1
    return 0

for i in arr2:
    print(binary_search(i), end=' ')
```

7. 수들의 합 5(https://www.acmicpc.net/problem/2018)
```
n = int(input())

left = 1
right = 1
Sum = 0
ans = 0

while left<=right and right<=n+1:
    if Sum < n:
        Sum += right
        right+=1
    elif Sum > n:
        Sum -= left
        left+=1
    else:
        ans+=1
        Sum += right
        right+=1
print(ans)
```

### DAY 45 (20.10.15)
1. 알약(https://www.acmicpc.net/problem/4811)
```
dp =[[0 for _ in range(32)] for _ in range(32)]
for i in range(0,31):
    dp[0][i] = 1
for w in range(1,31):
    for h in range(0,31):
        if h == 0:
            dp[w][h] = dp[w-1][1]
        else:
            dp[w][h] = dp[w-1][h+1] + dp[w][h-1]
while True:
    n = int(input())
    if n == 0:
        break
    else:
        print(dp[n-1][1])
```

### DAY 46 (20.10.16)
1. 유기농 배추(https://www.acmicpc.net/problem/1012)
```
from collections import deque

dx = [-1,0,0,1]
dy = [0,1,-1,0]

for _ in range(int(input())):
    m,n,k = map(int,input().split())
    board = [[0]*(m) for _ in range(n)]
    for _ in range(k):
        x,y = map(int,input().split())
        board[y][x] = 1
    visited = [[0]*(m) for _ in range(n)]

    def bfs(x,y):
        visited[x][y] = 1
        dq = deque()
        dq.append((x,y))
        while dq:
            x,y = dq.popleft()
            for i in range(4):
                nx = x + dx[i]
                ny = y + dy[i]
                if nx < 0 or nx >= n or ny < 0 or ny >= m:
                    continue
                else:
                    if visited[nx][ny] == 0 and board[nx][ny] == 1:
                        visited[nx][ny] = 1
                        dq.append((nx,ny))
    cnt = 0
    for i in range(n):
        for j in range(m):
            if visited[i][j] == 0 and board[i][j] == 1:
                bfs(i,j)
                cnt+=1
    print(cnt)
```

2. 친구(https://www.acmicpc.net/problem/1058)
```
INF = 2**32

n = int(input())
board = [[INF]*(50) for _ in range(50)]

for i in range(n):
    arr = list(input())
    for j in range(n):
        if i == j:
            board[i][j] = 0
        else: 
            if arr[j] == 'Y':
                board[i][j] = 1

def floyd():
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i==j or j==k or k==i:
                    continue
                elif board[i][j] > board[i][k] + board[k][j]:
                    board[i][j] = board[i][k] + board[k][j]
floyd()

ans = 0
for i in range(n):
    cnt = 0
    for j in range(n):
        if i == j:
            continue
        elif board[i][j] <=2:
            cnt+=1
    ans = max(ans,cnt)
print(ans)
```

3. 쿼드트리(https://www.acmicpc.net/problem/1992)
```
n = int(input())
board = [input() for _ in range(n)]

def quad(x,y,n):
    global ans
    tmp = board[x][y]
    for i in range(x,x+n):
        for j in range(y,y+n):
            if tmp != board[i][j]:
                ans += '('
                quad(x,y,n//2)
                quad(x,y+n//2,n//2)
                quad(x+n//2,y,n//2)
                quad(x+n//2,y+n//2,n//2)
                ans += ')'
                return
    if tmp == '0':
        ans += '0'
    else:
        ans += '1'
ans = ''
quad(0,0,n)
print(ans)
```

4. 종이의 개수(https://www.acmicpc.net/problem/1780)
```
import sys
n = int(input())
board = [list(map(int,sys.stdin.readline().split())) for _ in range(n)]

minus_one = 0
zero = 0
one = 0

def quad(x,y,n):
    global minus_one,zero,one
    tmp = board[x][y]

    for i in range(x,x+n):
        for j in range(y,y+n):
            if tmp != board[i][j]:
                for k in range(3):
                    for l in range(3):
                        quad(x+k*n//3,y+l*n//3,n//3)
                return
    if tmp == -1:
        minus_one += 1
    elif tmp == 0:
        zero += 1
    elif tmp == 1:
        one +=1
quad(0,0,n)
print(minus_one)
print(zero)
print(one)
```

### DAY 48 (20.10.18)
1. 키 순서(https://www.acmicpc.net/problem/2458)
```
import sys
INF=2**32
n,m = map(int,input().split())
board = [[INF]*n for _ in range(n)]

for i in range(m):
    a,b = map(int,sys.stdin.readline().split())
    board[a-1][b-1] = 1

for i in range(n):
    board[i][i] = 0

def floyd():
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i==j or j==k or k==i:
                    continue
                elif board[i][j] == 1 or (board[i][k] == 1 and board[k][j] == 1):
                    board[i][j] = 1
floyd()
cnt = [0]*n
for i in range(n):
    for j in range(n):
        if board[i][j] == 1:
            cnt[i] += 1
            cnt[j] += 1
print(cnt.count(n-1))
```

2. 알고스팟(https://www.acmicpc.net/problem/1261)
```
from collections import deque

m,n = map(int,input().split())
board = [list(map(int,input())) for _ in range(n)]
visited =[[-1]*m for _ in range(n)]

dx = [-1,0,1,0]
dy = [0,1,0,-1]

def bfs():
    while dq:
        x,y = dq.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if 0 <= nx < n and 0 <= ny < m:
                if visited[nx][ny] == -1:
                    if board[nx][ny] == 0:
                        visited[nx][ny] = visited[x][y]
                        dq.appendleft([nx,ny])
                    else:
                        visited[nx][ny] = visited[x][y]+1
                        dq.append([nx,ny])
dq = deque()
dq.append([0,0])
visited[0][0] = 0
bfs()
print(visited[n-1][m-1])
```

### DAY 50 (20.10.20)
1. LCS(https://www.acmicpc.net/problem/9251)
```
A=list(input())
B=list(input())
 
lcs=[[0 for _ in range(len(B)+1)] for _ in range(len(A)+1)]
 
for i in range(1,len(A)+1):
    for j in range(1,len(B)+1):
        if A[i-1]==B[j-1]:
            lcs[i][j]=lcs[i-1][j-1]+1
        else:
            lcs[i][j]=max(lcs[i-1][j],lcs[i][j-1])
            
print(lcs[len(A)][len(B)])
```

2. 줄 세우기(https://www.acmicpc.net/problem/2252)
```
import sys
from collections import deque

n,m = map(int,sys.stdin.readline().split())
tree = [[] for _ in range(n+1)]

inDegree = [0 for _ in range(n+1)]
dq = deque()
result = []

for _ in range(m):
    s,e = map(int,sys.stdin.readline().split())
    tree[s].append(e)
    inDegree[e]+=1

for i in range(1,n+1):
    if inDegree[i] == 0:
        dq.append(i)

while dq:
    a = dq.popleft()
    result.append(a)
    for t in tree[a]:
        inDegree[t]-=1
        if inDegree[t]==0:
            dq.append(t)
            
print(*result)
```

### DAY 51 (20.10.21)
1. 집합의 표현(https://www.acmicpc.net/problem/1717)
```
import sys

input = sys.stdin.readline

def get_parent(x):
    if parent[x] == x: 
        return x
    p = get_parent(parent[x])
    parent[x] = p
    return p

def union(x,y):
    x = get_parent(x)
    y = get_parent(y)

    if x != y:
        parent[y] = x

def find_parent(x):
    if parent[x] == x:
        return x
    return find_parent(parent[x])

n,m = map(int,input().split())
parent = {}

for i in range(n+1):
    parent[i] = i

for _ in range(m):
    z,x,y = map(int,input().split())

    if not z:
        union(x,y)
    else:
        if find_parent(x) == find_parent(y):
            print("YES")
        else:
            print('NO')
```

2. 스타트링크(https://www.acmicpc.net/problem/5014)
```
import sys
from collections import deque
input = sys.stdin.readline

f,s,g,u,d = map(int,input().split())

dx = [u,-d]
dp = [-1 for i in range(f)]

def bfs(x):
    visited = [0 for _ in range(f)]
    visited[x] = 1
    dq = deque()
    dq.append(x)
    while dq:
        x = dq.popleft()
        for i in range(2):
            nx = x + dx[i]
            if nx < 0 or nx >= f:
                continue
            else:
                if visited[nx] == 0:
                    dq.append(nx)
                    dp[nx] = dp[x] + 1
                    visited[nx] = 1
dp[s-1] = 0
bfs(s-1)

if dp[g-1] == -1:
    print("use the stairs")
else:
    print(dp[g-1])
```

3. 보물섬(https://www.acmicpc.net/problem/2589)
```
from collections import deque

n,m = map(int,input().split())
board = [list(map(str,input())) for _ in range(n)]

dx = [-1,0,1,0]
dy = [0,1,0,-1]

def bfs(x,y):
    visited =[[0]*(m) for _ in range(n)]
    visited[x][y] = 1
    dq = deque()
    dq.append((x,y))
    num = 0
    while dq:
        x,y = dq.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if nx < 0 or nx >= n or ny < 0 or ny >= m:
                continue
            else:
                if visited[nx][ny] == 0 and board[nx][ny] == 'L':
                    visited[nx][ny] = visited[x][y] + 1
                    num = max(num,visited[nx][ny])
                    dq.append((nx,ny))
    return num-1
cnt = 0
for i in range(n):
    for j in range(m):
        if board[i][j] == 'L':
            cnt = max(cnt,bfs(i,j))
print(cnt)
```

### DAY 53 (20.10.23)
1. 랜선 자르기(https://www.acmicpc.net/problem/1654)
```
import sys
from collections import deque
import math
input = sys.stdin.readline

K,N = map(int,input().split())
lan = [int(sys.stdin.readline()) for _ in range(K)]
lan.sort()
left = 1
right = max(lan)

while left<=right:
    mid = (left+right)//2
    lines = 0
    for i in lan:
        lines += i//mid
    if lines >= N:
        left = mid+1
    else:
        right = mid-1
print(right)
```

2. 벽 부수고 이동하기(https://www.acmicpc.net/problem/2206)
```
import sys
from collections import deque
import math
input = sys.stdin.readline

N,M= map(int,input().split())
board = [[0]*M for _ in range(N)]
for i in range(N):
    arr = input()
    for j in range(len(arr)-1):
        board[i][j] = int(arr[j])
dx = [-1,1,0,0]
dy = [0,0,1,-1]

def bfs():
    dq = deque()
    dq.append((0,0,1))
    visited = [[[0]*2 for _ in range(M)] for _ in range(N)]
    visited[0][0][1] = 1
    while dq:
        x,y,z = dq.popleft()
        if x == N-1 and y == M-1:
            return visited[x][y][z]
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if nx < 0 or nx >= N or ny <0 or ny >= M:
                continue
            else:
                if board[nx][ny] == 1 and z == 1:
                    visited[nx][ny][0] = visited[x][y][1] + 1
                    dq.append((nx,ny,0))
                elif board[nx][ny] == 0 and visited[nx][ny][z] == 0:
                    visited[nx][ny][z] = visited[x][y][z] + 1
                    dq.append((nx,ny,z))
    return -1
print(bfs())
```

3. 케빈 베이컨의 6단계 법칙(https://www.acmicpc.net/problem/1389)
```
import sys
from collections import deque
import math
input = sys.stdin.readline
INF=2**32

N,M = map(int,input().split())
board = [[INF]*N for _ in range(N)]

for i in range(M):
    a,b = map(int,sys.stdin.readline().split())
    board[a-1][b-1] = 1
    board[b-1][a-1] = 1

for i in range(N):
    board[i][i] = 0

def floyd():
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if i==j or j==k or k==i:
                    continue
                else:
                    board[i][j] = min(board[i][j],board[i][k]+board[k][j])
floyd()
ans = []
for i in board:
    ans.append(sum(i))
for i in range(N):
    if ans[i] == min(ans):
        print(i+1)
        break
```

### DAY 58 (20.10.28)
1. 공주님을 구해라!(https://www.acmicpc.net/problem/17836)
```
import sys
from collections import deque
import math
import copy
input = sys.stdin.readline

N,M,T = map(int,input().split())
board = [list(map(int, input().split())) for _ in range(N)]
visited = [[[0]*2 for _ in range(M)] for _ in range(N)]

dx = [-1,1,0,0]
dy = [0,0,1,-1]

def bfs():
    dq = deque()
    dq.append((0,0,0,0))
    visited[0][0][0] = 1
    while dq:
        x,y,cnt,sw = dq.popleft()
        if board[x][y] == 2:
            sw = 1
        if x == N-1 and y == M-1: 
            return cnt
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if sw == 1:
                if 0<=nx<N and 0<=ny<M and visited[nx][ny][sw] == 0:
                    visited[nx][ny][sw] = cnt + 1
                    dq.append((nx,ny,cnt+1,sw))
            else:
                if 0<=nx<N and 0<=ny<M and visited[nx][ny][sw] == 0 and board[nx][ny] != 1:
                    visited[nx][ny][sw] = cnt + 1
                    dq.append((nx,ny,cnt+1,sw))
    return -1
ans = bfs()
if 0 <= ans <= T:
    print(ans)
else:
    print("Fail")
```

### DAY 60 (20.10.30)
1. 보석 도둑(https://www.acmicpc.net/problem/1202)
```
import sys
from collections import deque
import math
import copy
import heapq
input = sys.stdin.readline

n,k = map(int,input().split())
gem = []
for i in range(n):
    m,v = map(int,input().split())
    gem.append((m,v))
bag = []
for i in range(k):
    c = int(input())
    bag.append(c)
gem.sort()
bag.sort()

heap = []
j = 0
ans = 0
for i in range(k):
    while j<n and bag[i]>=gem[j][0]:
        heapq.heappush(heap,-gem[j][1])
        j+=1
    if heap:
        temp = heapq.heappop(heap)
        ans += abs(temp)
print(ans)
```

### DAY 61 (20.10.31)
1. 다리 만들기(https://www.acmicpc.net/problem/2146)
```
import sys
from collections import deque
input = sys.stdin.readline
sys.setrecursionlimit(10**6)

n = int(input())
board = [list(map(int, input().split())) for _ in range(n)]
visited = [[False]*n for _ in range(n)]
dx = [-1,1,0,0]
dy = [0,0,-1,1]
idx = 1

# 섬 번호 붙이기
def dfs(x,y):
    visited[x][y] = True
    board[x][y] = idx
    for i in range(4):
        nx,ny = x + dx[i], y + dy[i]
        if nx < 0 or nx >= n or ny < 0 or ny >= n:
            continue
        if not visited[nx][ny] and board[nx][ny]:
            dfs(nx,ny)

for i in range(n):
    for j in range(n):
        if board[i][j] and not visited[i][j]:
            dfs(i,j)
            idx+=1

# 최솟값 저장
ans = 10**9
# 섬에서 섬끼리 거리 찾기
def bfs(idx):
    global ans
    dq = deque()
    dist = [[-1]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if board[i][j] == idx:
                dq.append((i,j))
                dist[i][j] = 0
    while dq:
        x,y = dq.popleft()
        for i in range(4):
            nx,ny = x + dx[i], y + dy[i]
            if nx < 0 or nx >= n or ny < 0 or ny >= n:
                continue
            if board[nx][ny] and board[nx][ny] != idx:
                ans = min(ans,dist[x][y])
                return
            if not board[nx][ny] and dist[nx][ny] == -1:
                dist[nx][ny] = dist[x][y] + 1
                dq.append((nx,ny))

for i in range(1,idx+1):
    bfs(i)

print(ans)
```

### DAY 62 (20.11.01)
1. 점프(https://www.acmicpc.net/problem/1890)
```
import sys
from collections import deque
import math
input = sys.stdin.readline

n = int(input())
board = [list(map(int, input().split())) for _ in range(n)]
dp = [[0]*n for _ in range(n)]
dp[0][0] = 1

for i in range(n):
    for j in range(n):
        if board[i][j] == 0:
            continue
        if i + board[i][j] < n:
            dp[i+board[i][j]][j] += dp[i][j]
        if j + board[i][j] < n:
            dp[i][j+board[i][j]] += dp[i][j]
print(dp[n-1][n-1])
```

2. 내리막 길(https://www.acmicpc.net/problem/1520)
```
import sys
from collections import deque
import math
input = sys.stdin.readline
sys.setrecursionlimit(10000)

m,n = map(int,input().split())
board = [list(map(int, input().split())) for _ in range(m)]
dp = [[-1]*n for _ in range(m)]

dx = [-1,1,0,0]
dy = [0,0,-1,1]

def dfs(x,y):
    if x == 0 and y == 0:
        return 1
    if dp[x][y] == -1:
        dp[x][y] = 0
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if 0 <= nx < m and 0 <= ny < n:
                if board[x][y] < board[nx][ny]:
                    dp[x][y] += dfs(nx,ny)
    return dp[x][y]
print(dfs(m-1,n-1))
```

### DAY 66 (20.11.05)
1. 뱀(https://www.acmicpc.net/problem/3190)
```
import sys
from collections import deque
input = sys.stdin.readline

n = int(input())
board = [[0]*n for _ in range(n)]
k = int(input())
for i in range(k):
    a,b = map(int,input().split())
    board[a-1][b-1] = 1
l = int(input())
move = [list(map(str, input().split())) for _ in range(l)]
move = deque(move)
tail = deque()

dx = [0,1,0,-1]
dy = [1,0,-1,0]

def bfs():
    dq = deque()
    dq.append((0,0,0))
    tail.append((0,0))
    visited = [[0]*n for _ in range(n)]
    visited[0][0] = 1
    time = 0
    while True:
        x,y,d = dq.popleft()
        time+=1
        nx,ny = x + dx[d], y + dy[d]
        # 벽에 부딪히거나 자신과 부딪힐 경우
        if nx < 0 or nx >= n or ny < 0 or ny >= n or visited[nx][ny] == 1:
            print(time)
            exit()
        else:
            tail.append((nx,ny))
            visited[nx][ny] = 1

            # 방향 전환
            if len(move) > 0:
                if int(move[0][0]) == time:
                    # 좌회전
                    if move[0][1] == 'L':
                        d = (d-1)%4
                    # 우회전
                    else:
                        d = (d+1)%4
                    move.popleft()
                    
            dq.append((nx,ny,d))

            # 사과가 있다면
            if board[nx][ny] == 1:
                board[nx][ny] = 0
                continue
            # 사과가 없다면
            else:
                sx,sy = tail.popleft()
                visited[sx][sy] = 0
bfs()
```

2. 프로그래머스 월간 코드 챌린지 시즌1(https://abit.ly/d76z0m)

![1](https://user-images.githubusercontent.com/26870568/98255167-78744600-1fc0-11eb-9f25-526bc2fea16e.PNG)

### DAY 67 (20.11.06)
1. 달이 차오른다, 가자.(https://www.acmicpc.net/problem/1194)
```
import sys
from collections import deque
input = sys.stdin.readline

n,m = map(int,input().split())
board = [list(input().strip()) for _ in range(n)]
visited = [[[0]*64 for _ in range(m)] for _ in range(n)]

dx = [-1,1,0,0]
dy = [0,0,-1,1]

def bfs(x,y):
    dq = deque()
    dq.append((x,y,0,0))
    visited[x][y][0] = 1
    while dq:
        x,y,key,dist = dq.popleft()

        if board[x][y] == '1':
            return dist

        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if 0 <= nx < n and 0 <= ny < m and visited[nx][ny][key] == 0 and board[nx][ny] != '#':
                #print(nx,ny,board[nx][ny],dist)
                if board[nx][ny].isupper():
                    if key & 1 << (ord(board[nx][ny]) - 65):
                        visited[nx][ny][key] = visited[x][y][key] +1
                        dq.append((nx,ny,key,dist+1))
                elif board[nx][ny].islower():
                    nKey = key | (1 << ord(board[nx][ny]) - 97)
                    if visited[nx][ny][nKey] == 0:
                        visited[nx][ny][nKey] = visited[x][y][key] + 1
                        dq.append((nx,ny,nKey,dist+1))
                else:
                    visited[nx][ny][key] = visited[x][y][key] + 1
                    dq.append((nx,ny,key,dist+1))
    return -1

for i in range(n):
    for j in range(m):
        if board[i][j] == '0':
            print(bfs(i,j))
```

2. 부등호(https://www.acmicpc.net/problem/2529)
```
import sys
sys.setrecursionlimit(10000)
input = sys.stdin.readline

k = int(input())
st = list(input().split())
node = [0,1,2,3,4,5,6,7,8,9]
visited = [0,0,0,0,0,0,0,0,0,0]
ans = []

def dfs(v,cnt,num,idx):
    if cnt == k:
        ans.append(num)
    else:
        for i in node:
            if visited[i] == 0:
                if st[idx] == '<':
                    if v >= i:
                        continue
                else:
                    if v <= i:
                        continue
                visited[i] = True
                dfs(i,cnt+1,num+str(i),idx+1)
    visited[v] = False

for i in node:
    visited[i] = 1
    dfs(i,0,str(i),0)
    
print(ans[-1])
print(ans[0])
```

### DAY 120 (20.12.29)
1. 프렌즈4블록(https://programmers.co.kr/learn/courses/30/lessons/17679)
```
import copy

def solution(m, n, board):
    # 초기 배열
    n_board = [[0]*n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            n_board[i][j] = board[i][j]
    flag = True;
    while flag:
        # 지울 블록 담은 배열
        tmp_board = copy.deepcopy(n_board)
        # 지울 블록 선택하기
        flag = False;
        for x in range(m):
            for y in range(n):
                if x+1 < m and y+1 < n and n_board[x][y] != '0' and (n_board[x][y] == n_board[x+1][y] and n_board[x][y] == n_board[x][y+1] and n_board[x][y] == n_board[x+1][y+1]):
                    tmp_board[x][y] = '0'
                    tmp_board[x+1][y] = '0'
                    tmp_board[x][y+1] = '0'
                    tmp_board[x+1][y+1] = '0'
                    flag = True;
        after_board = []
        # 블록 지우고 아래로 내리기
        for i in range(n):
            tmp_list = []
            for j in range(m):
                if tmp_board[j][i] != '0':
                    tmp_list.append(tmp_board[j][i])
            after_board.append(tmp_list)
        # 마지막 연산후 배열
        last_board = [['0']*n for _ in range(m)]
        for i in range(n):
            cnt = len(after_board[i])
            idx = 0
            for j in range(m):
                if cnt == m:
                    last_board[j][i] = after_board[i][idx]
                    idx+=1
                else:
                    cnt+=1
        n_board = last_board
    ans = 0
    for i in range(m):
        for j in range(n):
            if last_board[i][j] == '0':
                ans+=1
    return ans
```

2. 압축(https://programmers.co.kr/learn/courses/30/lessons/17684)
```
def solution(msg):
    table = {}
    num = 65
    for i in range(1,27):
        table[chr(num)] = i
        num+=1
    w = 0
    c = 0
    ans = []
    while True:
        # 현재글자 + 다음글자가 사전에 있다면 w는 변화없음, c = c + 1
        c += 1
        # c가 마지막 인덱스 번호라면 while문 종료
        if len(msg) == c:
            ans.append(table[msg[w:c]])
            break
        # 현재글자 + 다음글자가 사전에 없다면 w = c, c = c + 1
        if msg[w:c+1] not in table:
            table[msg[w:c+1]] = len(table) + 1
            ans.append(table[msg[w:c]])
            w = c
    return ans
```

### DAY 129 (21.01.07)
1. 자물쇠와 열쇠(https://programmers.co.kr/learn/courses/30/lessons/60059)
```
def attach(x, y, M, key, board):
    for i in range(M):
        for j in range(M):
            board[x+i][y+j] += key[i][j]

def detach(x, y, M, key, board):
    for i in range(M):
        for j in range(M):
            board[x+i][y+j] -= key[i][j]

def rotate90(arr):
    return list(zip(*arr[::-1]))
    
def check(board, M, N):
    for i in range(N):
        for j in range(N):
            if board[M+i][M+j] != 1:
                return False
    return True

def solution(key, lock):
    M, N = len(key), len(lock)
    
    if key == lock:
        return True

    board = [[0] * (M*2 + N) for _ in range(M*2 + N)]
    # 자물쇠 중앙 배치
    for i in range(N):
        for j in range(N):
            board[M+i][M+j] = lock[i][j]

    rotated_key = key
    # 모든 방향 (4번 루프)
    for _ in range(4):
        rotated_key = rotate90(rotated_key)
        for x in range(1, M+N):
            for y in range(1, M+N):
                # 열쇠 넣어보기
                attach(x, y, M, rotated_key, board)
                # lock 가능 check
                if(check(board, M, N)):
                    return True
                # 열쇠 빼기
                detach(x, y, M, rotated_key, board)          
    return False
```

### DAY 134 (21.01.12)
1. 연료 채우기(https://www.acmicpc.net/problem/1826)
```
import heapq

n = int(input())
# 최소힙
heap = []
for i in range(n):
    heapq.heappush(heap, list(map(int,input().split())))
l,p = map(int,input().split())

cnt = 0
# 최대힙
target = []
# 현재 기름으로 도착지에 도착할 수 있으면 멈춤
while p < l:
    # 현재 연료로 갈 수 있는 주유소 리스트 최대힙에 push
    while heap and heap[0][0] <= p:
        a,b = heapq.heappop(heap)
        heapq.heappush(target, [-1*b,a])
        
    # 이동할 수 있는 주유소 없으면 불가능
    if not target:
        cnt = -1
        break
        
    # 최대힙에서 연료가 가장 많은 주유소 pop
    a,b = heapq.heappop(target)
    p += -1*a
    cnt+=1
    
print(cnt)
```

### DAY 136 (21.01.14)
1. 카드 정렬하기(https://www.acmicpc.net/problem/1715)
```
import sys
import heapq
input = sys.stdin.readline

n = int(input())
heap = []
for i in range(n):
    heapq.heappush(heap,int(input()))

if len(heap) == 1:
    print(0)
else:
    ans = 0
    while len(heap) > 1:
        Sum = heapq.heappop(heap) + heapq.heappop(heap)
        ans += Sum
        heapq.heappush(heap,Sum)
    print(ans)
```

2. 빗물(https://www.acmicpc.net/problem/14719)
```
h,w = map(int,input().split())
arr = list(map(int,input().split()))
maxH = 0
maxIdx = 0
for i in range(len(arr)):
    if maxH < arr[i]:
        maxH = arr[i]
        maxIdx = i
total = 0
temp = 0
for i in range(0,maxIdx+1):
    if arr[i] > temp:
        temp = arr[i]
    total += temp
temp = 0
for i in range(len(arr)-1,maxIdx,-1):
    if arr[i] > temp:
        temp = arr[i]
    total += temp
print(total-sum(arr))
```

3. 평범한 배낭(https://www.acmicpc.net/problem/12865)
```
n,k = map(int,input().split())
bag = [(0,0)]
for i in range(n):
    w,v = map(int,input().split())
    bag.append((w,v))
dp = [[0]*(k+1) for _ in range(n+1)]
for i in range(1,n+1):
    for j in range(1,k+1):
        w = bag[i][0]
        v = bag[i][1]
        if j >= w:
            dp[i][j] = max(v+dp[i-1][j-w],dp[i-1][j])
        else:
            dp[i][j] = dp[i-1][j]
print(dp[n][k])
```

### DAY 138 (21.01.16)
1. ABCDE(https://www.acmicpc.net/problem/13023)
```
import sys
input = sys.stdin.readline
sys.setrecursionlimit(10 ** 9)

n,m = map(int,input().split())
graph = [[] for _ in range(n)]

for i in range(m):
    a,b = map(int,input().split())
    graph[a].append(b)
    graph[b].append(a)

visited = [False for _ in range(n)]

def dfs(v,depth):
    global ans
    visited[v] = True
    if depth >= 4:
        ans = True
        return
    for next_node in graph[v]:
        if not visited[next_node]:
            dfs(next_node,depth+1)
            visited[next_node] = False
            
ans = False
for i in range(n):
    dfs(i,0)
    visited[i] = False
    if ans:
        print(1)
        exit()
print(0)
```

### DAY 153 (21.01.31)
1. 메뉴 리뉴얼(https://programmers.co.kr/learn/courses/30/lessons/72411)
```
from itertools import combinations

def solution(orders, course):
    Dict = {}
    for i in orders:
        arr = []
        for j in i:
            arr.append(j)
        #print(arr)
        for j in course:
            if j <= len(arr):
                #print(j)
                n_course = list(combinations(arr,j))
                for k in n_course:
                    elem = ''.join(sorted(k))
                    #print(elem)
                    if elem not in Dict:
                        Dict[elem] = 1
                    else:
                        Dict[elem] += 1
    ans = []
    for i in course:
        Max = 2
        for j in Dict:
            if i == len(j):
                Max = max(Max,Dict[j])
        for j in Dict:
            if i == len(j) and Max == Dict[j]:
                ans.append(j)
    return(sorted(ans))
```

2. 합승 택시 요금(https://programmers.co.kr/learn/courses/30/lessons/72413)
```
import sys
from heapq import heappush, heappop
INF = sys.maxsize

def solution(n, s, a, b, fares):
    graph = [[] for _ in range(n+1)]
    
    for i in fares:
        # 무방향 그래프
        graph[i[0]].append([i[1],i[2]])
        graph[i[1]].append([i[0],i[2]])
    
    def dijkstra(start,target):
        dp = [INF for i in range(n+1)]
        heap = []
        dp[start] = 0
        heappush(heap,[0,start])
        while heap:
            wei,now = heappop(heap)
            
            if dp[now] < wei:
                continue
            for next_node, w in graph[now]:
                next_wei = w + wei
                if next_wei < dp[next_node]:
                    dp[next_node] = next_wei
                    heappush(heap,[next_wei,next_node])
        return dp[target]
    ans = dijkstra(s,a) + dijkstra(s,b)
    for i in range(1,n+1):
        if s!= i:
            ans = min(ans,dijkstra(s,i) + dijkstra(i,a) + dijkstra(i,b))
    return ans
```

### DAY 300 (21.05.27)
1. 눈덩이 굴리기(https://www.acmicpc.net/problem/21735)
```
n,m = map(int,input().split())
arr = [0] + list(map(int,input().split()))

def dfs(index,snow,depth):
    global ans
    if depth > m:
        return
    if depth <= m:
        ans = max(ans,snow)
    if index <= n-1:
        dfs(index+1,snow+arr[index+1],depth+1)
    if index <= n-2:
        dfs(index+2,snow//2+arr[index+2],depth+1)
    return
ans = -1
dfs(0,1,0)
print(ans)
```
2. 헌내기는 친구가 필요해(https://www.acmicpc.net/problem/21736)
```
from collections import deque

dx = [0,0,1,-1]
dy = [1,-1,0,0]
n,m = map(int,input().split())
board = [list(map(str, input())) for _ in range(n)]

def bfs(x,y):
    dq = deque()
    dq.append((x,y))
    visited = [[False]*m for _ in range(n)]
    cnt = 0
    while dq:
        x,y = dq.popleft()
        for k in range(4):
            nx,ny = x+dx[k],y+dy[k]
            if 0 <= nx < n and 0 <= ny < m:
                if visited[nx][ny] == False:
                    if board[nx][ny] == 'O':
                        dq.append((nx,ny))
                        visited[nx][ny] = True
                    elif board[nx][ny] == 'P':
                        dq.append((nx,ny))
                        cnt+=1
                        visited[nx][ny] = True
    if cnt == 0:
        print('TT')
    else:
        print(cnt)
for i in range(n):
    for j in range(m):
        if board[i][j] == 'I':
            bfs(i,j)
```

3. SMUPC 계산기(https://www.acmicpc.net/problem/21737)
```
from collections import deque

n = int(input())
s = input()
if s.count('C') == 0:
    print("NO OUTPUT")
else:
    dq = deque()
    NUM = ''
    for i in s:
        if i == 'S':
            if NUM != '':
                dq.append(int(NUM))
                NUM = ''
            dq.append('-')
        elif i == 'M':
            if NUM != '':
                dq.append(int(NUM))
                NUM = ''
            dq.append('*')
        elif i == 'U':
            if NUM != '':
                dq.append(int(NUM))
                NUM = ''
            dq.append('/')
        elif i == 'P':
            if NUM != '':
                dq.append(int(NUM))
                NUM = ''
            dq.append('+')
        elif i == 'C':
            result = 0
            if NUM != '':
                dq.append(int(NUM))
                NUM = ''
            cnt = 0
            while dq:
                x = dq.popleft()
                if cnt == 0:
                    result = x
                    cnt+=1
                if x == '-':
                    y = dq.popleft()
                    result -= y
                if x == '+':
                    y = dq.popleft()
                    result += y
                if x == '/':
                    y = dq.popleft()
                    if result < 0 and y > 0:
                        result *= -1
                        result //= y
                        result *= -1
                    elif result > 0 and y < 0:
                        y *= -1
                        result //= y
                        result *= -1
                    else:
                        result //= y
                if x == '*':
                    y = dq.popleft()
                    result *= y
            print(result,end=' ')
            dq.append(result)
        else:
            NUM += i
```
