#!/usr/bin/env python
# coding: utf-8

# # 배열 만들기

# In[4]:


import numpy as np
test_array = np.array(["1","4",5,8],float)
test_array


# In[9]:


test_array.dtype # 데이터 타입


# *shape  

# In[10]:


test_array.shape # shape 반환 ( 4개 ) 중요 개념!!!!


# In[11]:


tensor  = [[[1,2,5,8],[1,2,5,8],[1,2,5,8]], 
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]], 
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]], 
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]]]
np.array(tensor, int).shape # (x : 4, y : 3, z : 4)


# # rehape

# In[12]:


test_matrix = [[1,2,3,4],[1,2,5,8]]
np.array(test_matrix).shape


# In[13]:


np.array(test_matrix).reshape(2,2,2)


# In[14]:


test =np.array(test_matrix).reshape(8,)
test


# In[15]:


test.reshape(-1, 1) # row값이 1로 무조건 맞춰짐


# In[16]:


test_matrix = [[[1,2,3,4], [1,2,5,8]], [[1,2,3,4], [1,2,5,8]]]
np.array(test_matrix).flatten() # 다차원을 1차원으로 바꿔줌


# # Indexing & Slicing

# In[17]:


test_exmaple = np.array([[1, 2, 3], [4.5, 5, 6]], int)
test_exmaple


# In[18]:


test_exmaple[0][0]


# In[19]:


test_exmaple[0,0]


# In[20]:


test_exmaple[0,0] = 10 # Matrix 0,0 에 12 할당
test_exmaple


# In[22]:


test_exmaple[0][0] = 5 # Matrix 0,0 에 12 할당
test_exmaple[0,0]


# In[23]:


test_exmaple = np.array([
    [1, 2, 5,8], [1, 2, 5,8],[1, 2, 5,8],[1, 2, 5,8]], int)
test_exmaple[:2,:] # 전체 row의 2열이상


# In[26]:


test_exmaple[1,:2]  #2번째row의 2개 (0부터 시작)


# In[28]:


test_exmaple[1:3] 


# In[31]:


np.arange(30) #array의 범위를 지정하여 값의 list를 생성


# In[34]:


np.arange(0,5,0.5) #floating point도 표시가능


# In[46]:


a = np.arange(100).reshape(10,10)
a[:,-1].reshape(-1,1)


# In[47]:


np.arange(30).reshape(5,6)


# # one,zero & empty

# In[48]:


np.zeros(shape=(10,), dtype=np.int8) # 10 - zero vector 생성


# In[49]:


np.zeros((2,5)) # 2 by 5 - zero matrix 생성


# In[50]:


np.ones(shape=(10,), dtype=np.int8)


# In[51]:


np.empty(shape=(10,), dtype=np.int8) # empty는 메모리 공간을 말함.


# In[52]:


np.eye(N=3, M=5, dtype=np.int8) # 대각선에 1인 값을 넣음 (N:행 M :열 )


# In[53]:


np.eye(3,5,k=2) # 열에서 3번째 부터 시작


# In[54]:


matrix = np.arange(9).reshape(3,3)
np.diag(matrix)  # 대각 행렬의 값을 추출함


# In[55]:


np.identity(n=3, dtype=np.int8)


# In[56]:


np.random.uniform(0,1,10).reshape(2,5) # 균등 분포에서 변수 추출


# In[57]:


np.random.normal(0,1,10).reshape(2,5) # 정규 분포에서 변수 추출


# # operation in array

# In[58]:


test_array = np.arange(1,11)
test_array


# In[59]:


test_array.sum(dtype=np.float)


# In[61]:


test_array = np.arange(1,13).reshape(3,4)
test_array


# In[62]:


test_array.sum()


# ## asix 중요

# In[63]:


test_array.sum(axis=1)  # 행(row)


# In[64]:


test_array.sum(axis=0) # 열(column)


# In[65]:


third_order_tensor = np.array([test_array,test_array,test_array])
third_order_tensor


# In[66]:


third_order_tensor.sum(axis=2) # 행


# In[69]:


third_order_tensor.sum(axis=1) # 열


# In[68]:


third_order_tensor.sum(axis=0) # 깊이


# In[70]:


test_array.mean(), test_array.mean(axis=0) # 평균


# In[71]:


test_array.std(), test_array.std(axis=0) # 표준편차


# In[72]:


a = np.array([ [1], [2], [3]])
b = np.array([ [2], [3], [4]])
np.hstack((a,b)) #concatenate  -  독립된 배열들을 하나로 합침


# In[73]:


a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

np.concatenate( (a,b.T) ,axis=1)


# # operations b/t arrays (중요)

# In[74]:


test_a = np.array([[1,2,3],[4,5,6]], float)


# In[75]:


test_a + test_a # Matrix + Matrix 연산


# In[76]:


test_a - test_a # Matrix - Matrix 연산


# In[77]:


test_a * test_a # Matrix내 element들 간 같은 위치에 있는 값들끼리 연산


# In[78]:


matrix_a = np.arange(1,13).reshape(3,4)
matrix_a * matrix_a   #array들의 shape은 같아야한다.


# In[79]:


test_a = np.arange(1,7).reshape(2,3)
test_b = np.arange(7,13).reshape(3,2)


# In[80]:


test_a.dot(test_b) # dot product의 연산기능 지원


# ## broadcasting(중요)

# In[81]:


test_matrix = np.array([[1,2,3],[4,5,6]], float)
scalar = 3 # 모든 위치에 3이 더해침


# In[82]:


test_matrix + scalar # Matrix - Scalar 덧셈


# In[83]:


test_matrix = np.arange(1,13).reshape(4,3)
test_vector = np.arange(10,40,10)
test_matrix+ test_vector


# ## comparison(중요)

# In[84]:


a = np.arange(10)
a


# In[85]:


a>5


# In[86]:


np.any(a>5), np.any(a<0) # 하나라도 조건이 만족한다면


# In[87]:


np.all(a>5) , np.all(a < 10) # 모두 만족한다면


# In[88]:


a = np.array([1, 3, 0], float)
a


# In[89]:


np.where(a > 0, 3, 2) # 이 조건에 만족하는 index값 3 : ture. 2:false


# In[90]:


np.where(a>0) # 이 조건에 만족하는 index값


# In[91]:


a = np.array([1, np.NaN, np.Inf], float)
np.isnan(a) # null값 추출


# In[92]:


np.isfinite(a)


# In[93]:


a = np.array([1,2,4,5,8,78,23,3])
np.argmax(a) , np.argmin(a) # 최대값 최소값 찾기


# In[94]:


a=np.array([[1,2,4,7],[9,88,6,45],[9,76,3,4]])
np.argmax(a, axis=1) , np.argmin(a, axis=0) # row 기준으로, column을 기준으로


# In[95]:


test_array = np.array([1, 4, 0, 2, 3, 8, 9, 7], float)
test_array > 3


# In[96]:


test_array[test_array > 3] #boolean index  - 조건을 만족하는 값만 뽑음


# In[97]:


A = np.array([
[12, 13, 14, 12, 16, 14, 11, 10,  9],
[11, 14, 12, 15, 15, 16, 10, 12, 11],
[10, 12, 12, 15, 14, 16, 10, 12, 12],
[ 9, 11, 16, 15, 14, 16, 15, 12, 10],
[12, 11, 16, 14, 10, 12, 16, 12, 13],
[10, 15, 16, 14, 14, 14, 16, 15, 12],
[13, 17, 14, 10, 14, 11, 14, 15, 10],
[10, 16, 12, 14, 11, 12, 14, 18, 11],
[10, 19, 12, 14, 11, 12, 14, 18, 10],
[14, 22, 17, 19, 16, 17, 18, 17, 13],
[10, 16, 12, 14, 11, 12, 14, 18, 11],
[10, 16, 12, 14, 11, 12, 14, 18, 11],
[10, 19, 12, 14, 11, 12, 14, 18, 10],
[14, 22, 12, 14, 11, 12, 14, 17, 13],
[10, 16, 12, 14, 11, 12, 14, 18, 11]])
B = A < 15
B.astype(np.int) #ture : 1 flase : 0


# In[98]:


a = np.array([2, 4, 6, 8], float) # fancy index - index를 주면 해당하는 곳에 값을 추출


# In[100]:


a[a>4]


# In[ ]:




