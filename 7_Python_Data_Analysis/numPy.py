from numpy  import *
a = arange(15).reshape(3, 5)
print("--array--")
print(a)

print("a.shape")
print(a.shape)
print("a.size")
print(a.size)
print(" a.ndim")
print(a.ndim)
print("a.dtype.name")
print(a.dtype)


print("the number of rows in a:", a.shape[0])
print("the number of columns in a:", a.shape[1])

print("---create array---")
a = array( [2,3,4] )
print(a)
print("a.dtype is...")
print(a.dtype)
b = array([1.2, 3.5, 5.1])
print(b)
print("b.dtype is...")
print(b.dtype)

c = array( [ ((1.5,2,3), (4,5,6),(7,8,9)),((2.5,5,3), (4,5,6),(7,8,9)) ] )
print(c)
print("c.ndim is ...")
print(c.ndim)
print("c.dtype is...")
print(c.dtype)

d = zeros( (3,4) )
print(d)
e = ones( (3,4),dtype=float )
print(e)
#create a matrix with random values
f = empty( (2,3) )
print(f)
# arange() is similar to range()
g = arange( 10, 30, 3 ) #(start,end,difference)
print(g)

h = linspace( 0, 2, 6 )                 # 6 numbers between 0 and 2
print("h=")
print(h) #[0.  0.4 0.8 1.2 1.6 2. ]
i = random.random((2,3))
print(i)

print("--------print a matrix--------")

print("arange(6)")
print(arange(6))
print( "arange(12).reshape(4,3)" )
print( arange(12).reshape(4,3) )
print("arange(24).reshape(2,3,4)" )
print( arange(24).reshape(2,3,4) )
print( "arange(10000)")
print( arange(10000))
print("arange(10000).reshape(100,100)")
print (arange(10000).reshape(100,100))

print("--------basic operations--------")
a = array( [20,30,40,50] )
print("a=")
print(a)
b = array([0, 1, 2, 3])
print("b=")
print(b)
c = a-b
print("c=a-b")
print(c)
print("b**2=")
print(b**2)
print("a<35")
print(a<35)

A = array( [[1,1],
            [0,1]] )
B = array( [[2,0],
            [3,4]] )
print("A=")
print(A)
print("B=")
print(B)
print("A*B=")
print(A*B)
print("dot(A,B)=")
print(dot(A,B))
print(A.dot(B))

print("--------override operations--------")
print("A+B=")
print(A+B)
print("A=")
print(A)
print("exec A+=B")
A += B
print("A=")
print(A)

print("A*B=")
print(A*B)
print("A=")
print(A)
print("exec A*=B")
A *= B
print("A=")
print(A)

C = ones((2,3), dtype=int)
print("C")
print(C.dtype)
print(C)
D =array([[ 3.69092703,  3.8324276 ,  3.0114541 ],
       [ 3.18679111,  3.3039349 ,  3.37600289]])

print("D")
print(D.dtype)
print(D)
print("exec C =+ D")
# C+=D #>>TypeError: Cannot cast ufunc add output from dtype('float64') to dtype('int64') with casting rule 'same_kind'

print("unary operations")
a = array([1,2,3,4])
print("a=")
print(a)
print("a.sum()")
print(a.sum())
print("a.min()")
print(a.min())
print("a.max()")
print(a.max())


b = arange(12).reshape(3,4)
print("b=")
print(b)
print("sum of each column")
print("b.sum(axis=0)")
print(b.sum(axis=0))  
print("sum of each row")
print("b.sum(axis=1)")
print(b.sum(axis=1))
print("")

print("min of each row")
print("b.min(axis=1)") 
print(b.min(axis=1))                  

print("cumulative sum along each row")
print(" b.cumsum(axis=1)" )
print( b.cumsum(axis=1) )    

print("Universal Functions")
B = arange(3)
print("B=")
print(B)
print("exp(B)")
print(exp(B))
print("sqrt(B)")
print(sqrt(B))
C = array([2., -1., 4.])
print("C=")
print(C)
print(add(B,C))

print("Indexing, Slicing and Iterating")
a = arange(10)**3
print("a")
print(a)
print("a[2]")
print(a[2])
print("a[2:5]")
print(a[2:5])
print(" a[ : :-1]")
print( a[ : :-1])
for i in a:
  print(i**(1/3.))

print("Multidimensional arrays ")
def fun(x,y):
  return 100*x + y
b = fromfunction(fun,(5,4),dtype=int) # fill elements in matrix utilizing function
print("b")
print(b)
print("b[2]")#second row
print(b[2])
print("b[:,2]") #second column
print(b[:,2])
print("b[2,3]")
print(b[2,3])
print("b[0:3, 1]")
print(b[0:3, 1]) # each row in the second column of b
print("b[ : ,1]")
print(b[ : ,1])

print("b[1:3, : ]")  # each column in the second and third row of b
print(b[1:3, : ])
print("for loop rows in b")
for row in b:
  print(row)
print("for loop elements in b (flat)")
for element in b.flat:
  print (element)

#b[i] のブラケットの中の式は、i の後に残りの軸を表現するのに必要な分の : が続いているものとして扱われる。
#dots (...) represent as many colons as needed to produce a complete indexing tuple.
c = array( [ [[  0,  1,  2],               # ３次元配列（２つ積まれた２次元配列）
             [ 10, 12, 13]],
            [[100,101,102],
            [110,112,113]] ] )
print("c=")
print(c)
print("c.shape")
print(c.shape)
print("c[1,...]")
print(c[1,...])
print("c[1,:,:]")
print(c[1,:,:])

print("Shape Manipulation")

a= array([[ 2.,  8.,  0.,  6.],
       [ 4.,  5.,  1.,  1.],
       [ 8.,  9.,  3.,  6.]])
print("a")
print(a)
print("a.shape")
print(a.shape)
 # returns the array, flatteneda.ravel()  # returns the array, flattened
print("a.ravel() ")
print(a.ravel())
# returns the array, transposed 転置
print("a.T")
print(a.T)
print("a.T.shape")
print(a.T.shape)

# returns the array with a modified shape
print("a.reshape(6,2)")
print(a.reshape(6,2)  )
print("a.reshape(2,6)")
print(a.reshape(2,6)  )

print("a.reshape(3,-1)") #>>equal to a.reshape(3,4)
print(a.reshape(3,-1))

print("Stacking together different arrays")
a=array([[ 8.,  9.],
       [ 0.,  1.]])
print("a")
print(a)

b=array([[ 1.,  8.],
       [ 0.,  4.]])
print("b")
print(b)       
print("vstack((a,b)) virtical stack")
print(vstack((a,b)))
print("hstack((a,b)) horizontal stack")
print(hstack((a,b)))

print("column_stack((a,b))")
print(column_stack((a,b)))
print("row_stack((a,b))")
print(row_stack((a,b)))

c= array([(1,2,3),(4,5,6)])
d= array([(7,8,9),(10,11,12)])
print("c")
print(c)       
print("d")
print(d)       
print("column_stack((c,d))")
print(column_stack((c,d)))
print("row_stack((c,d))")
print(row_stack((c,d)))

a=array([4.,2.])
print("a")
print(a)
print("a[:,newaxis]")
print(a[:,newaxis])

print("Splitting one array into several smaller ones")
a=array([[ 9.,  5.,  6.,  3.,  6.,  8.,  0.,  7.,  9.,  7.,  2.,  7.],
       [ 1.,  4.,  9.,  2.,  2.,  1.,  0.,  6.,  2.,  2.,  4.,  0.]])
print("a")
print(a)
print("hsplit(a,3)")
print(hsplit(a,3))   # Split a into 3
print("vsplit(a,2)")
print(vsplit(a,2))
print("hsplit(a,(3,5))") 
print(hsplit(a,(3,5)))   # aを３列目と5列目で分割

print("Copies and Views")
a = arange(12)
b = a
print("b is a")
print(b is a)
print("id(a)")
print(id(a))
print("id(b)")
print(id(b))
a[3]=1000
print("a")
print(a)
print("b")
print(b)
b.shape=3,4
print("a.shape")
print(a.shape)
print("b.shape")
print(b.shape)

##Shallow Copy##viewメソッドは同じデータを参照する新しい配列オブジェクトを作成する。
print("Shallow Copy##view")
a = arange(12)
print("a")
print(a)

c = a.view()
print("c is a")
print(c is a)
print("id(a)")
print(id(a))
print("id(c)")
print(id(c))
c[3]=30
print("a")
print(a)
print("c")
print(c)

c.shape = 2,6
print("a.shape")
print(a.shape)
print("c.shape")
print(c.shape)

##Deep Copy
print("Deep Copy")
a = arange(12)
d = a.copy() 

print("a")
print(a)
print("d")
print(d)
print("d is a")
print(d is a)
print("id(a)")
print(id(a))
print("id(d)")
print(id(d))
print("d[0] = 9999")
d[0] = 9999
print("a")
print(a)
print("d")
print(d)
print("d.shape=2,6")
d.shape=2,6
print("a")
print(a)
print("d")
print(d)

print("Fancy indexing and index tricks")
a = array([0, 10, 20, 30, 40, 50, 60])
print("a")
print(a)
i = array( [ 1,1,3,5 ] ) 
print("i")
print(i)
print("a[i] ")
print(a[i] )
print("a[i]=[100,100,300,500]")
a[i]=[100,100,300,500]
print("a")
print(a)

a = array([0, 10, 20, 30, 40, 50, 60])
print("a")
print(a)
j = array( [ [ 3, 4], [ 2, 5 ] ] )
print("j")
print(j)
print("a[j] ")
print(a[j] )

#添字付けされた配列aが多次元配列のとき、添字の単一配列はaの最初の次元を参照する。以下の例では、ラベル画像をパレットを用いてカラー画像に変換することでこの振舞いを例示する。
#When the indexed array a is multidimensional, a single array of indices refers to the first dimension of a. 
# The following example shows this behavior by converting an image of labels into a color image using a palette.
palette = array( [ [0,0,0,0],                # black
                    [100,255,0,0],              # red
                    [200,0,255,0],              # green
                    [300,0,0,255],              # blue
                    [400,255,255,255] ] ) # white
print("palette")
print(palette)
image = array( [ [ 0, 1, 2, 0 ],           # each value corresponds to a color in the palette
                  [ 0, 3, 4, 0 ]  ] )                    
print("image")
print(image)
print("palette[image]")                  
print(palette[image])                  

##the search of the maximum value of time-dependent series:
time = linspace(20, 145, 5)
data = sin(arange(20)).reshape(5,4) 
print("time")
print(time)
print("data")
print(data)
ind = data.argmax(axis=0) #dataの各列ごとの最大値を見つけ、その最大要素のインデックスを返す
print("ind")
print(ind)

time_max = time[ind]   
print("time_max : time[ind]   ")
print(time[ind])

data_max = data[ind, range(data.shape[1])] # => data[ind[0],0], data[ind[1],1]... 
#.shape[1] >> the number of columns in matrix 
# therefore, range(shape[1]) -> [0,1,2,3]
print("data_max")
print(data_max)

print("all(data_max == data.max(axis=0))")
print(all(data_max == data.max(axis=0)))

##Indexing with Boolean Arrays
a = arange(12).reshape(3,4)
b = a > 4
print("a")
print(a)
print("b")
print(b)
print("a[b]")
print(a[b])
print("a[b]=0")
a[b]=0
print("a")
print(a)

print("Mandelbrot")
a = arange(12).reshape(3,4)
print("a")
print(a)
b1 = array([False,True,True])        
b2 = array([True,False,True,False])  

print("a[b1,b2]")
print(a[b1,b2])

print("a[(1,2),(0,2)]")
print(a[(1,2),(0,2)])

print("The ix_() function")
a = array([2,3,4,5])
b = array([8,5,4])
c = array([5,4,6,8,3])
#a[i] b[j] c[k]のすべての組み合わせについての計算結果を配列で出力したい→ix_()を使う →4x3x5の配列になる
ax, bx, cx = ix_(a,b,c)
print("ax")
print(ax)
print("bx")
print(bx)
print("cx")
print(cx)
print("ax.shape, bx.shape, cx.shape")
print(ax.shape, bx.shape, cx.shape)

result = ax+bx*cx
print("result = ax+bx*cx")
print(result)
print("result.shape")
print(result.shape)

print("result[3,2,4]")
print(result[3,2,4])
print("a[3]+b[2]*c[4]")
print(a[3]+b[2]*c[4])

print("Simple Array Operations")
a = array([[1.0, 2.0], [3.0, 4.0]])
print("a")
print(a)
print("a.transpose()")
print(a.transpose())
print("a.T")
print(a.T)
print("linalg.inv(a) inverse matrix")
print(linalg.inv(a))
print("dot(linalg.inv(a),a)")
print(dot(a,linalg.inv(a)))

u = eye(2)
print("u=eye(2)")
print(eye(2))
print("trace(u)")# trace = the sum of the elements on the main diagonal 
print(trace(u))

y = array([[5.], [7.]])
print("a")
print(a)
print("y")
print(y)
print("linalg.solve(a, y)") #solve simultaneous linear equation
print(linalg.solve(a, y))

print("linalg.eig(j)")#固有ベクトルeigenvalue (Ax = λx)
print(linalg.eig(j))

a=array([(8,1),(4,5)])
print(a)
print(linalg.eig(a))
#λ1=9, eigenvalue:t(1,1)
#λ2=4, eigenvalue:t(1,-4)

#“Automatic” Reshaping
#To change the dimensions of an array, you can omit one of the sizes which will then be deduced automatically:
a = arange(30)
print("a")
print(a)
a.shape = 2,-1,3  # -1 means "whatever is needed"
print("a.shape")
print(a.shape)


"""
#histogram
import pylab
mu, sigma = 2, 0.5
v = random.normal(mu,sigma,10000)
# Plot a normalized histogram with 50 bins
pylab.hist(v, bins=50, normed=1)       # matplotlib版 (plot)
pylab.show()
# numpyでヒストグラムを計算して描画
(n, bins) = histogram(v, bins=50, normed=True)  # NumPy 版 (no plot)
pylab.plot(.5*(bins[1:]+bins[:-1]), n)
pylab.show()
"""