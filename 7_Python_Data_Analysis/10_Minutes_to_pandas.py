import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Series 1dimension
#DataFrame 2dimentions 
#...without indice/columns

s = pd.Series([1,3,5,np.nan,6,8])
print(s)



dates = pd.date_range('20130101', periods=6)
print("dates = pd.date_range('20130101', periods=6)")
print(dates)
print("list('ABCD')")
print(list('ABCD'))
 #randn(6,4) 標準正規分布による 6x4 の行列
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print("df=")
print(df)

print(np.random.randn())

df2 = pd.DataFrame({ 'A' : 1.,
                 'B' : pd.Timestamp('20130102'),
                 'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                 'D' : np.array([3] * 4,dtype='int32'),
                 'E' : pd.Categorical(["test","train","test","train"]),
                 'F' : 'foo' })
print("df2=")                 
print(df2)                 
print("df2.dtypes")
print(df2.dtypes)
print("df.head()")
print(df.head())
print("df.tail(3)")
print(df.tail(3))
print("")
print("df.index")
print(df.index)
print("df.columns")
print(df.columns)
print("df.values")
print(df.values)

##show statistic data
print("df.describe()")
print(df.describe())
#Transposing your data:
print("df.T")
print( df.T)

print("df.sort_index(axis=1, ascending=False) sort column by descending")
print(df.sort_index(axis=1, ascending=False))
print("df.sort_index(axis=0, ascending=False) sort index by descending")
print(df.sort_index(axis=0, ascending=False))

print("")
print("df.sort_values(by='B')") #列名を指定できる
print(df.sort_values(by='B'))

print("df['C']")
print(df['C'])
print("df[1:3]")
print(df[1:3])

print("df['20130102':'20130104']")
print(df['20130102':'20130104']) 

print("df.loc[dates[1]]")
print( df.loc[dates[1]])
print("df.loc[:,['A','B']]")
print(df.loc[:,['A','B']])
# print(df.loc[:,'A':'B'])
print("df.loc['20130102':'20130104',:]")
print(df.loc['20130102':'20130104',:])

print("df.loc['20130102':'20130104',['A','B']]")
print(df.loc['20130102':'20130104',['A','B']])

print("df.loc['20130101','A']")
print(df.loc['20130101','A'])
print("df.loc[dates[0],'A']")
print(df.loc[dates[0],'A'])
#at is fater than loc 
df.at[dates[0],'A']
#Select via the position of the passed integers
print('#Select via the position of the passed integers:')
print("df")
print(df)
print("df.iloc[0]")
print(df.iloc[0]) #row 0
print("df.iloc[3:5,0:2]")
print(df.iloc[3:5,0:2])
print("df.iloc[1,1]")
print(df.iloc[1,1])

# Boolean Indexing
print("Boolean Indexing")
print("df[df.A > 0]")
print(df[df.A > 0])
print("df[df > 0]")
print(df[df > 0])

df2 = df.copy() #deep copy
df2['E'] = ['one', 'one','two','three','four','three']
print("df2")
print(df2)
print("df2[df2['E'].isin(['two','four'])]")
print(df2[df2['E'].isin(['two','four'])])


# Setting a new column automatically aligns the data by the indexes.
# 新規カラム追加

s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
print("s1")
print(s1)

#add series to existing dataframe as a new column
df['F'] = s1
print("df")
print(df)

df.at[dates[0],'A'] = 0 #same with df.iat[0,0] = 0
df.iat[0,1] = 0 #same with df.at[dates[0],'B'] = 0
df.loc[:,'D'] = np.array([5] * len(df))
print("df")
print(df)

# newarr=np.array([5] * 3)
# print(newarr)

df2 = df.copy()
df2[df2 < 0] = -df2
print("df2")
print(df2)

# Missing Data
print("df")
print(df)

#df1 and df2 are the same
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
print("df1")
print(df1)

df2 = df.reindex(index=pd.to_datetime(['2013-01-01','2013-01-02','2013-01-03','2013-01-04']), columns=list(df.columns) + ['E'])
print("df2")
print(df2)


df3 = df.reindex(pd.to_datetime(['2013-01-03','2013-01-01','2013-01-02','2023-01-02'])) #型が同じであれば、任意の値をインデックスに変更できる
print("df3")
print(df3)

df1.loc[dates[0]:dates[1],'E'] = 1
print("df1")
print(df1)
# .dropna()での欠損値を含む行または列を、行/列ごと消去
print("df1.dropna(how='any')")
print(df1.dropna(how='any'))

# Filling missing data.
print("df1.fillna(value=5)")
print(df1.fillna(value=5))
# To get the boolean mask where values are nan.
print("pd.isna(df1)")
print(pd.isna(df1))

print("Stats")
df4 = pd.DataFrame(np.arange(0,24).reshape(6,4), index=dates, columns=list('ABCD'))
print("df4")
print(df4)
print("df4.mean()")#argument 0...index 1..columns
print(df4.mean())
print("df4.mean(1)")
print(df4.mean(1))

s = pd.Series([1,3,5,np.nan,6,8], index=dates)
print("before shift 2")
print(s)
s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2) #slide down in column
print("after shift 2")
print(s)

print("df4")
print(df4)
print("df4.sub(s, axis='index')")
print(df4.sub(s, axis='index')) # df4 - s 

# Applying functions to the data:
print("Applying functions to the data:")
print("df4")
print(df4)
df5 = df4.apply(lambda x: x * 2 )
print('df5 = df4.apply(lambda x: x * 2)')
print(df5)
print(" df5.apply(lambda x: x.max() - x.min() )")
print( df5.apply(lambda x: x.max() - x.min() ))

# Concat¶
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                   index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                   index=[4, 5, 6, 7])
print(df1)
print("df2")
print(df2)
print("pd.concat([df1, df2])")
print(pd.concat([df1, df2]))

left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})

print("Join")
left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['bar', 'foo'], 'rval': [4, 5]})
print("left")
print(left)
print("right")
print(right)
print("pd.merge(left, right, on='key')")
print(pd.merge(left, right, on='key'))

# Append
print("df4")
print(df4)
s = df4.iloc[3]
print("s= df4.iloc[3]")
print(s)
print("df.append(s, ignore_index=True)")
print(df4.append(s, ignore_index=True))

# Grouping
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
           'foo', 'bar', 'foo', 'foo'],
    'B' : ['one', 'one', 'two', 'three',
           'two', 'two', 'one', 'three'],
    'C' : np.arange(0,8),
    'D' : np.arange(8,16)})

print("df")
print(df)
print("df.groupby('A').sum()")
print(df.groupby('A').sum())
print("df.groupby(['A','B']).sum()")
print(df.groupby(['A','B']).sum())

print("Reshaping")
# Series で使えるのは unstack のみです。また、インデックスが階層化されている場合にかぎり有効です。
ss = pd.Series([1,2,3,4,5], index=[['AA','AA','AA','BB','BB'],
                                ['test1','test2','test3','test1','test3']]) 
print("ss")
print(ss)
print("ss.unstack()")
print(ss.unstack())

df=pd.DataFrame([[1,2,3],[4,np.nan,5]],
             columns=['test1','test2','test3'],
             index=['AA','BB'])
print("df=")
print(df)
print("df.stack()=")  
print(df.stack())      

#Pivot Tables
df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
             'B' : ['p', 'q', 'r'] * 4,
             'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
             'D' : np.arange(12),
             'E' : np.arange(12)})
print("df=")
print(df)             
print("pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])")
print(pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C']))
