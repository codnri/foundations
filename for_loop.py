"""
hoge = [[1,2,3],[4,5,6],[7,8,9]]
for(x,y,z) in hoge:
  print("x="+str(x))
  print("y="+str(y))
  print("z="+str(z))
"""
  # print("y="+y)
  # print("z="+z)

print("Hi {0}, Good {1}".format("Tom","Evening"))
print("Hi {2}, Good {1}. What would like to have for {0}".format("Dinner","Evening","Tom"))


# print("***dictionary***")
# # dictTest = {"oneKey":1, "twoKey":2, "threeKey":3}
# # print("oneKey:"+str(dictTest["oneKey"]))
# # for val in dictTest.keys():
# #   print(val+":::"+str(dictTest[val]))

# mydict = {"apple":1, "orange":2, "banana":3}
# mydict["peach"] = 4
# print(mydict)

# lst = []
# print("list comprehension")
# line= '"12",2012,"01","Suicide",0,"M",21,"Native American/Native Alaskan",100,"Home",2'
# lst = [ int(element) if element.isdigit() else element.strip('"') for element in line.strip().split(',') ]
# print(lst)

print("list comprehension")
fruits = ['apple', 'banana', 'melon', 'apple', 'apple', 'apple']
lst= set(f for f in fruits)
print(lst)
