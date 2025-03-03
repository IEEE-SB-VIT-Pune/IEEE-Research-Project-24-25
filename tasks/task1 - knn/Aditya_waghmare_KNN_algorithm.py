import random

#matrix creation

m=int(input("Enter no. of rows: "))
n=int(input("Enter no. of columns: "))

def create_matrix(row,column):
    matrix=[]
    for i in range(row):
        row=[]
        for j in range(column):
            x=random.randint(1,20)
            row.append(x)
        matrix.append(row)
    return matrix
    
def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))

        
        
chart=create_matrix(m,n)
print_matrix(chart)
dict={}
const_row=int(input("Enter the row index: "))

for i,v in enumerate(chart):
    if i==const_row:
        continue
    else:
        dist=0
        n=0
        for x in v:
            val=abs(x-chart[const_row][n])
            n+=1
            dist+=val
        dict[dist]=i
sorted_dist=sorted(dict.keys())
l=0
print("Top three rows having minimum distance:")
while l<3:
    idx=dict[sorted_dist[l]]
    l+=1
    print(f"Index:{idx},Row:{chart[idx]},Distance:{sorted_dist[l]}")
    
