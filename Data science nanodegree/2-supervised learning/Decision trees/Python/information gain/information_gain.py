import pandas as pd
from math import log2
data = pd.read_csv('ml-bugs.csv',delimiter = ',')

print(data)

Parent = data['Color']
Child = data['Species','Length']
n = len(Parent[Parent =='Brown'])
m = len(Parent[Parent == 'Blue'])
o = len(Parent[Parent == 'Green'])
S_parent = -n/(n+m+o)*log2(n/(n+m+o))-m/(n+m)*log2(m/(n+m+o)) - o/(n+m)*log2(o/(n+m+o))
S_child =


def two_group_ent(first, tot):
    return -(first/tot*np.log2(first/tot) +
             (tot-first)/tot*np.log2((tot-first)/tot))

tot_ent = two_group_ent(10, 24)
g17_ent = 15/24 * two_group_ent(11,15) +
           9/24 * two_group_ent(6,9)

answer = tot_ent - g17_ent