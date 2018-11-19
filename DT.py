#!/usr/bin/env python
# coding: utf-8
import sklearn.datasets as datasets
import pandas as pd

data=pd.read_excel('Rule.xls')
data.head()

data.columns

att = pd.DataFrame(data, columns=['Transportation expense','Distance from Residence to Work',
       'Age', 'Education', 'Son', 'Social drinker', 'Social smoker', 'Pet',
       'Weight', 'Absenteeism time in hours'])

target = pd.DataFrame(data, columns=['Absent'])
#target

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
#dtree.fit(att,target)
model = dtree.fit(att, target)

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

TF = ['FALSE','TRUE']

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=False,
                special_characters=True,feature_names = att.columns, class_names = TF)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())

graph.write_pdf("Decision_Tree.pdf")
graph.write_png("Decision_Tree.png")
