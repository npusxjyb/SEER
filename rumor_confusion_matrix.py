
matrix = [[807,80],[55,756]]



TP = matrix[1][1]
FP = matrix[1][0]
FN = matrix[0][1]
TN = matrix[0][0]

#真实新闻
Precision_1=TP/(TP+FP)
Recall_1=TP/(TP+FN)
F1_score_1=2*Precision_1*Recall_1/(Precision_1+Recall_1)
print("Precision_fake:"+str(Precision_1))
print("Recall_fake:"+str(Recall_1))
print("F1_score_fake:"+str(F1_score_1))

#假新闻
Precision_0=TN/(TN+FN)
Recall_0=TN/(TN+FP)
F1_score_0=2*Precision_0*Recall_0/(Precision_0+Recall_0)
print("Precision_real:"+str(Precision_0))
print("Recall_real:"+str(Recall_0))
print("F1_score_real:"+str(F1_score_0))

#总体精确率
Accuracy=(TP+TN)/(TP+FP+FN+TN)
print("Accuracy:"+str(Accuracy))