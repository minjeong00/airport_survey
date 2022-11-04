import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

df_ori = pd.read_excel("C:/Users/82104/Desktop/fproject/V10 Survey Passengers raw data_2101_edit0804_Kihun Kim.xlsx",
                       header=1)

# 응답 0개인 사람 삭제 (95명)
idx = df_ori[df_ori['Progress'] == 1].index
df = df_ori.drop(idx)

df = df.drop([0])
#################
# 데이터 시각화
x = np.arange(5)
degree = ['Not at all Confident', '2', '3', '4', 'Very Confident']
values = [len(df[df['Q9_5']=='Not at all Confident']), len(df[df['Q9_5']=='Not Very Confident']),
          len(df[df['Q9_5']=='Neutral']), len(df[df['Q9_5']=='Somewhat Confident']), len(df[df['Q9_5']=='Very Confident'])]

plt.bar(x, values)
plt.xticks(x, degree)
plt.title("Use of contact tracing apps")

plt.show()

df1 = df[df['Q10_1']=='Very Concerned']
df2 = df[df['Q10_1']=='Somewhat Concerned']
df3 = pd.concat([df1, df2])
#############
# 카이제곱검정
# 가설1
df1 = df.drop(df[df['Q2'] == 'Prefer not to say'].index)
df1 = df1.replace({'55-64':'55 and above', '65-74':'55 and above', '75 and above':'55 and above'}) # 나이
df1 = df1.replace({'Emigration': 'Other', 'Study Trip': 'Other', 'Repatriation': 'Other'}) # 여행목적

result = pd.crosstab(df1['Q2'], df1['Q2.1'])
stats.chi2_contingency(observed=result)

# 가설2
df2 = df.drop(df[df['Q3'] == 'Other'].index)
df2 = df2[['Q3', 'Q1.2']]
df2 = df2.dropna(axis=0)
result2 = pd.crosstab(df2['Q3'], df2['Q1.2'])
stats.chi2_contingency(observed=result2)