import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import bartlett
import scipy.stats
from airport_survey import preprocessing

# 코로나 전 후 importance 관련 문항 비교
df_ori = pd.read_excel("C:/Users/82104/Desktop/fproject/V10 Survey Passengers raw data_2101_edit0804_Kihun Kim.xlsx",
                       header=1)

# t검정으로 코로나 전후 이용객의 생각에 차이가 생겼는지 확인
def t_test(Q_after, Q_before):
    df1 = df_ori[[Q_after, Q_before]]
    df1 = df1.replace(
        {'Not at all Important': 1, 'Not very Important': 2, 'Neutral': 3, 'Somewhat important': 4,
         'Somewhat Important': 4, 'Very Important': 5, 'Very important': 5})
    df1 = df1.dropna(axis=0)
    df1 = df1.drop([0])

    after = df1[Q_after].tolist()
    before = df1[Q_before].tolist()
    when = ['before', 'after']
    when = [when[j] for j in range(2) for i in range(len(df1))]
    data = pd.DataFrame({'when': when, 'score': before + after})
    data.head(3)

    plt.figure(figsize=(6, 6))
    sns.boxplot(x='when', y='score', data=data)

    # 정규성 검정 #p-value모두 0.05보다 크면 정규성에 문제가 없음.
    normal1 = shapiro(before)
    normal2 = shapiro(after)
    print(normal1, normal2)

    # 등분산성 고려 #p-value가 0.05보다 크면 등분산성이 있다고 할 수 있다.
    print(levene(before, after))
    print(bartlett(before, after))

    # 대응표본 t검정 수행
    print(scipy.stats.ttest_rel(before, after))

# 테스트
t_test('Q2_1.1', 'Q2_2')
t_test('Q3_1', 'Q3_2')