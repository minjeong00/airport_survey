# 함수로
import pandas as pd
from scipy.stats import shapiro
import scipy.stats
from preprocessing_ori_data import preprocessing

# 코로나 전 후 importance 관련 문항 비교
df_ori = pd.read_excel("C:/Users/82104/Desktop/fproject/V10 Survey Passengers raw data_2101_edit0804_Kihun Kim.xlsx",
                       header=1)
# 스피어만 상관계수로 문항간 상관관계를 확인한다.
def correlation(s_Q1, s_Q2):
    df1 = df_ori[[s_Q1, s_Q2]]
    df1 = preprocessing(df1)
    df1 = df1.dropna(axis=0)  # nan 처리
    corr = scipy.stats.spearmanr(df1[[s_Q1]], df1[[s_Q2]])

    return corr  # 스피어만 상관계수 1에 가까우면 양의 상관관계

# 확인
correlation('Q1_2.1', 'Q3_2.1')

## 2번째
# timing 문자 포함 칼럼 삭제
timing_list = list(df_ori.loc[0])
no_list = [i for i in range(len(timing_list)) if "Timing" in timing_list[i]]
all_list = [i for i in range(len(timing_list))]
df = df_ori.iloc[:, list(set(all_list) - set(no_list))]

df1 = df.iloc[:, 19:26]
df2 = df.iloc[:, 37:62]
df3 = df.iloc[:, [64, 65, 68, 86, 91, 92, 93, 94, 100, 103]]
df4 = df.iloc[:, 73:85]
df_concat = pd.concat([df1, df2, df3, df4], axis=1)

df_concat = df_concat.replace(
        {'Not at all Confident': 1, 'Not Very Confident': 2, 'Neutral': 3, 'Somewhat Confident': 4, 'Very Confident': 5})
df_concat = df_concat.replace(
        {'Not at all Concerned': 1, 'Not at all concerned':1, 'Fairly Unconcerned': 2, 'Neutral': 3, 'Somewhat Concerned': 4,
         'Somewhat concerned':4, 'Very Concerned': 5})
df_concat = df_concat.replace(
        {'Not at all Important': 1, 'Not very Important': 2, 'Neutral': 3, 'Somewhat Important': 4,
         'Somewhat important':4, 'Very Important': 5, 'Very important':5})
df_concat = df_concat.replace({'Very Dissatisfied': 1, 'Not Satisfied': 2, 'Neutral': 3, 'Satisfied': 4,
                     'Very Satisfied': 5, 'Very satisfied':5})
df_concat = df_concat.replace(
        {'Very Unpleasant': 1, 'Somewhat Unpleasant': 2, 'Neutral': 3, 'Somewhat Pleasant': 4, 'Very Pleasant': 5})
df_concat = df_concat.replace({'Unaffected': 1, 'Relatively unaffected': 2, 'Neutral': 3, 'Somewhat affected': 4, 'Very Affected': 5})
df_concat = df_concat.replace({'No health measures/screenings': 1, 'Less than 5 minutes': 2, 'Between 5 to 15 minutes': 3,
                     'Between 15 to 30 minutes': 4, 'More than 30 minutes': 5, })
df_concat = df_concat.replace(
        {'Much too little': 1, 'Somewhat too little': 2, 'Appropriate': 3, 'Somewhat too much': 4, 'Far too much': 5})

df_corr = pd.DataFrame(columns=['Q1', 'Q2','correlation_spear', 'pvalue_spear', 'count'])
for i in range(0, len(df_concat.columns)-2):
    for j in range(i+1, len(df_concat.columns)-1):
        df1 = df_concat.iloc[:, [i, j]]
        df1 = df1.dropna(axis=0)
        df1 = df1.drop([0])

        corr_spear = scipy.stats.spearmanr(df1.iloc[:, 0], df1.iloc[:, 1])
        # corr_kendal = scipy.stats.kendalltau(df1.iloc[:, 0], df1.iloc[:, 1])

        df_corr = df_corr.append(pd.DataFrame([[df1.columns[0], df1.columns[1],
                                                corr_spear[0], corr_spear[1], len(df1)]],
                                              columns=['Q1', 'Q2', 'correlation_spear',
                                                                        'pvalue_spear', 'count']), ignore_index=True)

        print(i)

# correlation 0.7보다 큰 문항 분석하기
# df_corr = df_corr[df_corr['Q1'] != df_corr['Q2']] # 동일문항 삭제
df_corr = df_corr[df_corr['count'] > 50] # 문항 100개 이상인것만 가져옴
df_corr_plus = df_corr[df_corr['correlation_spear'] > 0.5] # correlation 0.7보다 작은 문항 삭제
df_corr_plus = df_corr_plus[df_corr_plus['correlation_spear'] < 0.7]
df_corr_minus = df_corr[df_corr['correlation_spear'] < -0.7]