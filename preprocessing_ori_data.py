import pandas as pd
from kmodes.kmodes import KModes
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score

df_ori = pd.read_excel("C:/Users/82104/Desktop/fproject/V10 Survey Passengers raw data_2101_edit0804_Kihun Kim.xlsx",
                       header=1)

def preprocessing(df_ori):
    # timing 문자 포함 칼럼 삭제
    timing_list = list(df_ori.loc[0])
    no_list = [i for i in range(len(timing_list)) if "Timing" in timing_list[i]]
    all_list = [i for i in range(len(timing_list))]
    df = df_ori.iloc[:, list(set(all_list) - set(no_list))]
    # len([i for i in range(len(df.columns)) if "Q" in df.columns[i]]) # 질문 갯수 97개

    # 응답 0개인 사람 삭제 (95명)
    idx = df[df['Progress'] == 1].index
    df = df.drop(idx)

    # 필요없는 칼럼 삭제 -> 이번 분석에서 필요없는 문항
    df = df.drop(['StartDate', 'EndDate', 'Duration (in seconds)', 'RecordedDate', 'LocationLatitude', 'LocationLongitude',
                  'DistributionChannel', 'UserLanguage', 'CH', 'Q6', 'Q7', 'Progress', 'Finished', 'ResponseId'], axis=1)
    df = df.drop([0])

    # 필요없는 칼럼 삭제 -> 주관식 문항
    df = df.drop(['Q3_6_TEXT', 'Q4_6_TEXT', 'Q5_6_TEXT', 'Q5.2_6_TEXT', 'Q1_7_TEXT', 'Q2_7_TEXT',
                  'Q4_7_TEXT', 'Q1_8_TEXT'], axis=1)

    # 칼럼별 결측값 갯수 확인
    # pd.set_option('display.max_rows', None) # 생략없이 모두 출력해서 볼 수 있도록 설정
    # pd.reset_option("display.max_rows")
    df_null_check = pd.DataFrame(df.isnull().sum())

    # df = df_ori.drop([0]) # 임시코드
    # 순서형 데이터 수치화
    df = df.replace(
        {'Not at all Confident': 1, 'Not Very Confident': 2, 'Neutral': 3, 'Somewhat Confident': 4, 'Very Confident': 5})
    df = df.replace(
        {'Not at all Concerned': 1, 'Fairly Unconcerned': 2, 'Neutral': 3, 'Somewhat Concerned': 4, 'Very Concerned': 5})
    df = df.replace(
        {'Not at all Important': 1, 'Not very Important': 2, 'Neutral': 3, 'Somewhat Important': 4, 'Very Important': 5})
    df = df.replace({'Very Dissatisfied': 1, 'Not Satisfied': 2, 'Neutral': 3, 'Satisfied': 4,
                     'Very Satisfied': 5, 'Very satisfied':5})
    df = df.replace(
        {'Very Unpleasant': 1, 'Somewhat Unpleasant': 2, 'Neutral': 3, 'Somewhat Pleasant': 4, 'Very Pleasant': 5})
    df = df.replace({'Unaffected': 1, 'Relatively unaffected': 2, 'Neutral': 3, 'Somewhat affected': 4, 'Very Affected': 5})
    df = df.replace({'No health measures/screenings': 1, 'Less than 5 minutes': 2, 'Between 5 to 15 minutes': 3,
                     'Between 15 to 30 minutes': 4, 'More than 30 minutes': 5, })
    df = df.replace(
        {'Much too little': 1, 'Somewhat too little': 1, 'Appropriate': 1, 'Somewhat too much': 2, 'Far too much': 2})

    # 명목형 데이터 수치화 # 순서형 데이터 정도에 따라 수치가 부여되어야한다면 사용 불가. 아니면 이걸로 한번에 인코딩 하면 됨.
    cat_col_names = ['Q1', 'Q3', 'Q8', 'Q1.1', 'Q2.1', 'Q3.2', 'Q4.2', 'Q5.1', 'Q5.2', 'Q4.3', 'Q4.1', 'Q1.2', 'Q2.2',
                     'Q4.4', 'Q1.3', 'Q1.4', 'Q1.5', 'Q4.5', 'Q4.1.1', 'Q1.6', 'Q3.3', 'Q3.1']

    for col_name in cat_col_names:
        encoder = LabelEncoder()
        df[col_name] = encoder.fit_transform(df[col_name])

    return df

df = preprocessing(df_ori)
# 문항 분리
df1 = df[['Q2_1.5', 'Q3_1.4', 'Q4_1.3', 'Q5_1.3', 'Q5_2.2']] # 수하물 관련 문항
df2 = df[['Q9_1', 'Q9_2', 'Q9_3', 'Q9_4', 'Q9_5']] # how confident 관련 문항

# nan 값 처리
df1.isnull().sum()
df1_drop = df1.dropna(axis=0) # 확인 241-218 = 23
df2_drop = df2.dropna(axis=0) # 확인 241-0 = 241