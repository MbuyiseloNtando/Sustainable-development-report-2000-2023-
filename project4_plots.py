import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#dataset
#https://www.kaggle.com/datasets/sazidthe1/sustainable-development-report

#loading up data frames
data=pd.read_csv('sdg_index_2000-2022.csv') 
data2=pd.read_csv('sustainable_development_report_2023.csv')
print(data.info())
print(data2.info())

data2.drop(columns='region', inplace=True) #drolping the regions column in 2023 dataframe

data2.assign(year='2023') # assigning a column with constant value='2003'
data2.insert(2, 'year', '2023')

#converting year columns to integers
data['year']=data.year.astype('int')
data2['year']=data2.year.astype('int')

#renaming the column so that columns containing the data for both datasets have the same namea
data.rename(columns={'sdg_index_score':'overall_score'}, inplace=True)

#concating the two data frames
df = pd.concat([data, data2], ignore_index = True, sort = False)
print(df.info())

#renaming columns for better readability
df.rename(columns={'goal_1_score' : 'no_poverty', #score for goal2:no poverty
                                       'goal_2_score':'no_hunger',#score for goa13: no hunger
                                       'goal_3_score':'health_wellbeing',#The score for Goal3: Good Health and Wellbeing.
                                       'goal_4_score':'quality_edu', #The score for Goal 4: Quality Education.
                                       'goal_5_score':'gender_equal',#The score for Goal 5: Gender Equality
                                        'goal_6_score':'water_sanitation', #The score for Goal 6: Clean Water and Sanitation.
                                       'goal_7_score':'clean_energy', #The score for Goal 7: Affordable and Clean Energy
                                       'goal_8_score':'work_econ_grow',#The score for Goal 8: Decent Work and Economic Growth.
                                        'goal_9_score':'inov_infra', #The score for Goal 9: Industry, Innovation and Infrastructure
                                       'goal_10_score':'reduce_ineq', #The score for Goal 10: Reduced Inequalities.
                                       'goal_11_score':'sust_com_citi', #The score for Goal 11: Sustainable Cities and Communitie
                                        'goal_12_score':'resp_cons_prod', #The score for Goal 12: Responsible Consumption and Production.
                                       'goal_13_score':'clim_action', #The score for Goal 13: Climate Action.
                                       'goal_14_score':'life_water', #The score for Goal 14: Life Below Water.
                                       'goal_15_score':'life_land', #The score for Goal 15: Life on Land.     
                                       'goal_16_score':'peace_just_strong',#The score for Goal 16: Peace, Justice and Strong
                                       'goal_17_score':'patner_goals'#The score for Goal 17: Partnerships for the Goals
}, inplace=True)

print(df.info())
#print(df.value_counts())

grp_2023=data2.groupby('country')['overall_score'].mean()
print(grp_2023)

index_score=df.groupby(['year','country'])['overall_score'].mean() #Overall score group 

index_score=index_score.unstack()
print(index_score)
print(df.isna().sum()) #printing all the null values in each column

print(df.duplicated().sum()) #cheking for duplicated entries

print(df.describe()) #descriptive statistics

#sns.heatmap(df.corr(numeric_only=True))
#plt.show()

#overall score
#plt.figure(figsize = (10,70))
#sns.set_theme(style="whitegrid")
#sns.set_color_codes("pastel")
#sns.barplot(data = data2, x = 'overall_score', y = 'country')
#plt.title('Countries Overall Sustainable Development Score')
#plt.show()

#locaring South AFrican entries
df_sa=df[df.country=='South Africa'] #Dataframe for South Africa

print(df_sa.head(50))

print(df_sa.corr(numeric_only=True))
#sns.heatmap(df_SA.corr(numeric_only=True))
#plt.show()

df_sa_pov=df_sa.groupby('year')[['no_poverty', 'no_hunger', 'health_wellbeing','quality_edu', 'gender_equal','water_sanitation','clean_energy','work_econ_grow','inov_infra', 'peace_just_strong','overall_score']].mean()
print(df_sa_pov)

#df_sa_pov.plot(kind='line')
#plt.title('The average scores by year')
#plt.ylabel('Score')
#plt.show()

#How does the overall scores of South Africa compare in 2023

df_sa_2023=df_sa[df_sa.year==2023]
print(df_sa_2023)
df_sa_2023.drop(columns=['year'], inplace=True)

df_sa_2023.plot(kind='bar')
plt.title('South Africa Sustainable Development report')
plt.xlabel('Goals')
plt.ylabel('Scores')
plt.show()

df_2023=df[df.year==2003]






