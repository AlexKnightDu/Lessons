import pandas as pd
import matplotlib.pyplot as plt



# Process the data
category = ['age', 'work', 'weight', 'education_degree', 'education_time', 'marriage', 'job', 'family', 'race', 'sex', 'capital_gain', 'capital_loss', 'week_work_hours', 'country', 'income']
data = pd.read_csv("./adult.txt", names=category)


education_degrees = [' Preschool',' 1st-4th',' 5th-6th',' 7th-8th',' 9th',' 10th',' 11th',' 12th',' HS-grad',' Assoc-voc',' Prof-school',' Assoc-acdm',' Some-college',' Bachelors',' Masters',' Doctorate']
countrys = ' United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands'
countrys = countrys.split(',')
jobs = ' Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces'
jobs = jobs.split(',')
marriages = ' Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse'
marriages = marriages.split(',')
works = ' Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked'
works = works.split(',')
races  = ' White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black'
races = races.split(',')

###############################################
#
#
# age_fig = plt.figure('Ages')
# ages = []
# max_age = max(data['age'])
# for x in range(0,max_age+1):
#     ages += [len(data[data['age'] == x])]
#
# plt.bar(list(range(0,max_age+1)),ages)
# plt.ylabel('Num')
# plt.xlabel('Ages')
#
#
# ###############################################
#
#
# work_fig = plt.figure('Work')
# work_num = []
# for x in range(0,len(works)):
#     work_num += [len(data[data['work'] == works[x]])]
#
# plt.bar(list(range(0,len(works))),work_num)
# plt.ylabel('Num')
# plt.xlabel('Work')
# plt.xticks(list(range(len(works))), works, rotation=90)
#
#
# ###############################################
#
#
# edu_fig = plt.figure('Education')
# edu_num = []
# colors = []
# sum_all = len(data)
# for x in range(0,len(education_degrees)):
#     edu_num += [len(data[data['education_degree'] == education_degrees[x]])]
#     colors += [[x / len(education_degrees), 0.2, 1 - x / len(education_degrees)]]
# plt.pie(edu_num,colors=colors,labels=education_degrees)
#
#
# ###############################################
#
#
# job_fig = plt.figure('Jobs')
# job_num = []
# colors = []
# sum_all = len(data)
# for x in range(0,len(jobs)):
#     job_num += [len(data[data['job'] == jobs[x]])]
#     colors += [[x / len(jobs), 0.5, 1 - x / len(jobs)]]
# plt.pie(job_num,colors=colors,labels=jobs)
#
#
# ###############################################
#
#
# edu_time_fig = plt.figure('Education time')
# edu_time_num = []
# colors = []
# # sum_all = len(data)
# max_time = max(data['education_time'])
# for x in range(0,max_time+1):
#     edu_time_num += [len(data[data['education_time'] == x])]
#     # colors += [[x / len(education_degrees), 0.2, 1 - x / len(education_degrees)]]
# # plt.pie(edu_num,colors=colors,labels=education_degrees)
# plt.plot(list(range(0,max_time+1)), edu_time_num)
# plt.ylabel('Num')
# plt.xlabel('Education time')
#
#
# ###############################################
#
#
# race_fig = plt.figure('Race')
# race_num = []
# colors = []
# sum_all = len(data)
# for x in range(0,len(races)):
#     race_num += [len(data[data['race'] == races[x]])]
#     colors += [[x / len(races), 0.5, 1 - x / len(races)]]
# plt.pie(race_num,colors=colors,labels=races)
#

###############################################


cap_fig = plt.figure('Capital')
gain_num = []
loss_num = []
# colors = []
sum_all = len(data)
max_gain = max(data['capital_gain'])
max_loss = max(data['capital_loss'])
# max_time = max(data['education_time'])
for x in range(0,max_gain+1):
    gain_num += [len(data[data['capital_gain'] == x])]
for x in range(0,max_loss+1):
    loss_num += [len(data[data['capital_loss'] == x])]
plt.plot(list(range(0,max([max_gain,max_loss])+1)), gain_num)
# plt.plot(list(range(0,max([max_gain,max_loss])+1)), loss_num)
plt.ylabel('Num')
plt.xlabel('Capital')


###############################################
#
# edu_ratio = {}
# edu_num = {}
# for education_degree in education_degrees:
#     edu_ratio[education_degree] = []
#     edu_num[education_degree] = []
#     for country in countrys:
#         country_data = data[data['country'] == country]
#         sum_all_weights = sum(country_data['weight'])
#         country_edu_data = country_data[country_data['education_degree'] == education_degree]
#         sum_edu_weights = sum(country_edu_data['weight'])
#         if sum_all_weights == 0:
#             ratio = 0
#         else:
#             ratio = sum_edu_weights * 1.0 / sum_all_weights
#         edu_ratio[education_degree] += [ratio]
#         edu_num[education_degree] += [sum_edu_weights]
#
#
# country_edu_fig = plt.figure("Country - Education")
# plt.bar(list(range(len(countrys))),edu_ratio[education_degrees[0]],color=[0,0,1],label=education_degrees[0])
# bottom = edu_ratio[education_degrees[0]]
# for i in range(1,len(education_degrees)):
#     plt.bar(list(range(len(countrys))),edu_ratio[education_degrees[i]],bottom=bottom,color=[i/15.0,0.0,1-i/15.0],label=education_degrees[i])
#     for j in range(len(bottom)):
#         bottom[j] += edu_ratio[education_degrees[i]][j]
# plt.xticks(list(range(len(countrys))), countrys, rotation=90)
# plt.legend(loc='upper right')
# plt.xlim([-1,len(countrys)+8])
#
#
# country_population_edu_fig = plt.figure("Country weight - Education")
# plt.bar(list(range(len(countrys))),edu_num[education_degrees[0]],color=[0,0.0,1],label=education_degrees[0])
# bottom = edu_num[education_degrees[0]]
# for i in range(1,len(education_degrees)):
#     plt.bar(list(range(len(countrys))),edu_num[education_degrees[i]],bottom=bottom,color=[i/15.0,0.0,1-i/15.0],label=education_degrees[i])
#     for j in range(len(bottom)):
#         bottom[j] += edu_num[education_degrees[i]][j]
# plt.xticks(list(range(len(countrys))), countrys, rotation=90)
# plt.legend(loc='upper right')
# plt.xlim([-1,len(countrys)])
#
#
# ###############################################
#
#
# edu_ratio = {}
# edu_num = {}
# for education_degree in education_degrees:
#     edu_ratio[education_degree] = []
#     edu_num[education_degree] = []
#     for job in jobs:
#         job_data = data[data['job'] == job]
#         sum_all_weights = sum(job_data['weight'])
#         job_edu_data = job_data[job_data['education_degree'] == education_degree]
#         sum_edu_weights = sum(job_edu_data['weight'])
#         if sum_all_weights == 0:
#             ratio = 0
#         else:
#             ratio = sum_edu_weights * 1.0 / sum_all_weights
#         edu_ratio[education_degree] += [ratio]
#         edu_num[education_degree] += [sum_edu_weights]
#
#
# jobs_edu_fig = plt.figure("Jobs - Education")
# plt.bar(list(range(len(jobs))),edu_ratio[education_degrees[0]],color=[0,1,0],label=education_degrees[0])
# bottom = edu_ratio[education_degrees[0]]
# for i in range(1,len(education_degrees)):
#     plt.bar(list(range(len(jobs))),edu_ratio[education_degrees[i]],bottom=bottom,color=[i/15.0,1-i/15.0,0],label=education_degrees[i])
#     for j in range(len(bottom)):
#         bottom[j] += edu_ratio[education_degrees[i]][j]
# plt.xticks(list(range(len(jobs))), jobs, rotation=90)
# plt.legend(loc='upper right')
# plt.xlim([-1,len(jobs)+2])
#
#
#
# ###############################################
#
#
# edu_ratio = {}
# edu_num = {}
# for education_degree in education_degrees:
#     edu_ratio[education_degree] = []
#     edu_num[education_degree] = []
#     for marriage in marriages:
#         marriage_data = data[data['marriage'] == marriage]
#         sum_all_weights = sum(marriage_data['weight'])
#         marriage_edu_data = marriage_data[marriage_data['education_degree'] == education_degree]
#         sum_edu_weights = sum(marriage_edu_data['weight'])
#         if sum_all_weights == 0:
#             ratio = 0
#         else:
#             ratio = sum_edu_weights * 1.0 / sum_all_weights
#         edu_ratio[education_degree] += [ratio]
#         edu_num[education_degree] += [sum_edu_weights]
#
#
# marriages_edu_fig = plt.figure("Relationship - Education")
# plt.bar(list(range(len(marriages))),edu_ratio[education_degrees[0]],color=[0,1,0],label=education_degrees[0])
# bottom = edu_ratio[education_degrees[0]]
# for i in range(1,len(education_degrees)):
#     plt.bar(list(range(len(marriages))),edu_ratio[education_degrees[i]],bottom=bottom,color=[i/15.0,1-i/15.0,0],label=education_degrees[i])
#     for j in range(len(bottom)):
#         bottom[j] += edu_ratio[education_degrees[i]][j]
# plt.xticks(list(range(len(marriages))), marriages, rotation=90)
# plt.legend(loc='upper right')
# plt.xlim([-1,len(marriages)+2])
#
#
#
# ###############################################
#
# capital_gain_work_hour_fig = plt.figure('Captial_gain_Week_work_hour')
# for i in range(len(education_degrees)):
#     p = plt.scatter(data[data['education_degree'] == education_degrees[i]]['capital_loss'],data[data['education_degree'] == education_degrees[i]]['week_work_hours'],marker='.',color=[i/15.0,0.0,1-i/15.0])
#     # plt.legend(education_degrees[i])
# plt.legend(education_degrees,loc='upper right')
# plt.ylabel('Work hours per week')
# plt.xlabel('capital loss')
#
#
# ###############################################
#
#
# higher_income_dist_fig = plt.figure('Higher income distribution on education degree')
# higher_income = data[data['income'] == ' >50K.']
# lower_income = data[data['income'] == ' <=50K.']
#
# sizes = []
# colors = []
# sum_higher_income = sum(higher_income['weight'])
# for i in range(len(education_degrees)):
#     sum_edu_higher_income = sum(higher_income[higher_income['education_degree'] == education_degrees[i]]['weight'])
#     sizes += [sum_edu_higher_income]
#     colors += [[i/15.0,0.0,1-i/15.0]]
# plt.pie(sizes,colors=colors,labels=education_degrees)
#
# sizes = []
# colors = []
# lower_income_dist_fig = plt.figure('Lower income distribution on education degree')
# sum_lower_income = sum(lower_income['weight'])
# for i in range(len(education_degrees)):
#     sum_edu_lower_income = sum(lower_income[lower_income['education_degree'] == education_degrees[i]]['weight'])
#     sizes += [sum_edu_lower_income]
#     colors += [[i/15.0,0.0,1-i/15.0]]
# plt.pie(sizes,colors=colors,labels=education_degrees)
#

###############################################

plt.show()