# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 23:37:39 2020

@author: Masaya Heywood (Capt/Captworgen)

"""

import numpy as np
import scipy.stats as sp
import sklearn.metrics as sk
from scipy.linalg import svd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB as gnb

#old initilizers
#dummyList = [3, 4, 8]
#dummyListAlt = [3, 2, 5]
# dummy2DList = [(0,50), (10,5), (5,10)]
# dummyNumpy = np.array(dummyList)
# dummy2DNumpy = np.array(dummy2DList)
# dummyRand2DNumpy = np.random.rand(100, 2) 


#---------------------------------------------------------------------------------------
## Preparation
schoolDataCSV = pd.read_csv("middleSchoolData.csv", header=1)
schoolDataUnclean = schoolDataCSV.copy() #working copy

#First step is to clean and prepare the data
schoolDataUnclean.columns = ["DBN", "School", "Num Apply", "Num Accept", "Pupil Spending", "Class Size", 
                             "Asian", "Black", "Hispanic", "Inter", "White", "Rigor", "Collab", "Support",
                             "Effective", "Family", "Trust", "Disabled", "Poverty", "ESL", "Size",
                             "Achievement", "Reading", "Math"]

#keeping the unclean version seperate, since some datais dropped in the "schoolData" version
schoolData = schoolDataUnclean.copy()

#removing DBN, PPS and CS due to unnecessary/incomplete data
schoolData.drop(columns = ["DBN", "Pupil Spending", "Class Size"], inplace = True)
schoolData.dropna(axis=0, inplace = True)
schoolData.replace([np.inf, -np.inf], np.nan, inplace = True)
schoolData.fillna(value=0, inplace = True)
schoolData.reset_index(drop=True, inplace = True)

#---------------------------------------------------------------------------------------

##Question 1
#finding correlation
corr = sp.pearsonr(schoolData["Num Apply"], schoolData["Num Accept"])
#the correlation is .8

#---------------------------------------------------------------------------------------

##Question 2
# what is a better predictor of admission, raw number or rate of applications?
#how many people are apply per school size?
rateOfApply = schoolData["Num Apply"] / schoolData["Size"]
rateOfApply.replace([np.inf, -np.inf], np.nan, inplace = True)
rateOfApply.fillna(value=0, inplace = True)
    
#compare the rate of acceptance to the rate of applications
appRateCorr = sp.spearmanr(rateOfApply, schoolData["Num Accept"])
applyCorr = sp.spearmanr(schoolData["Num Apply"], schoolData["Num Accept"])

outputOfAppCorr = np.array([appRateCorr, applyCorr])
#raw number of applicans are have a higher correlation to number of accepted 
#---------------------------------------------------------------------------------------

##Question 3
#best odds of going to HSPHS
#how many people are applying and get in?
schoolData["Rate of Accept"] = schoolData["Num Apply"] / schoolData["Num Accept"]
schoolData.replace([np.inf, -np.inf], np.nan, inplace = True)
schoolData["Rate of Accept"].fillna(value=0, inplace = True)
#THE CHRISTA MCAULIFFE SCHOOL\I.S. 187 comes out as the best school with a 1.22 rate of acceptance => 1:1.22 odds / 81% chance of acceptance
#---------------------------------------------------------------------------------------

##Question 4
#trust vs objective measurements
studentsRatingDataFirst = schoolData[["Rigor", "Collab", "Support"]].copy()
studentsRatingDataSecond = schoolData[["Effective", "Family", "Trust"]].copy().copy()
objectiveRatingData = schoolData[["Achievement", "Reading", "Math"]].copy()

stuRateUF, stuRateSF, stuRateVF = svd(studentsRatingDataFirst)
stuRateUS, stuRateSS, stuRateVS = svd(studentsRatingDataSecond)

stuRateS = stuRateSF + stuRateSS

objRateU, objRateS, objRateV = svd(objectiveRatingData)
outputOfSVDCorr = sp.spearmanr(stuRateS, objRateS)
#this seems to be a perfect correlation no matter which way the SVD is bent

#---------------------------------------------------------------------------------------

"""
# #Question 5
# Call me bold, but schools with a lower proverty percentage seem to enjoy a greater acceptance to HSPHS. Once again, unoriginal, I know.
# Alternate Hypothesis: Schools with a lower poverty rate have more students sumbitted to HSPHS.
# Null Hypothesis: Poverty cannot be confirmed as an impacting factor on admission to HSPHS.
# According to https://www.childrensdefense.org/, 1 in 6 children lived in poverty in the United States. This comes to around a 15% poverty rate among children. 
# Knowing the above, I am ranking the poverty of schools into 5 sections. 
# A school's section is determined by the distance it is from 100 on the poverty observation, measured in increments of 20 (15% might be more precise but 20 makes an easy round to 100).
# The assumption is that the less poverty in a school, the more likey there are wealthy families in that school.
(i.e a school's poverty percentage is 44, making it Average since there may be more middle class families there)
 
"""
from scipy.stats import kruskal
#copy to leave above data alone
schoolsPovHypo = schoolData.copy()

secCalcUpper = 0
secCalcLower = 20

secNames = ["Wealthy", "Rich", "Average", "Poor", "Poverty"]
#schoolsPovHypo[(schoolsPovHypo["Poverty"] >= 0) & (schoolsPovHypo["Poverty"] < secCalcUpper)]

for each in range(0,5):
    schoolsPovHypo.loc[(schoolsPovHypo["Poverty"] >= secCalcUpper) & (schoolsPovHypo["Poverty"] < secCalcLower), "Financial Section(Names)"] = secNames[each]
    schoolsPovHypo.loc[(schoolsPovHypo["Poverty"] >= secCalcUpper) & (schoolsPovHypo["Poverty"] < secCalcLower), "Financial Section(Numerical)"] = each
    secCalcUpper += 20
    secCalcLower += 20

schoolsPovHypo["Odds"] = 1 
ratioTable = schoolsPovHypo[["Odds", "Rate of Accept"]].copy()
#oddsratio, pvalue = sp.fisher_exact()
ratioTable["Sum"] = ratioTable["Odds"] + ratioTable["Rate of Accept"]
schoolsPovHypo["Chance"] = ((ratioTable["Odds"]/ratioTable["Sum"])/(ratioTable["Rate of Accept"]/ratioTable["Sum"])) * 100
schoolsPovHypo.replace([np.inf, -np.inf], np.nan, inplace = True)
schoolsPovHypo.fillna(value=0, inplace = True)
schoolsPovHypo.drop(columns = ["Odds"], inplace = True)

wealth_and_acceptance = schoolsPovHypo[["Financial Section(Names)", "Rate of Accept", "Chance", "Num Accept", "Achievement"]]

#kruskal test 
   # schoolsPovHypo.loc[(schoolsPovHypo["Financial Section(Names)"] == "Wealthy")]
povertyKruskalStat, povertyKruskalP = kruskal(wealth_and_acceptance.loc[(wealth_and_acceptance["Financial Section(Names)"] == "Wealthy"), "Rate of Accept"], 
                                       wealth_and_acceptance.loc[(wealth_and_acceptance["Financial Section(Names)"] == "Rich"), "Rate of Accept"], 
                                       wealth_and_acceptance.loc[(wealth_and_acceptance["Financial Section(Names)"] == "Average"), "Rate of Accept"], 
                                       wealth_and_acceptance.loc[(wealth_and_acceptance["Financial Section(Names)"] == "Poor"), "Rate of Accept"], 
                                       wealth_and_acceptance.loc[(wealth_and_acceptance["Financial Section(Names)"] == "Poverty"), "Rate of Accept"])

#print(povertyKruskalStat, povertyKruskalP)
#.000000000003 p-value? Wow! I think this isn't quite the way to do it though. I'm going to try again.
#I'm comparing inequal sample sizes here, the wealthy category just reaches 10 entries while the poverty sample has a significantly larger amount of entries. This can't be good for anything.

secVariance = np.zeros(5)
for each in range(0,5):
    secVariance[each] = wealth_and_acceptance.loc[(wealth_and_acceptance["Financial Section(Names)"] == secNames[each]), "Rate of Accept"].var()

#print(secVariance)  
#As I thought, the variance is wild: 16.68769794  27.45296984  57.61349108 100.85311038 142.22002373

povertyNormalStat, povertyNormalP = sp.shapiro(wealth_and_acceptance.loc[(wealth_and_acceptance["Financial Section(Names)"] == "Poverty"), "Rate of Accept"])
#print(povertyNormalP)

#I heavily doubt this is a normally distributed graph.
#sns.scatterplot(data=schoolsPovHypo, x="Poverty", y="Chance", hue="Financial Section(Names)")
#sns.displot(wealth_and_acceptance.loc[(wealth_and_acceptance["Financial Section(Names)"] == "Wealthy"), "Chance"])


#They're not normally distributed, but I don't think that the Kruskal test was far off.
#sns.regplot(x="Poverty", y="Chance", data=schoolsPovHypo)


povertyPearson = sp.pearsonr(schoolsPovHypo["Poverty"], schoolsPovHypo["Chance"])

#print(povertyPearson)

#The graphs and the pearsonr help prove the Kruskal test, but the issues are still present with the data. I reject the null hypothesis, but see clear issues with the methods used. 
 
#---------------------------------------------------------------------------------------

##Question 6
#Since I dropped the the class size and spending per pupil when cleaning the data, I decided to base this question around the idea of school size vs poverty

#sizePearson = sp.pearsonr(schoolsPovHypo["Size"], schoolsPovHypo["Achievement"])
#sns.regplot(x="Size", y="Achievement", data=schoolsPovHypo)
# plt.clf()

#sns.scatterplot(data=schoolsPovHypo, x="Size", y="Achievement", hue="Financial Section(Names)")
#plt.clf()
#print(sizePearson)

#School size does have an significant impact either Achievement or Chance

#---------------------------------------------------------------------------------------

##Question 7
#taking the rankings from above, maybe that will show the proportions the best
wealth_and_acceptanceSum = np.zeros(5)

for each in range(0,5):
    wealth_and_acceptanceSum[each] = wealth_and_acceptance.loc[(wealth_and_acceptance["Financial Section(Names)"] == secNames[each]), "Num Accept"].sum()
    
#WAAS = pd.DataFrame(wealth_and_acceptanceSum)  
#WAAS.columns = secNames
#sns.histplot(data=wealth_and_acceptanceSum)
#not a good histplot

#print(wealth_and_acceptanceSum)
#the data looks like it is pretty closely clustered around the poorer schools. Manually looking at the data seems to show that schools with high Asian population have the highest admittance to HSPHS
schoolsPovHypo.loc[(schoolsPovHypo["Asian"] >= 50), "Majority Asian"] = True
schoolsPovHypo.loc[(schoolsPovHypo["Asian"] < 50), "Majority Asian"] = False
mostAsianAccept = schoolsPovHypo.loc[(schoolsPovHypo["Majority Asian"] == True), "Num Accept"].sum()
mostOtherAccept = schoolsPovHypo.loc[(schoolsPovHypo["Majority Asian"] == False), "Num Accept"].sum()

#print(mostAsianAccept, mostOtherAccept)
#Although majority asian schools do pull in a sizeable amount of acceptances, I think I'm chasing the wrong lead now. Going back to wealth.
#Many schools have a high poverty rating. Splitting the data in a way different from before may prove useful.
schoolsPovHypo.loc[(schoolsPovHypo["Poverty"] >= 86), "Exceptional Poverty"] = True
schoolsPovHypo.loc[(schoolsPovHypo["Poverty"] < 86), "Exceptional Poverty"] = False
mostPovertyAccept = schoolsPovHypo.loc[(schoolsPovHypo["Exceptional Poverty"] == True), "Num Accept"].sum()
mostWealthyAccept = schoolsPovHypo.loc[(schoolsPovHypo["Exceptional Poverty"] == False), "Num Accept"].sum()

#print(mostPovertyAccept, mostWealthyAccept)
#There is a clear divide here, exceptional poverty is common but amounts to low acceptance totals to HSPHS

#---------------------------------------------------------------------------------------

##Question 8
#lets see how we can predict acceptance chance from and poverty and achievement

from sklearn import linear_model
X = schoolsPovHypo[["Poverty","Achievement"]]
Y = schoolsPovHypo["Chance"]
 
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)
print("Intercept: \n", regr.intercept_)
print("Coefficients: \n", regr.coef_)

#prediction with sklearn
newSchoolPoverty = 15.3
newSchoolTrust = 4.0
#print ("Predicted Acceptance Chance: ", regr.predict([[newSchoolPoverty,newSchoolTrust]]))
#Looks cool, lets test this

testStudents = pd.DataFrame(np.random.rand(523, 2))
testStudents.columns = ["Poverty", "Achievement"]
testStudents["Poverty"] = np.random.uniform(0,100, testStudents.shape[0])
testStudents["Achievement"] = np.random.uniform(0,5, testStudents.shape[0])
testStudents["Chance"] = np.zeros(523)

for each in range(len(testStudents["Poverty"])):
    testStudents.iloc[each, 2] = regr.predict([[testStudents.iloc[each, 0],testStudents.iloc[each, 1]]])
    
    if (testStudents.iloc[each, 2] < 0):
        testStudents.iloc[each, 2] = 0

#fig, ax = plt.subplots(figsize=a4_dims)
#sns.regplot(data = testStudents, x="Poverty", y="Chance", marker="+")
#sns.regplot(data = schoolsPovHypo, x="Poverty", y="Chance", color="red", marker="+")

##The testStudents data (blue) does a good job simulating the actual chance 


##Results
#This section show the results of questions above (some questions do not need to be printed)
showAnswers = True
if showAnswers == True:  
    print("First Corr:", corr)
    print("Applications Corrs:", outputOfAppCorr)
    print("Ratings Type Corr:", outputOfSVDCorr)
    print("Hypothesis Test:", povertyKruskalStat, povertyKruskalP)
    print("Proportion Test:",mostPovertyAccept, mostWealthyAccept)


