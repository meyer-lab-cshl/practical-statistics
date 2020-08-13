#Python based statistics
import math
#Contingency table examples

#Example 1
#https://www.statsmodels.org/stable/contingency_tables.html
import numpy as np
import pandas as pd
import statsmodels.api as sm

#Call data file
df = sm.datasets.get_rdataset("dataset1", "dataset2").data
tab = pd.crosstab(df['rowtitle'], df['columntitle'])
tab = tab.loc[:, ["column1", "column2", "column3"]]
table = sm.stats.Table(tab)               #prints counts
print(table)

#Table will print out similar to below
#columntitle:  column1  column2  column3
#rowtitle:       
#dataset1         0        0        0 
#dataset2         0        0        0

#Example 2
#https://www.tutorialspoint.com/contingency-table-in-python
datainput = pd.read_csv("filename.csv")
#To print counts, mean, std, min, 25%, 50%, 75%, max
stats = (datainput.describe())
print('stats =', stats)
#any series passed will have their name attributes used unless row or column names for the cross-tabulation are specified.
width_species = pd.crosstab(datainput['dataset1'],datainput['dataset2'],margins = False)

#Example 3
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.crosstab.html
a = np.array(["foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar", "foo", "foo", "foo"], dtype=object)
b = np.array(["one", "one", "one", "two", "one", "one", "one", "two", "two", "two", "one"], dtype=object)
c = np.array(["dull", "dull", "shiny", "dull", "dull", "shiny", "shiny", "dull", "shiny", "shiny", "shiny"], dtype=object)
table = pd.crosstab(a, [b, c], rownames=['a'], colnames=['b', 'c'])   #This version prints table with tiles and uses integers only
tab = sm.stats.Table(table)                                           #This version prints table with no titles but uses floats
print('table stats =', tab)
print('table =', table)

#Example 4
#If you already have your counts and do not need labels
table = np.asarray([[35, 21], [25, 58]])
#prints out odds ratio, log odds ratio, risk ratio, log risk ratio
t22 = sm.stats.Table2x2(table)
print(t22.summary())




#Standard deviation of sample, mean, variance

import statistics 
N = 9
sample = [1,2,3,4,5,6,7,8,9]
mean = statistics.mean(sample)
sd = statistics.stdev(sample)
var = statistics.variance(sample)
print('mean =',mean)
print('stardard deviation =', sd)
print('variance =',var)



#Sensitivity, specificity, prevalence, PPV, likelihood ratio, prior odds, and posterior odds

#Example and values from class
#Number of individuals in the sample that have the disease
nd = 100
#Number of individuals in the sample that do not have the disease
nnd = 999000
#Total number of individuals in the sample
tot = 1000000
#Contingency table
tp = 82               #True positive: disease+, test+: sensitivity measurement   
fp = 36996            #False positive: disease-, test+: Type 1 error
fn = 18               #False negative: disease+, test-: Type 2 error
tn = 962904           #True negative: disease-, test-: specificity measurement

#sensitivity
#P[test+|disease+] 
sen = (tp/(tp+fn))*100       #the probability of testing positive given that the subject has the disease

#P[test-|disease+] 
senalt = (fn/(fn+tp))*100    #the probability of testing negative given that the subject has the disease

#specificity
#P[test-|disease-]
spe = (tn/(tn+fp))*100       #the probability of a negative test given that the subject does not have the disease

#P[test+|disease-]
spealt = (fp/(fp+tn))*100   #the probability of a positive test given that the subject does not have the disease

#Prevalence
#P[disease+]
pre = (nd/tot)*100           #the fraction of individuals in a population who have a disease; the probability of having such disease

#P[disease-]
prealt = (nnd/tot)*100

#Positive predictive value (PPV):
#P[disease+|test+]
ppv = (tp/(tp+fp))*100             #the probability of actually having the disease if you test positive; this is what a positive case truly cares about

#Likelihood ratio
#P[test+|disease+]/P[test+|disease-] = sensitivity/1-sensitivity
lr = (sen/100)/(1-(spe/100))               #property of test

#Prior odds
#Bayes's theorem
#P[disease+]/P[disease-] = prevalence/1-prevalence
#Quantify uncertainty
po = (pre/(1-pre))/100                #property of population; not a property of an ind per se but rather one's state of knowledge about that ind

#Posterior odds
##P[disease+|test+]/P[disease-|test+] = PPV/1-PPV
#Quantify uncertainty
poso = (lr * po)*100                     #what you care about


#PPV << sensitivity in this case and it is bc FP >> FN bc undiseased inds >> diseased inds
#Prevalence: if a subject's sibling has a disease, there is a 50% chance they do too; otherwise there would be 0.01% chance bt you and a random ind
#PPV: just knowing that your sibling has a disease increases the PPV of the test enormously (from 0.22% to 95.7%)

print('Sensitivity =', sen)
print('P[test-|disease+] =', senalt)
print('Specificity =', spe)
print('P[test+|disease-] =', spealt)
print('Prevalence =', pre)
print('P[disease-] =', prealt)
print('Positive predictive value (PPV) =', ppv)
print('Likelihood ratio =', lr)
print('Prior odds =', po)
print('Posterior odds =', poso)




#Fisher's exact test

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html
#Valid for all sample sizes
#Is there a statistical dependence bt the row an observation falls in and the column that observation falls in?
import scipy.stats as stats
oddsratio, pvalue = stats.fisher_exact([[8, 2], [1, 5]],alternative = 'two-sided')
print('oddsratio =', oddsratio)
print('pvalue =', pvalue)




#Standard deviation of the sampling distribution, z statistic

#https://www.datajourneyman.com/2015/04/13/hypothesis-testing.html
#A neurologist is testing the effect of a drug on response time by injecting 100 rats with a unit dose of the drug, subjecting each to neurological stimulus, and recording its response time.
#The neurologist knows that the mean response time for rats not injected with the drug is 1.2 seconds.
#The mean of the 100 infected rats' response time is 1.05 seconds with a sample standard deviation of 0.5 seconds. 
#Has the drug effected response time?

#null hypothesis: drug has no effect; mean stays the same (1.2 seconds) even with the drug
#alternate hypothesis: drug has an effect; mean is not the same with the drug

#If we assume the null hypothesis is true, so there is no effect, the mean of our sampling distribution will be the same as the population distribution

#Sample size
N = 100        
#Mean
mu1 = 1.2       #mean of sampling distribution
mu2 = 1.05      #mean of the population
#Sample standard deviation
ssd = 0.5
#To calculate this manually from your data:
#ssd = sum(sqrt((x - mu)/(n-1)))     #x = ind value; mu = mean of the sample; n = sample size
#To caluclate using function:
import statistics
ssd = statistics.stdev(sampledata)


#Standard deviation of the sampling distribution
sdp/math.sqrt(N)               #should be equal to the standard deviation of the population (sdp) divided by the sqroot of the sample size
sds = ssd/math.sqrt(N)         #equivalent to sdp/(N**2) 

#How many standard deviations away from this mean is 1.05, and what is the P[getting a result at least that many standard deviations away from the mean]?

#Z statistic
#Normal dist: -3<=z<=3
#How far are we away from the mean?
z = abs((mu2 - mu1)/sds)           #the numerator tells you how far we are and the denominator puts it in terms of standard deviations
print('St dev of the sampling dist =', sds)
print('Z stat =', z)

#mu2 is z standard deviations from the mean
#What is the P[of getting a result that is more or less than z standard deviations away (more extreme than mu2 seconds)]? 
#Empirical rule: 99.7% of the probability is within z standard deviations and outside of this area (0.3%); these are assumptions of a normal distribution
#The P[of getting a sample this extreme or more extreme] = 0.3% 
#Reject null hypothesis: if the null hypothesis were true, only 1 in 300 chance of getting more or less than z standard deviations away 

#pv = P[of getting a result that is more or less extreme than mu2|null hypothesis)] = 0.003
#When pvalues claim significance it means detectable.
#pvalues do not describe the magnitude of the effect or how important it is.
#pvalues do not actually quantify how likely or unlikely your null hypothesis is
#pvalues quanity fow likely you data would be if the null hypothesis were true (P[data|null hypo]), not the other way around
    #For this to make sense, we must accept the base rate fallacy (P[data|null hypo])~(P[null hypo|data])
    #Whether or not this is true depends on the prior odds.




#Confidence intervals

#Example1
import numpy as np
#Values used are from the class
#Confidence intervals are more infomative than pvalues
#CI communicates detectability like pvalue but also results in effect size and uncertainty of that effect size
#95% CI does not mean the true value of the parameter lies within that interval 
pp = 0.26          #population proportion or mean
z = 1.96           #z score
obs = 97           #number of observations
#Standard error
se = np.sqrt(pp * (1-pp)/obs)
#Confidence intervals
cil = pp - z * se
cir = pp + z * se
print('standard error =', se)
print('Left CI =', cil)
print('Right CI =', cir)

#Example2 --> using a t distribution
#https://www.statology.org/confidence-intervals-python/
import numpy as np
import scipy.stats as st
#Useful for small sample sizes
data = [12, 12, 13, 13, 15, 16, 17, 22, 23, 25, 26, 27, 28, 28, 29]
#create 95% confidence interval for population mean weight
stat = st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
print('Left/Right CI =', stat)

#Example3 --> using a normal distribution
import numpy as np
import scipy.stats as st
np.random.seed(0)
data = np.random.randint(10, 30, 50)
#create 95% confidence interval for population mean weight
stat = st.norm.interval(alpha=0.95, loc=np.mean(data), scale=st.sem(data))
print('Left/Right CI =', stat)



#T-test

#T-test from scratch see: https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/

#Example1
#Two-sample T-test using a poisson distribution
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
import numpy as np
from scipy import stats
#Tells us whether two data samples have different means.
np.random.seed(12)
dataset1 = stats.poisson.rvs(loc=0.5,mu=1.2,size=100)
dataset2 = stats.poisson.rvs(loc=0.5,mu=1.05,size=100)
pvalue multiply by 100 to get the percent chance that the sample data is such far apart for two identical groups
#statistic tells us how aberrant the sample mean is from the null hypothesis
stat = stats.ttest_ind(a=dataset1,b=dataset2,equal_var=False)
print('ttest/pvalue =', stat)

#Example2
#One-sample T-test using a normal distribution
#https://www.tutorialspoint.com/python_data_science/python_p_value.htm
from scipy import stats
rvs = stats.norm.rvs(loc = 5, scale = 10, size = (50,2))
s = stats.ttest_1samp(rvs,5.0)
#Prints ttest array and pvalue array
print('ttest/pvalue =', s)

#Example3
#Two-sample T-test using a normal distribution
#https://www.tutorialspoint.com/python_data_science/python_p_value.htm
from scipy import stats
rvs1 = stats.norm.rvs(loc = 5,scale = 10,size = 500)
rvs2 = stats.norm.rvs(loc = 5,scale = 10,size = 500)
s = stats.ttest_ind(rvs1,rvs2)
print('ttest/pvalue =', s)

#Example4
#Paired two-sample T-test using a normal distribution
import numpy as np
from scipy import stats
import pandas as pd
#When you want to check how different samples from the same group are
np.random.seed(12)
before = stats.norm.rvs(scale=1,loc=1.2,size=100)
after = before+stats.norm.rvs(scale=1,loc=1.05,size=100)
#Creates table
weight_df = pd.DataFrame({"weight_before":before, "weight_after":after, "weight_change":after-before})
#Prints counts, mean, std, min...
print(weight_df.describe())
#pvalue should be multiplied by 100 and this is the chance in finding such a huge difference between samples
print(stats.ttest_rel(a=before,b=after))




#Distributions

#Bernoulli distribution

#Example1
#https://www.tutorialspoint.com/python_data_science/python_bernoulli_distribution.htm
#Describes probabilities for a binary variable
#Coin flip
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import seaborn as sb
data_bern = bernoulli.rvs(size=1000,p=0.3)
ax = sb.distplot(data_bern,kde=True,color='crimson',hist_kws={"linewidth": 25,'alpha':1})
ax.set(xlabel='Bernouli', ylabel='Frequency')
plt.show()

#Example2
#https://data-flair.training/blogs/python-probability-distributions/
import numpy as np
import matplotlib.pyplot as plt
s=np.random.binomial(10,0.5,1000)
plt.hist(s,16,color='Brown')
plt.show()



#Binomial distribution

#Example1
#https://data-flair.training/blogs/python-probability-distributions/
#Describes variation in the number of one outcome from replicate to replicate
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import binom
n=20
p=0.8
binom.rvs(size=10,n=n,p=p)
moments = binom.stats(n, p, moments='mvsk')
data_binom = binom.rvs(n=20,p=0.8,loc=0,size=1000)
ax = sb.distplot(data_binom,kde=True,color='blue',hist_kws={"linewidth": 25,'alpha':1})
ax.set(xlabel='Binomial', ylabel='Frequency')
plt.show()
print('mean,var,skew,kurt =', moments)

#Example2
#Describes variation in the number of one outcome from replicate to replicate
from scipy.stats import binom
import matplotlib.pyplot as plt
import seaborn as sb
n = 484382                   #number of of particular set of trials  (e.g. male births)
N = 938223                   #total number 
ep = (n-1)/(N+2)              #estimated probability of occurence of each trial
size = shape of returned array
binom.rvs(size=n,n=n,p=ep)
moments = binom.stats(n=n, p=ep, moments='mvsk')
data_binom = binom.rvs(n=n,p=ep,loc=0,size=n)
ax = sb.distplot(data_binom,kde=True,color='blue',hist_kws={"linewidth": 25,'alpha':1})
ax.set(xlabel='Binomial', ylabel='Frequency')
plt.show()
print('mean,var,skew,kurt =', moments)

#Confidence intervals are more infomative than pvalues
#CI communicates detectability like pvalue but also results in effect size and uncertainty of that effect size
#95% CI does not mean the true value of the parameter lies within that interval 

#Standard error
se = np.sqrt(ep * (1-ep)/N)
W = 1.96 * se             #We can assume the z statistic = 1.96 is we assume alpha = 0.05
#Other possible intervals: 99% CI means z = 2.58; 68% CI means z = 0.99

#Confidence intervals
cil = ep - W
cir = ep + W

print('standard error =', se)
print('Left CI =', cil)
print('Right CI =', cir)



#Poisson distribution

#https://data-flair.training/blogs/python-probability-distributions/
import numpy as np
import matplotlib.pyplot as plt
s=np.random.poisson(5, 10000)
plt.hist(s,16,normed=True,color='Green')



#Normal distribution

#https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html
import numpy as np
import matplotlib.pyplot as plt
mu = 0.0       #mean
sd = 0.1    #standard deviation
s = np.random.normal(mu, sd, 1000)
count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sd * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sd**2) ),linewidth=2, color='r')
plt.show()




#Chi-square test

#Example1: nown proportions
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
from scipy.stats import chisquare
#first list is of observed and second list is of expected
#To find expected values: multiply the row total by the column total and divide by the grand total
#To manually calculate chi square: (observed - expected)^2/expected sum and total
K = 1    #number of categories
dof = K - 1
chi = chisquare([16, 18, 16, 14, 12, 12], f_exp=[16, 16, 16, 16, 16, 8], ddof=dof)
print('chi/pvalue=', chi)

#Example2: known proportions
#https://www.geeksforgeeks.org/python-pearsons-chi-square-test/
from scipy.stats import chi2_contingency   
data = [[207, 282, 241], [234, 242, 232]] 
stat, p, dof, expected = chi2_contingency(data) 
alpha = 0.05
print('pvalue =', p) 
if p <= alpha: 
   print('Dependent (reject H0)') 
else: 
   print('Independent (H0 holds true)') 

#Example3: unknown proportions
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html#scipy.stats.chisquare
from scipy.stats import chisquare
m = 6            #number of possible values for variable A
n = 6             #number of possible values for variable B
dof = n*m - m - n + 1
chi = chisquare([16, 18, 16, 14, 12, 12], f_exp=[16, 16, 16, 16, 16, 8], ddof=dof)
print('chi/pvalue=',chi)




#Covariance, correlation, and variance of a sample

#Covariance

#Example1
#https://numpy.org/doc/stable/reference/generated/numpy.cov.html
import numpy as np
x = np.array([[0, 2], [1, 1], [2, 0]]).T
c = np.cov(x)
print('covariance matrix =', c)

#Covariance matrix prints
#[a,a]  [a,b]
#[a,b]  [b,b]

#Example2
import numpy as np
x = [-2.1, -1,  4.3]
y = [3,  1.1,  0.12]
X = np.stack((x, y), axis=0)
c = np.cov(x, y)
print('covariance matrix =', c)


#Correlation

#Example1
#https://realpython.com/numpy-scipy-pandas-correlation-python/
import numpy as np
x = np.arange(10, 20)
y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
r = np.corrcoef(x, y)
print('correlation matrix =', r)

#Example2
import numpy as np
import scipy.stats
x = np.arange(10, 20)
y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
p = scipy.stats.pearsonr(x, y)    # Pearson's r
s = scipy.stats.spearmanr(x, y)   # Spearman's rho
k = scipy.stats.kendalltau(x, y)  # Kendall's tau
print('Pearsons r/twotailed pvalue =', p)
print('Spearmeans rho/pvalue = ', s)
print('Kendalls tau/pvalue =', k)


#Variance

#Example1
#https://appdividend.com/2019/08/08/python-variance-example-python-statistics-variance-function/
import statistics 
sample = [2.74, 1.23, 2.63, 2.22, 3, 1.98] 
v = statistics.variance(sample)
print('variance =', v)

#Example2
import numpy as np
dataset= [21, 11, 19, 18, 29, 46, 20]
v= np.var(dataset)
print('variance =', v)




#QQ Plots

#Example1
#https://www.statology.org/q-q-plot-python/
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
#follows a normal distribution
np.random.seed(0)
data = np.random.normal(0,1, 1000)
#create Q-Q plot with 45-degree line added to plot
fig = sm.qqplot(data, line='45')
plt.show()

#Example2
#uniformally distributed values
data = np.random.uniform(0,1, 1000)
fig = sm.qqplot(data, line='45')
plt.show()




#Power analysis

#Example1
#https://machinelearningmastery.com/statistical-power-and-power-analysis-in-python/
#estimate sample size via power analysis
from statsmodels.stats.power import TTestIndPower
effect = 0.8
alpha = 0.05
power = 0.8
analysis = TTestIndPower()
result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
print('Sample Size =', result)

#Example2
#calculate power curves for varying sample and effect size
from numpy import array
from matplotlib import pyplot
from statsmodels.stats.power import TTestIndPower
effect_sizes = array([0.2, 0.5, 0.8])
sample_sizes = array(range(5, 100))
analysis = TTestIndPower()
analysis.plot_power(dep_var='nobs', nobs=sample_sizes, effect_size=effect_sizes)
pyplot.show()

#Example3
#https://towardsdatascience.com/introduction-to-power-analysis-in-python-e7b748dfa26
import numpy as np
import pandas as pd
from statsmodels.stats.power import TTestIndPower
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode()
effect_size = 0.8
alpha = 0.05 # significance level
power = 0.8
power_analysis = TTestIndPower()
sample_size = power_analysis.solve_power(effect_size = effect_size,power = power,alpha = alpha)
print('Required sample size =', sample_size)

#How does the sample size influence the power while keeping the significance level and the effect size at certain levels?
#the bigger the sample, the higher the power, keeping other parameters constant
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
fig = TTestIndPower().plot_power(dep_var='nobs',nobs= np.arange(2, 200),effect_size=np.array([0.2, 0.5, 0.8]),alpha=0.01,ax=ax, title='Power of t-Test' + '\n' + r'$\alpha = 0.01$')
ax.get_legend().remove()
ax = fig.add_subplot(2,1,2)
fig = TTestIndPower().plot_power(dep_var='nobs',nobs= np.arange(2, 200),effect_size=np.array([0.2, 0.5, 0.8]),alpha=0.05,ax=ax, title=r'$\alpha = 0.05$') 
fig.subplots_adjust(top = 1.4)

#Expanded to 3D
@np.vectorize
def power_grid(x,y):
    power = TTestIndPower().solve_power(effect_size = x, nobs1 = y, alpha = 0.05)
    return power
X,Y = np.meshgrid(np.linspace(0.01, 1, 51), np.linspace(10, 1000, 100))
X = X.T
Y = Y.T
Z = power_grid(X, Y) # power
data = [Surface(x = X, y= Y, z = Z)]
layout = Layout(title='Power Analysis',scene = dict(xaxis = dict(title='effect size'),yaxis = dict(title='number of observations'),zaxis = dict(title='power'),))
fig = Figure(data=data, layout=layout)
iplot(fig)





#One way ANOVA

#Example1
#https://www.marsja.se/four-ways-to-conduct-one-way-anovas-using-python/
import pandas as pd
datafile = "PlantGrowth.csv"
data = pd.read_csv(datafile)
data.boxplot('weight', by='group', figsize=(12, 8))
ctrl = data['weight'][data.group == 'ctrl']
grps = pd.unique(data.group.values)
d_data = {grp:data['weight'][data.group == grp] for grp in grps}
k = len(pd.unique(data.group))  # number of conditions
N = len(data.values)  # conditions times participants
n = data.groupby('group').size()[0] #Participants in each condition

#Example2
from scipy import stats
datafile = "PlantGrowth.csv"
data = pd.read_csv(datafile)
F, p = stats.f_oneway(d_data['ctrl'], d_data['trt1'], d_data['trt2'])
DFbetween = k - 1
DFwithin = N - k
DFtotal = N - 1
print('fvalue/pvalue =', F,p)

#Example3
datafile = "PlantGrowth.csv"
data = pd.read_csv(datafile)
SSbetween = (sum(data.groupby('group').sum()['weight']**2)/n) \- (data['weight'].sum()**2)/N
sum_y_squared = sum([value**2 for value in data['weight'].values])
SSwithin = sum_y_squared - sum(data.groupby('group').sum()['weight']**2)/n
SStotal = sum_y_squared - (data['weight'].sum()**2)/N
MSbetween = SSbetween/DFbetween
MSwithin = SSwithin/DFwithin
F = MSbetween/MSwithin
p = stats.f.sf(F, DFbetween, DFwithin)
eta_sqrd = SSbetween/SStotal
om_sqrd = (SSbetween - (DFbetween * MSwithin))/(SStotal + MSwithin)
print('fvalue =', F)
print('pvalue =', p)
print('effect size =', eta_sqrd)
print('omega sqrd =', om_sqrd)




#Two way ANOVA

#Example1
#https://www.marsja.se/three-ways-to-carry-out-2-way-anova-with-python/
import pandas as pdimport statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from scipy import stats
datafile = "ToothGrowth.csv"
data = pd.read_csv(datafile)
fig = interaction_plot(data.dose, data.supp, data.len,colors=['red','blue'], markers=['D','^'], ms=10)
N = len(data.len)
df_a = len(data.supp.unique()) - 1
df_b = len(data.dose.unique()) - 1
df_axb = df_a*df_b 
df_w = N - (len(data.supp.unique())*len(data.dose.unique()))
grand_mean = data['len'].mean()
ssq_a = sum([(data[data.supp ==l].len.mean()-grand_mean)**2 for l in data.supp])
ssq_b = sum([(data[data.dose ==l].len.mean()-grand_mean)**2 for l in data.dose])
ssq_t = sum((data.len - grand_mean)**2)
vc = data[data.supp == 'VC']
oj = data[data.supp == 'OJ']
vc_dose_means = [vc[vc.dose == d].len.mean() for d in vc.dose]
oj_dose_means = [oj[oj.dose == d].len.mean() for d in oj.dose]
ssq_w = sum((oj.len - oj_dose_means)**2) +sum((vc.len - vc_dose_means)**2
ssq_axb = ssq_t-ssq_a-ssq_b-ssq_w
ms_a = ssq_a/df_a
ms_b = ssq_b/df_b
ms_axb = ssq_axb/df_axb
ms_w = ssq_w/df_w
f_a = ms_a/ms_w
f_b = ms_b/ms_w
f_axb = ms_axb/ms_w
p_a = stats.f.sf(f_a, df_a, df_w)
p_b = stats.f.sf(f_b, df_b, df_w)
p_axb = stats.f.sf(f_axb, df_axb, df_w)
results = {'sum_sq':[ssq_a, ssq_b, ssq_axb, ssq_w],'df':[df_a, df_b, df_axb, df_w],'F':[f_a, f_b, f_axb, 'NaN'],'PR(>F)':[p_a, p_b, p_axb, 'NaN']}
columns=['sum_sq', 'df', 'F', 'PR(>F)']
aov_table1 = pd.DataFrame(results, columns=columns,index=['supp', 'dose','supp:dose', 'Residual'])
def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    return aov
def omega_squared(aov):
    mse = aov['sum_sq'][-1]/aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))/(sum(aov['sum_sq'])+mse)
    return aov
eta_squared(aov_table1)
omega_squared(aov_table1)
print(aov_table1)

#Example2
#https://github.com/marsja/jupyter/blob/master/Python_ANOVA/Python_ANOVA_Factorial_Using_Statsmodels.ipynb
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
data = 'https://vincentarelbundock.github.io/Rdatasets/csv/datasets/ToothGrowth.csv'
df = pd.read_csv(data, index_col=0)
formula = 'len~C(supp)+C(dose)+C(supp):C(dose)'
model = ols(formula, df).fit()
aov_table = anova_lm(model, typ=2)
print(aov_table)
def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    return aov
def omega_squared(aov):
    mse = aov['sum_sq'][-1]/aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))/(sum(aov['sum_sq'])+mse)
    return aov
eta_squared(aov_table)
omega_squared(aov_table)
res = model.resid 
fig = sm.qqplot(res, line='s')
plt.show()
print(aov_table.round(4))




#Tukey's test

#https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.signal.tukey.html
from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt
window = signal.tukey(51)
plt.plot(window)
plt.title("Tukey window")
plt.ylabel("Amplitude")
plt.xlabel("Sample")
plt.ylim([0, 1.1])
plt.figure()
A = fft(window, 2048) / (len(window)/2.0)
freq = np.linspace(-0.5, 0.5, len(A))
response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
plt.plot(freq, response)
plt.axis([-0.5, 0.5, -120, 0])
plt.title("Frequency response of the Tukey window")
plt.ylabel("Normalized magnitude [dB]")
plt.xlabel("Normalized frequency [cycles per sample]")





#Regressions

#Linear 

#Example1
#https://thecleverprogrammer.com/2020/06/05/statistics-tutorial-for-data-science/
#see website for linear regression from scratch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
x = np.random.uniform(low=20, high=260, size=100)
y = 50000 + 2000*x - 4.5 * x**2 + np.random.normal(size=100, loc=0, scale=10000)
plt.figure(figsize=(16,5))
plt.title('Title', fontsize='xx-large')
sns.regplot(x, y)

#Example2
#https://realpython.com/linear-regression-in-python/
import numpy as np
from sklearn.linear_model import LinearRegression
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
y_pred = model.predict(x)
y_pred = model.intercept_ + model.coef_ * x
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
print('predicted response:', y_pred, sep='\n')
print('predicted response:', y_pred, sep='\n')



#Polynomial regression

#Example1
##https://realpython.com/linear-regression-in-python/
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])
transformer = PolynomialFeatures(degree=2, include_bias=False)
transformer.fit(x)
x_ = transformer.transform(x)
x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
model = LinearRegression().fit(x_, y)
r_sq = model.score(x_, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('coefficients:', model.coef_)
x_ = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)
model = LinearRegression(fit_intercept=False).fit(x_, y)
r_sq = model.score(x_, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('coefficients:', model.coef_)
y_pred = model.predict(x_)
print('predicted response:', y_pred, sep='\n')

#Example2
#https://pythonbasics.org/polynomial-regression-in-python/
import numpy as np
import matplotlib.pyplot as plt
X = [1, 5, 8, 10, 14, 18]
Y = [1, 1, 10, 20, 45, 75]
degree = 2
poly_fit = np.poly1d(np.polyfit(X,Y, degree))
xx = np.linspace(0, 26, 100)
plt.plot(xx, poly_fit(xx), c='r',linestyle='-')
plt.title('Polynomial')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis([0, 25, 0, 100])
plt.grid(True)
plt.scatter(X, Y)
plt.show()
print( poly_fit(12) )




#Non-linear regression

#https://scipy-cookbook.readthedocs.io/items/robust_regression.html
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = (10, 6)
rcParams['legend.fontsize'] = 16
rcParams['axes.labelsize'] = 16
r = np.linspace(0, 5, 100)
linear = r**2
huber = r**2
huber[huber > 1] = 2 * r[huber > 1] - 1
soft_l1 = 2 * (np.sqrt(1 + r**2) - 1)
cauchy = np.log1p(r**2)
arctan = np.arctan(r**2)
plt.plot(r, linear, label='linear')
plt.plot(r, huber, label='huber')
plt.plot(r, soft_l1, label='soft_l1')
plt.plot(r, cauchy, label='cauchy')
plt.plot(r, arctan, label='arctan')
plt.xlabel("$r$")
plt.ylabel(r"$\rho(r^2)$")
plt.legend(loc='upper left')



#Log transform

#Example1
#https://www.geeksforgeeks.org/numpy-log-python/
import numpy as np 
in_array = [1, 3, 5, 2**8] 
print ("Input array : ", in_array) 
out_array = np.log(in_array) 
print ("Output array : ", out_array) 
print("\nnp.log(4**4) : ", np.log(4**4)) 
print("np.log(2**8) : ", np.log(2**8)) 

#Example2
import numpy as np 
import matplotlib.pyplot as plt 
in_array = [1, 1.2, 1.4, 1.6, 1.8, 2] 
out_array = np.log(in_array) 
print ("out_array : ", out_array) 
plt.plot(in_array, in_array,color = 'blue', marker = "*") 
plt.plot(out_array, in_array,color = 'red', marker = "o")
plt.title("numpy.log()") 
plt.xlabel("out_array") 
plt.ylabel("in_array") 
plt.show()  

















































