import random
import math
import statistics as stats
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


#simple linear regression: y = ax + b,
#generate data pairs (xi,yi)
samplesize = 5

# Xobs=np.random.randint(1, 10, samplesize)
Xobs = np.arange(1,21,4)
Yobs=Xobs * 3 + 8
# add random noise to observations
# Yobs = Yobs + np.random.normal(0, 1, size=(5,))
yStdev=stats.stdev(Yobs)

# plt.scatter(Xobs,Yobs)
# plt.show(block=False)
# stop=1

# measurement errors
def MeasurementErrorsDiagonalMatrix():
    r=np.random.normal(loc=0.0, scale=0.1, size=samplesize)
    r=[i**2 for i in r]
    R=np.matrix(np.diag(r))
    return R

def linear_fun(x, a, b):
    """

    :param x:
    :param a:
    :param b:
    :return:
    """
    output = x*b + a
    # output = np.zeros((b.shape[0], x.shape[0]))
    # for i in range(0, x.shape[0]):
    #     output[:,i] = np.multiply(x[i], b) + a

    return output


def prior(theta):

    prior_out = scipy.stats.multivariate_normal.pdf(theta[:2],mean=np.array([8,3]), cov=np.eye(2)*np.array([15,5]))

    return prior_out




def likelihood(x,y,theta):
    lhd_out = scipy.stats.norm.logpdf(y, theta[1] * x + theta[0], 0.03)
    lhd_out = np.sum(lhd_out)
    return lhd_out


#print(prior())

def MClikelihood(theat):
    # Ymod = theat[1] * Xobs + theat[0]
    Ymod = scipy.stats.multivariate_normal(theat[1] * Xobs + theat[0], 0.03).rvs(1)
    Deviation = np.matrix(Ymod - Yobs)

    # 1 / (math.sqrt(2 * math.pi * math.exp(1))) ** samplesize *

    cov = np.linalg.inv(np.eye(samplesize)) * 0.03 ** 2
    #Lieklihoodout = scipy.stats.multivariate_normal.pdf(x=Ymod, mean=Yobs, cov=cov)
    Lieklihoodout = 1 / (math.sqrt(2 * math.pi * math.exp(1))) ** samplesize *math.exp(-0.5 * Deviation * np.linalg.inv(np.eye(samplesize) * 0.03) * np.transpose(Deviation))
    return Lieklihoodout


#MC and main loop ex_a
#def main():


MC = 100000
meas_err = 0.03
posteriora = []
priora = np.arange(MC,dtype=float)
posteriorb = []
priorb = np.arange(MC,dtype=float)
Lieklihoodout = np.arange(MC,dtype=float)
#lik = np.arange(MC,dtype=float)

# sample_a = np.random.normal(loc=8, scale=15, size=(MC,))
# sample_b = np.random.normal(loc=3, scale=5, size=(MC,))
# Ymod = linear_fun(Xobs, sample_a, sample_b)
# R = np.eye(samplesize)*meas_err**2
# diff = np.subtract(Ymod, Yobs, axis=0)
# lk = scipy.stats.multivariate_normal()
for i in range(0,MC):
    priorcomb = np.zeros(3)
    priorcomb[0] = np.random.normal(loc=8, scale=15, size=1)
    # priorcomb[0]= np.random.uniform(0, 15, 1)# a
    priorcomb[1]= np.random.normal(loc=3, scale=5, size=1)
    # priorcomb[1]= np.random.uniform(-5, 5, 1)# b
    priorcomb[2]= random.random()
    erro = random.random()
    Ymod = priorcomb[1] * Xobs + priorcomb[0]
    Ymod = scipy.stats.multivariate_normal(priorcomb[1] * Xobs + priorcomb[0],erro).rvs(1)
    Deviation = np.matrix(Ymod - Yobs)
    cov = np.linalg.inv(np.eye(samplesize))*erro**2
    # Lieklihoodout[i] = scipy.stats.multivariate_normal.pdf(x=Ymod, mean=Yobs, cov=cov)
    #1 / (math.sqrt(2 * math.pi * math.exp(1))) ** samplesize *
    Lieklihoodout[i]=math.exp(-0.5*Deviation*np.linalg.inv(np.eye(samplesize)*erro)*np.transpose(Deviation))
    aout = priorcomb[0]
    priora[i] = aout
    bout = priorcomb[1]
    priorb[i] = bout


#
for i in range(0,MC):
    if ((Lieklihoodout[i] / max(Lieklihoodout))) > random.random():
        posteriora.append(priora[i])
        posteriorb.append(priorb[i])



# plt.figure(8)
# plt.scatter(MC, posteriora, c ="blue")

plt.figure(1)
plt.hist(priora)
plt.suptitle('prior_a')
plt.figure(2)
plt.hist(priorb)
plt.suptitle('prior_b')

plt.figure(7)
plt.hist(posteriora)
plt.suptitle('posterior_a')
plt.figure(8)
plt.hist(posteriorb)
plt.suptitle('posterior_b')






#ex_b
#proposal distribution
def distribution_proposal(theta):
    out_theta = np.zeros(2)
    out_theta[:2] = scipy.stats.multivariate_normal(mean=theta[:2], cov=np.eye(2) * 0.05).rvs(1)
    # the last component is the erro
    # out_theta[2] = np.random.normal(loc=0, scale=0.03, size=1)
    return out_theta




def proposal_ratio(theta_old, theta_new):
    prop_out1 = scipy.stats.multivariate_normal.logpdf(theta_old[:2],mean=theta_new[:2], cov=np.eye(2)*10)
    prop_out2 = scipy.stats.multivariate_normal.logpdf(theta_new[:2],mean=theta_old[:2], cov=np.eye(2)*10)
    prop_ratio_out = prop_out1 - prop_out2
    return prop_ratio_out



#main ex_b
# N=2000
# # theta = np.random.rand(2).reshape(1,-1)
# theta =np.zeros(2)
# theta[0] = 15
# theta[1] = 5
# # theta[2] = 0.03
# theta = theta.reshape(1,-1)
# theta_a = np.arange(N,dtype=float)
# theta_b = np.arange(N,dtype=float)
# rejection = 0
# acception = 0
# for i in range(0, N-1):
#     theta_new = distribution_proposal(theta[-1])
#     # lik_theta_new = np.exp(likelihood(Yobs,Xobs,theta_new))
#     MClik_theta_new = MClikelihood(theta_new)
#     # lik_theta =  likelihood(Yobs,Xobs,theta[-1])
#     MClik_theta = MClikelihood(theta[-1])
#
#     theta_new_prior = prior(theta_new)
#     theta_prior = prior(theta[-1])
#
#     #alpha1 = lik_theta_new + theta_new_prior - (lik_theta + theta_prior)
#     alpha1 = MClik_theta_new * theta_new_prior / (MClik_theta * theta_prior)
#     #alpha2 = proposal_ratio(theta[-1], theta_new)
#     alpha2 = 1
#     alpha = alpha1 * alpha2
#     if  random.random() < min(1,alpha):
#         theta=np.vstack((theta,theta_new))
#         # theta_a[i + 1] = theta_new[1]
#         # theta_b[i + 1] = theta_new[0]
#         acception += 1
#     else:
#         theta = np.vstack((theta,theta[-1]))
#         # theta_a[i + 1] = theta[1]
#         # theta_b[i + 1] = theta[0]
#         rejection += 1
#
#
#
#
# print(1-(rejection/(rejection+acception)))
# # test2=np.mean(theta_a)
# # test3=np.mean(theta_b)
# # print(test2,test3)
# plt.figure(3)
#
# plt.hist(priora)
# plt.hist(theta[:,0])
# labels= ["prior_a","positria_a"]
# plt.legend(labels)
#
# plt.figure(4)
#
# plt.hist(priorb)
# plt.hist(theta[:,1])
# labels= ["prior_b","positria_b"]
# plt.legend(labels)
#
#
# plt.figure(5)
#
# plt.plot(range(0,N),theta[:,0])
# plt.plot(range(0,N),theta[:,1])
# labels = ["positria_a","positria_b"]
# plt.legend(labels)
plt.show()
