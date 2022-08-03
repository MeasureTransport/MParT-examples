import numpy as np
from pandas import MultiIndex
import scipy 
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import multivariate_normal
import time
import os

os.environ['KOKKOS_NUM_THREADS'] = '8'
from mpart import *

print('Kokkos is using', Concurrency(), 'threads')

rho1 = multivariate_normal(np.zeros(1),np.eye(1))

def generate_SV_samples(d,N):
    # Sample hyper-parameters
    sigma = 0.25
    mu = np.random.randn(1,N)
    phis = 3+np.random.randn(1,N)
    phi = 2*np.exp(phis)/(1+np.exp(phis))-1
    X = np.vstack((mu,phi))
    if d > 2:
        # Sample Z0
        Z = np.sqrt(1/(1-phi**2))*np.random.randn(1,N) + mu

        # Sample auto-regressively
        for i in range(d-3):
            Zi = mu + phi * (Z[-1,:] - mu)+sigma*np.random.randn(1,N)
            Z = np.vstack((Z,Zi))

        X = np.vstack((X,Z))
    return X

def normpdf(x,mu,sigma):
    return  np.exp(-0.5 * ((x - mu)/sigma)**2) / (np.sqrt(2*np.pi) * sigma);

def SV_log_pdf(X):
    sigma = 0.25

    # Extract variables mu, phi and states
    mu = X[0,:]
    phi = X[1,:]
    Z = X[2:,:]

    # Compute density for mu
    piMu = multivariate_normal(np.zeros(1),np.eye(1))
    logPdfMu = piMu.logpdf(mu)
    # Compute density for phi
    phiRef = np.log((1+phi)/(1-phi))
    dphiRef = 2/(1-phi**2)
    piPhi = multivariate_normal(3*np.ones(1),np.eye(1))
    logPdfPhi = piPhi.logpdf(phiRef)+np.log(dphiRef)
    # Add piMu, piPhi to density
    logPdf = np.vstack((logPdfMu,logPdfPhi))

    # Number of time steps
    dz = Z.shape[0]
    if dz > 0:
        # Conditonal density for Z_0
        muZ0 = mu
        stdZ0 = np.sqrt(1/(1-phi**2))
        logPdfZ0 = np.log(normpdf(Z[0,:],muZ0,stdZ0))
        logPdf = np.vstack((logPdf,logPdfZ0))

        # Compute auto-regressive conditional densities for Z_i|Z_{1:i-1}
        for i in range(1,dz):
            meanZi = mu + phi * (Z[i-1,:]-mu)
            stdZi = sigma
            logPdfZi = np.log(normpdf(Z[i,:],meanZi,stdZi))
            logPdf = np.vstack((logPdf,logPdfZi))

    return logPdf


T=40 #number of time steps
d=T+2

N = 2000 #Number of training samples
X = generate_SV_samples(d,N)

# Negative log likelihood objective
def obj(coeffs, tri_map,x):
    """ Evaluates the log-likelihood of the samples using the map-induced density. """
    num_points = x.shape[1]
    tri_map.SetCoeffs(coeffs)

    # Compute the map-induced density at each point 
    map_of_x = tri_map.Evaluate(x)

    rho = multivariate_normal(np.zeros(tri_map.outputDim),np.eye(tri_map.outputDim))
    rho_of_map_of_x = rho.logpdf(map_of_x.T)
    log_det = tri_map.LogDeterminant(x)

    # Return the negative log-likelihood of the entire dataset
    return -np.sum(rho_of_map_of_x + log_det)/num_points
    
def grad_obj(coeffs, tri_map, x):
    """ Returns the gradient of the log-likelihood objective wrt the map parameters. """
    num_points = x.shape[1]
    tri_map.SetCoeffs(coeffs)

    # Evaluate the map
    map_of_x = tri_map.Evaluate(x)

    # Now compute the inner product of the map jacobian (\nabla_w S) and the gradient (which is just -S(x) here)
    grad_rho_of_map_of_x = -tri_map.CoeffGrad(x, map_of_x)

    # Get the gradient of the log determinant with respect to the map coefficients
    grad_log_det = tri_map.LogDeterminantCoeffGrad(x)
    
    return -np.sum(grad_rho_of_map_of_x + grad_log_det, 1)/num_points


def log_cond_pullback_pdf(triMap,rho,x):
    r = triMap.Evaluate(x)
    log_pdf = rho.logpdf(r.T)+triMap.LogDeterminant(x)
    return log_pdf

def compute_joint_KL(logPdfSV,logPdfTM):
    KL = np.zeros((logPdfSV.shape[0],))
    for k in range(1,d+1):
        KL[k-1]=np.mean(np.sum(logPdfSV[:k,:],0)-np.sum(logPdfTM[:k,:],0))
    return KL

Ntest=2000
Xtest = generate_SV_samples(d,Ntest)
logPdfSV = SV_log_pdf(Xtest)

opts = MapOptions()
opts.basisType = BasisTypes.HermiteFunctions

total_order = 1;
logPdfTM = np.zeros((Ntest,))
ListCoeffs=[];
for dk in range(1,d+1):
    if dk == -2:
        fixed_mset= FixedMultiIndexSet(dk-1,total_order)
        S = CreateComponent(fixed_mset,opts)
        Xtrain = X[dk-1,:].reshape(1,-1)
        Xtestk = Xtest[dk-1,:].reshape(1,-1)
    else:
        fixed_mset= FixedMultiIndexSet(dk,total_order)
        S = CreateComponent(fixed_mset,opts)
        Xtrain = X[:dk,:]
        Xtestk = Xtest[:dk,:]

    options={'gtol': 1e-3, 'disp': True}
    print("Number of coefficients: "+str(S.numCoeffs))
    ListCoeffs.append(S.numCoeffs)
    res = scipy.optimize.minimize(obj, S.CoeffMap(), args=(S, Xtrain), jac=grad_obj, method='BFGS', options=options)
    rho = multivariate_normal(np.zeros(S.outputDim),np.eye(S.outputDim))
    logPdfTM=np.vstack((logPdfTM,log_cond_pullback_pdf(S,rho,Xtestk)))
    xplot = np.linspace(-0.5,1.5,200).reshape(1,-1)
logPdfTM_to1=logPdfTM[1:,:]
KL_to1 = compute_joint_KL(logPdfSV,logPdfTM_to1)

total_order = 2;
logPdfTM = np.zeros((Ntest,))
ListCoeffs=[];
for dk in range(1,d+1):
    print(dk)
    if dk == -2:
        fixed_mset= FixedMultiIndexSet(dk-1,total_order)
        S = CreateComponent(fixed_mset,opts)
        Xtrain = X[dk-1,:].reshape(1,-1)
        Xtestk = Xtest[dk-1,:].reshape(1,-1)
    else:
        fixed_mset= FixedMultiIndexSet(dk,total_order)
        S = CreateComponent(fixed_mset,opts)
        Xtrain = X[:dk,:]
        Xtestk = Xtest[:dk,:]

    options={'gtol': 1e-3, 'disp': True}
    print("Number of coefficients: "+str(S.numCoeffs))
    ListCoeffs.append(S.numCoeffs)
    res = scipy.optimize.minimize(obj, S.CoeffMap(), args=(S, Xtrain), jac=grad_obj, method='BFGS', options=options)
    rho = multivariate_normal(np.zeros(S.outputDim),np.eye(S.outputDim))    
    logPdfTM=np.vstack((logPdfTM,log_cond_pullback_pdf(S,rho,Xtestk)))
logPdfTM_to2=logPdfTM[1:,:]
KL_to2 = compute_joint_KL(logPdfSV,logPdfTM_to2)


total_order = 2;
logPdfTM = np.zeros((Ntest,))
ListCoeffs=[];
mset_to= MultiIndexSet.CreateTotalOrder(4,total_order,NoneLim())

print("Number of coefficients: "+str(mset_to.Size()))
maxOrder=9
for dk in range(1,d+1):
    print(dk)
    if dk == 1:
        fixed_mset= FixedMultiIndexSet(1,total_order)
        S = CreateComponent(fixed_mset,opts)
        Xtrain = X[dk-1,:].reshape(1,-1)
        Xtestk = Xtest[dk-1,:].reshape(1,-1)
    elif dk == 2:
        fixed_mset= FixedMultiIndexSet(1,maxOrder)
        S = CreateComponent(fixed_mset,opts)
        Xtrain = X[dk-1,:].reshape(1,-1)
        Xtestk = Xtest[dk-1,:].reshape(1,-1) 
    elif dk==3:
        fixed_mset= FixedMultiIndexSet(dk,total_order)
        S = CreateComponent(fixed_mset,opts)
        Xtrain = X[:dk,:]
        Xtestk = Xtest[:dk,:]
    else:
        multis=np.zeros((mset_to.Size(),dk))
        for s in range(mset_to.Size()):
            multis_to = np.array([mset_to[s].tolist()])
            multis[s,:2]=multis_to[0,:2]
            multis[s,-2:]=multis_to[0,-2:]
        mset = MultiIndexSet(multis)
        fixed_mset = mset.fix(True)
        S = CreateComponent(fixed_mset,opts)
        Xtrain = X[:dk,:]
        Xtestk = Xtest[:dk,:]

    options={'gtol': 1e-2, 'disp': True}

    res = scipy.optimize.minimize(obj, S.CoeffMap(), args=(S, Xtrain), jac=grad_obj, method='BFGS', options=options)
    rho = multivariate_normal(np.zeros(S.outputDim),np.eye(S.outputDim))    
    logPdfTM=np.vstack((logPdfTM,log_cond_pullback_pdf(S,rho,Xtestk)))

logPdfTM_sa=logPdfTM[1:,:]
KL_sa = compute_joint_KL(logPdfSV,logPdfTM_sa)

fig, ax =plt.subplots()
ax.plot(range(1,d+1),KL_to1,'-o',label='total order 1')
ax.plot(range(1,d+1),KL_to2,'-o',label='total order 2')
ax.plot(range(1,d+1),KL_sa,'-o',label='custom MultiIndexSet')

ax.set_yscale('log')
ax.set_xlabel('d')
ax.set_ylabel('KL')
plt.legend()
plt.show()