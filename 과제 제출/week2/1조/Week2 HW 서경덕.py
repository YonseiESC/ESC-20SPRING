#Real data import
from io import StringIO
import requests
import pandas as pd

url = 'https://raw.githubusercontent.com/YonseiESC/ESC20-SPRING/master/세션%20자료/week2/real.csv'
url_text =requests.get(url).text
real=pd.read_csv(StringIO(url_text),index_col=0)

#data cleansing
real.columns = ['data','house_age','dist_mrt','no_cvs','lat','long','price']
real.info() #non-null -> NA없음.
real.sort_values(by=['price'], inplace=True) #inplace=T를 해야 실제 데이터셋의 값이 바뀐다
real.reset_index(inplace=True) #inplace: replace the original dataset
real = real.drop(['No'],axis=1) #axis=1 : col drop
real.describe()

#EDA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

real.hist(bins=50, figsize=(12,10)); plt.tight_layout()
np.percentile(real.price, 99.5)
real = real[real.price < 80]
plt.hist(real.price, bins=50); plt.tight_layout()
sns.pairplot(real[['dist_mrt','house_age','no_cvs','price']])

#Transformation
real2 = real.copy()
real2['dist_mrt']=np.log(1+real.dist_mrt)
sns.pairplot(real2[['dist_mrt','house_age','no_cvs','price']])

# 1.Train/test set splitting
from sklearn.model_selection import train_test_split
real2 = real2[['dist_mrt','house_age','price']]
x=real2.drop('price',axis=1);y=real2['price']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=123)

# 2. Polynomial Basis Feature Extraction
from sklearn.preprocessing import StandardScaler
'''
StandardScaler(X): 평균이 0과 표준편차가 1이 되도록 변환.
RobustScaler(X): 중앙값(median)이 0, IQR(interquartile range)이 1이 되도록 변환.
MinMaxScaler(X): 최대값이 각각 1, 최소값이 0이 되도록 변환
MaxAbsScaler(X): 0을 기준으로 절대값이 가장 큰 수가 1또는 -1이 되도록 변환
'''
# 학습용 데이터의 분포 추정 -> training set 변환 -> test set 변환
#splitting -> normalization : good to produce a predictive algorithm.
#normalization -> splitting :  good to understand the structure of the data

scaler = StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train) #mean->0, sd->1
x_test_scaled = scaler.transform(x_test) #x_train 기준으로 변환

#Polynomial Basis Fitting MSE comparison(Train vs Test)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.express as px

def Poly_Reg(X_train, Y_train, X_test, Y_test, m):
    # Feature Extraction
    poly = PolynomialFeatures(degree=m)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)

    # LR Fitting
    lin = LinearRegression(fit_intercept=False)
    lin.fit(X_train_poly, Y_train)
    Y_train_pred = lin.predict(X_train_poly)
    Y_test_pred = lin.predict(X_test_poly)

    # Compue MSE
    train_MSE = np.sqrt(np.linalg.norm(Y_train - Y_train_pred)**2/Y_train.shape[0])
    test_MSE = np.sqrt(np.linalg.norm(Y_test - Y_test_pred)**2/Y_test.shape[0])

    # Residual
    Resid = Y_train - Y_train_pred

    return {'coef': lin.coef_, 'train_MSE': train_MSE, 'test_MSE': test_MSE, 'Resid': Resid}

PRmodel = Poly_Reg(x_train_scaled,y_train,x_test_scaled,y_test,2)
PRmodel
fig = px.scatter_3d(x=x_train.dist_mrt, y=x_train.house_age, z=PRmodel['Resid'])
fig.update_traces(marker=dict(size=4,
                              line=dict(width=0.1,
                                        color='DarkSlateGrey')))
fig.show()

output = pd.DataFrame(columns=['d','Train MSE','Test MSE'])
for i in np.arange(12):
    m = i+1
    Reg = Poly_Reg(x_train_scaled, y_train, x_test_scaled, y_test, m)
    output.loc[i] = [m, Reg['train_MSE'], Reg['test_MSE']]
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(output['d'], np.log(output['Train MSE']), 'o-', label='Train MSE')
ax.plot(output['d'], np.log(output['Test MSE']), 'o-', label='Test MSE')
ax.legend()

#Feature Extraction
poly = PolynomialFeatures(degree=2)
phi_train = poly.fit_transform(x_train_scaled)
phi_test = poly.fit_transform(x_test_scaled)

#Simple Linear Regression
lin = LinearRegression()
lin.fit(phi_train,y_train)
print(lin.intercept_, lin.coef_)

# 3. Regularization
#Ridge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
alphas = np.logspace(-6,6,300)

def Ridge_Reg(Phi_train, Y_train, Phi_test, Y_test, alphas):
    reg = Ridge()
    coefs = []
    train_MSE = []
    test_MSE = []

    for a in alphas:
        reg.set_params(alpha=a)
        reg.fit(Phi_train, Y_train)
        coefs.append(reg.coef_)
        train_pred = reg.predict(Phi_train)
        train_MSE.append(mean_squared_error(Y_train, train_pred))
        test_pred = reg.predict(Phi_test)
        test_MSE.append(mean_squared_error(Y_test, test_pred))

    return {'coefs': coefs, 'train_MSE': train_MSE, 'test_MSE': test_MSE}

Ridge_output = Ridge_Reg(phi_train, y_train, phi_test, y_test, alphas)
Ridge_output

#Lasso
from sklearn.linear_model import Lasso
def Lasso_Reg(Phi_train, Y_train, Phi_test, Y_test, alphas):
    reg = Lasso()
    coefs=[]
    train_MSE=[]
    test_MSE = []
    for a in alphas:
        reg.set_params(alpha=a)
        reg.fit(Phi_train,Y_train)
        coefs.append(reg.coef_)
        train_pred=(reg.predict(Phi_train))
        train_MSE.append(mean_squared_error(Y_train, train_pred))
        test_pred=(reg.predict(Phi_test))
        test_MSE.append(mean_squared_error(Y_test,test_pred))

    return {'coefs':coefs, 'train_MSE':train_MSE, 'test_MSE':test_MSE}

Lasso_output=Lasso_Reg(phi_train,y_train,phi_test,y_test,alphas)
Lasso_output

# 4. k-cv sampling
from sklearn.model_selection import KFold

kcv = KFold(n_splits=5, shuffle=True)
kcvMSE_Ridge = np.zeros((kcv.get_n_splits(), alphas.size))
kcvMSE_Lasso = np.zeros((kcv.get_n_splits(), alphas.size))

i = 0
for train_index, test_index in kcv.split(phi_train, y_train):
    print("TRAIN:", train_index, "TEST:", test_index)
    CV_phi_train, CV_phi_test = phi_train[train_index, :], phi_train[test_index, :]
    CV_y_train, CV_y_test = y_train.iloc[train_index], y_train.iloc[test_index]
   # iloc -> 행데이터 추출

    Ridge_output = Ridge_Reg(CV_phi_train, CV_y_train, CV_phi_test, CV_y_test, alphas)
    kcvMSE_Ridge[i, :] = Ridge_output['test_MSE']

    Lasso_output = Lasso_Reg(CV_phi_train, CV_y_train, CV_phi_test, CV_y_test, alphas)
    kcvMSE_Lasso[i, :] = Lasso_output['test_MSE']
    i += 1

kcvMSE_Ridge

fig, ax = plt.subplots(figsize=(10,6))
meanMSE = np.apply_along_axis(np.mean, 1, kcvMSE_Ridge.T)
ax.plot(alphas, kcvMSE_Ridge.T, linestyle='--')
ax.plot(alphas, meanMSE, color='black', label='Average MSE')
ax.set_xscale('log')
ax.axvline(x=alphas[np.argmin(meanMSE)], linestyle='--', color='black')
alpha = np.around(alphas[np.argmin(meanMSE)],3)
ax.text(alphas[np.argmin(meanMSE)],np.amin(meanMSE),'alpha={0}'.format(alpha),size=15)
ax.set_title('Ridge Complexity')
ax.legend()

coefs = Ridge_output['coefs']
print("Weight:", coefs[np.argmin(meanMSE)])
Ridge_alphas = [alphas[np.argmin(meanMSE)]]

fig, ax = plt.subplots(figsize=(10,6))
meanMSE = np.apply_along_axis(np.mean, 1, kcvMSE_Lasso.T)
ax.plot(alphas, kcvMSE_Lasso.T, linestyle='--')
ax.plot(alphas, meanMSE, color='black', label='Average MSE')
ax.set_xscale('log')
ax.axvline(x=alphas[np.argmin(meanMSE)], linestyle='--', color='black')
alpha = np.around(alphas[np.argmin(meanMSE)],3)
ax.text(alphas[np.argmin(meanMSE)],np.amin(meanMSE),'alpha={0}'.format(alpha),size=15)
ax.set_title('Lasso Complexity')
ax.legend()

coefs = Lasso_output['coefs']
print("Weight:", coefs[np.argmin(meanMSE)])
Lasso_alphas = [alphas[np.argmin(meanMSE)]]

# 5.최종적으로 test MSE 보고 후, feature선택 및 이유설명
PRmodel['coef']
PRmodel['test_MSE']

Ridge_output = Ridge_Reg(phi_train, y_train, phi_test, y_test, Ridge_alphas)
Ridge_output['test_MSE']
Ridge_alphas
Ridge_output['coefs']

Lasso_output = Lasso_Reg(phi_train, y_train, phi_test, y_test, Lasso_alphas)
Lasso_output['test_MSE']


