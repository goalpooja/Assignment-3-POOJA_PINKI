import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

#Create a streamlit app
st.title("Effect of Cross Covariance")
st.write(
    "visualization of the effect of cross covariance by 3-sigma ellipse plots"
)
with st.sidebar:
    # Create a slider for degree
    a = st.slider("Covarience", -5.5,5.5,0.01)
    b=st.slider("mean",0.0,4.0,0.1)
    q=st.slider("varience", -7.5,7.5,0.01)

    # Create a slider for alpha
    
mean = [b, b]
cov = [[q, a], [a, q]]
data = np.random.multivariate_normal(mean, cov, 1000)
fig,ax=plt.subplots()
ax.set_xlabel("Feature 1")

ax.set_ylabel("Feature 2")
ax.scatter(data[:, 0], data[:, 1], s=3)
x = data[:, 0]
y = data[:, 1]
cov_xy = np.cov(x, y)[0][1]
std_x = np.std(x)
std_y = np.std(y)

# Calculate correlation coefficient r


r = cov_xy / (std_x * std_y)

covariance = np.cov(data.T)
inverse_covariance = np.linalg.inv(covariance)
mean = np.mean(data, axis=0)
scale_factor = 10
x = np.linspace(-scale_factor, scale_factor, 100)
y = np.linspace(-scale_factor, scale_factor, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i in range(len(x)):
    for j in range(len(y)):
        point = np.array([x[i], y[j]])
        deviation = point - mean
        Z[j, i] = np.dot(np.dot(deviation, inverse_covariance), deviation.T)

# Plot the 3 sigma ellipse plots
plt.contour(X, Y, Z, levels=[scale_factor**1/2], colors='r')
plt.contour(X, Y, Z, levels=[scale_factor**1/4], colors='g')
plt.contour(X, Y, Z, levels=[scale_factor**1/8], colors='#FFA500')
st.set_option('deprecation.showPyplotGlobalUse', False)
s=""
if r < 0.3 :
    s= "None or very weak"
elif r>0.3  and r<0.5:
    s="Weak"  
elif r>0.5 and r < 0.7: 
    s="Moderate"
elif r > 0.7:
    s="Strong"
if r==0:
    st.subheader(s+" No Correlation")
elif r>0:
    st.subheader(s+" Positive Correlation")
else:
    st.subheader(s+" Negative Correlation")

st.write("value of correlation co-efficient: r",r)
st.pyplot()
    
    
st.write("---")
