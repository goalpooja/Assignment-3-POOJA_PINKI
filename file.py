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
    a = st.slider("Covarience", -3.5,3.5,0.1)
    b=st.slider("mean",0.0,4.0,0.1)

    # Create a slider for alpha
    
mean = [b, b]
cov = [[4, a], [a, 4]]
data = np.random.multivariate_normal(mean, cov, 1000)
fig,ax=plt.subplots()
ax.set_xlabel("Feature 1")

ax.set_ylabel("Feature 2")
ax.scatter(data[:, 0], data[:, 1], s=3)
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
st.pyplot()
    
    
st.write("---")
