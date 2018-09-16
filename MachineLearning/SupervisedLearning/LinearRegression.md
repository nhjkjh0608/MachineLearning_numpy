#### LinearRegression  
###### 1. Ordinary Least Squares
Data num : n, assumes variance is constant
>![qwe](https://latex.codecogs.com/gif.latex?%5Cmathit%7B%5CTheta%7D%20%3D%20%5Cmathrm%7B%28X%5E%7BT%7DX%7D%29%5E%7B-1%7D%5Cmathrm%7BX%5E%7BT%7D%7D%5Cmathrm%7By%7D)

>![P](https://latex.codecogs.com/gif.latex?P%20%3D%20X%28X%5E%7BT%7DX%29%5E%7B-1%7DX%5E%7BT%7D)

>![L](https://latex.codecogs.com/gif.latex?L%20%3D%20I_%7Bn%7D%20-%20%5Cmathbf%7B11%7D%5E%7BT%7D/n)

>![R](https://latex.codecogs.com/gif.latex?R%5E%7B2%7D%20%3D%20%5Cfrac%7By%5E%7BT%7DP%5E%7BT%7DLPy%7D%7By%5E%7BT%7DLy%7D)

##### 2. Generalized Least Squares
V : covariance matrix
>![theta](https://latex.codecogs.com/gif.latex?%5Cmathit%7B%5Ctheta%7D%20%3D%20%28X%5E%7BT%7DV%5E%7B-1%7DX%29%5E%7B-1%7DXV%5E%7B-1%7Dy)
