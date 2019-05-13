'''

Name :- Siddharth Nahar
Entry No :- 2016csb1043
Date :- 11/9/18
Purpose :- Extract Features data and create vectors

'''
import numpy as np
from numpy.linalg import inv
import math
import random
import matplotlib.pyplot as plt

#-----------------------------------------------------------

#Function to Plot data of X,Y classes
def plot(X,Y,t):

	#Plot Data Structure for porper understanig
	fig = plt.figure(1)
	countx = 0
	county = 0
	for i in range(0,X.shape[0]):

		if(Y[i] == 1):
			if(countx == 0):
				plt.plot(X[i][0],X[i][1],'bs',label = "Positve")
			else:
				plt.plot(X[i][0],X[i][1],'bs')

			countx += 1
		else:
			if(county == 0):
				plt.plot(X[i][0],X[i][1],'r^',label = "Negative")
			else:
				plt.plot(X[i][0],X[i][1],'r^')
			county += 1

	#Providing more Attributes
	plt.legend(loc = "upper right")
	plt.title(t)

#--------------------------------------------------------------

#------------------------------------------------------------

#Function to get Sigmoid Fucntion
def sigmoid(Z):
	
	#Zdash = Z.astype(dtype=np.float128)

	return 1/(1 + np.exp(-Z))

#------------------------------------------------------------

#Function for Cost calculation

def costFunc(X,Y,W,lamb):

	#To chech if our gradient descent converges after each iteration

	f = sigmoid(X@W)
	e = 0.00001
	f = f+e

	#Formula for Cost of Reularized Hypothesis Logistic Regression 
	cost = (np.log(f)).transpose()@Y + (np.log(1-f)).transpose()@(1-Y)
	cost = -1*cost/X.shape[0]

	square = W**2

	weights = (np.sum(square) - square[0])*lamb
	weights = weights/(2*X.shape[0])

	cost = cost + weights

	return cost

#------------------------------------------------------------
#Function for Grad
def grad(X,Y,W,lamb):


	#Gradient = X'(f-Y) + lam*W
	f = sigmoid(X@W)
	Xt = X.transpose()
	gr = f-Y

	gr = Xt@gr
	gr = gr + lamb*W

	#We don't regularize w0 as prerequired basis
	gr[0] = gr[0] - lamb*W[0]

	gr = gr/X.shape[0]

	return gr
	
#-------------------------------------------------------------
def evalFunc(X,Y,W):

	#Calculation of Error of Hypothesis based on scale 1
	f = sigmoid(X@W)

	f = np.round(f)

	error = f - Y

	err = np.sum(error**2)

	return err/X.shape[0]

#--------------------------------------------------------------
#Function for Gradient Descent

def gradientDescent(X,Y,W,nIter,alpha,lamb):

	Wprev = W
	for i in range(0,nIter):
		
		error = costFunc(X,Y,Wprev,lamb)
		
		if(math.isnan(error) == True):
			break
		
		print("After iteration : "+str(i) + " ,Cost Evaluation for Convergence = "+str(error))
		
		#Upgrade W by calculation of gradient
		Wnew = Wprev - alpha*grad(X,Y,Wprev,lamb)

		#If W doesnt change much it has converged so break
		if(np.sum(np.absolute(Wnew-Wprev)) <= 0.00000000005):
			break

		#print(Wnew)
		Wprev = Wnew

	return Wprev	

#--------------------------------------------------------------
#Function to Return Inverse of f''

def hurestic(X,Y,W,lamb):

	#Calculate hypothesis value
	f = sigmoid(X@W)

	f = f + 0.000001
	R = np.identity(X.shape[0])
	Xt = X.transpose()

	fdash = f*(1-f)

	#Create X'RX create R as diagonal matrix
	for i in range(0,X.shape[0]):
		R[i][i] = fdash[i]

	#For Regularized f'' = lam*Identity
	Id = np.identity(X.shape[1])

	#We don't regularize w0 as it's basis
	Id[0][0] = 0


	#H = (X'RX + lam*I)^-1
	H = R@X
	H = Xt@H

	H = H + lamb*Id	

	H = H/X.shape[0]
	H = inv(H)
	
	return H

#----------------------------------------------------------------
#Function for Newton Raphson method
def newtonRaphson(X,Y,W,nIter,lamb):

	#print(Y.shape)
	Wprev = W
	for i in range(0,nIter):
		
		error = costFunc(X,Y,Wprev,lamb)

		#If error is nan that is sigmoid returns zero
		if(math.isnan(error) == True):
			break
		print("After iteration : "+str(i) + " ,Cost Evaluation for Convergence = "+str(error))
		
		#print("After iteration : "+str(error))
		
		#Calculate inverse of double differential of f
		H = hurestic(X,Y,Wprev,lamb)
	
		#Calculate Gradient of f
		G = grad(X,Y,Wprev,lamb)
		
		#Update W for next iteration
		Wnew = Wprev - H@G

		#If W doesnt change much it has converged so break
		if(np.sum(np.absolute(Wnew-Wprev)) <= 0.00000000005):
			break

		#print(Wprev)
		Wprev = Wnew

	return Wprev		
		

#--------------------------------------------------------------

#Function to create Polynomial Degrees

def kernel(X,degree):

	#print(X.shape)
	Rows = X.shape[0]
	Cols = X.shape[1]
	featureVector = list()
	for i in range(0,Rows):

		j = 0
		
		#Create Ascending Power of polynomial and store in list
		vector = list()
		while(j <= degree):
			k = j
			while(k >= 0):
				vector.append((X[i][0]**k)*(X[i][1]**(j-k)))
				k = k - 1
			j = j + 1

		#print(vector)
		featureVector.append(vector)
	
	#Convert List to numpy array
	Xnew = np.array(featureVector)
	return Xnew
#-------------------------------------------------------------
#Function to plot decision boundary
def plotDecisionBoundary(X,Y,W,degree,t):

	#print(X)

	#If Linear Boundary is to be Plot
	if(X.shape[1] <= 3):

		#Plot by only two points Calculations
		plot(X[:,1:3],Y,"")
		plot_x = [np.min(X[:,1]),np.max(X[:,1])]

		plot_x = np.array(plot_x)
		plot_y = W[1]*plot_x + W[0]
		plot_y = -1*plot_y/W[2]

		#Setting other Attrivutes of plot
		plt.plot(plot_x,plot_y,'--b')
		plt.gca().set_ylim([0,8])
		plt.title(t)
		#plt.savefig("Newton_Raphson_Linear_Boundary")
		plt.show()

	#For Polynomial Degree Plot
	else:

		#Plot for Polynomial Degree Using Contour Plot to plot
		plot(X[:,1:3],Y,"")

		#Create x1,x2 values by linspace and Calculate Y for each
		u = np.linspace(0.01,6.0,50)
		v = np.linspace(0.01,8.0,50)

		z = np.zeros((50,50),dtype = float)

		for i in range(0,50):
			for j in range(0,50):
				a = [u[i],v[j]]
				a = np.array(a)
				a = np.reshape(a,(1,2))
				z[i][j] = kernel(np.array(a),degree)@W

		z = z.transpose()

		#Plot the Contour
		plt.contour(u, v, z, 0, linewidths = 2,linestyles='dashed')
		plt.title(t)
		#plt.savefig("Sample")
		plt.show()
#-------------------------------------------------------------
#Function to Run the Logistic Algorithm

def run(featureMatrix):

	#Create featureMatrix tuple for extracting Rows and Cols	
	tuples = featureMatrix.shape

	deg = input("Enter the Degree of Polynomial To Fit,Enter Polynomial <=4  : ")

	lam = input("Input 0 for Overfitt and 1 for Underfitt : ")

	deg = int(deg)
	lam = int(lam)

	if(lam > 1):
		lam = 1
	#if deg < 4 alpha = 0.01 else apha = 0.00006

	#Initialise Degree of Polynomial to Fit model
	#deg = 1
	#Create X vector and Y vector
	featureMatrixX = featureMatrix[:,0:tuples[1]-1]

	featureMatrixY = featureMatrix[:,-1]

	#Plot DataSet for Visualization
	plot(featureMatrixX,featureMatrixY,"Representation of Data")
	plt.show()
	plt.clf()

	Y = featureMatrixY	

	#Create Y Vaector to single column
	Y = np.reshape(Y,(tuples[0],1))

	X = kernel(featureMatrixX,deg)

	Row,Cols = X.shape
		
	
	#Initialise W by random Values
	W = np.random.rand(Cols,1)
	
	W = W*0.0002 - 0.0001

	#------------------------------------------------------------------------
	#For Gradient Descent We have different Alphas so:
	alpha = 0.001
	if(deg >= 4):
		alpha = 0.00006

	if(deg > 4):
		print("Adjusting parameters for Gradient descent for such high polynomial is Difficult Please enter value less than 5")
		exit()
	else:
		Wnew = gradientDescent(X,Y,W.copy(),100000,alpha,lam)

	err = str(evalFunc(X,Y,Wnew))
	print("After Gradient Descent Error on DataSet :- " + err )	

	Title = "Gradient Descent ," + "Degree = "+str(deg)+"\n" + "Alpha = " + str(alpha) +",Regualrization Params : "+ str(lam) +"\nError : "+err 
	plotDecisionBoundary(X,Y,Wnew,deg,Title)

	plt.clf()
	#----------------------------------------------------------------------------
	#For Newton Raphson
	Wnew = newtonRaphson(X,Y,W.copy(),1000,lam)

	err = str(evalFunc(X,Y,Wnew))
	print("After Newton Raphson Error on DataSet :- " + err)	
	Title = "Newton Raphson ," + "Degree = "+str(deg)+"\n" + "Regualrization Params : "+ str(lam) + "\nError : "+err 
	plotDecisionBoundary(X,Y,Wnew,deg,Title)

	#------------------------------------------------------------------------------
#------------------------------------------------------------

if __name__ == '__main__':

	#Feature Extraction from linregdata file
	logiRegData = open('credit.txt','r')

	featureVector = list()
	#Traverse through file and create feature vector

	
	for line in logiRegData:

		line = line.split(',')
		
		vector = list()

		for data in line:

			val = float(data)
			vector.append(val)

		featureVector.append(vector)

	#Create Numpy array from feature vector
	featureMatrix = np.array(featureVector)
	run(featureMatrix)
