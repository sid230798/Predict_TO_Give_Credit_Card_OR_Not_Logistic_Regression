----------------------------------------------------------------

Name :- Siddharth Nahar
Entry No :- 2016csb1043
Date :- 27/8/18

Sys Requirement :- Python 3.x,Anaconda 3.x

----------------------------------------------------------------

/* To Run the Code */

Usr:/~Dir cd LogisticRegression
Usr:/~Dir python logisticRegression.py

Input : 

	1. Degree of Polynomial for Decision Boundary(Degree <= 4)
	2. Want to Overfitt over Underfitt on given polynomial degree

Output :

	*It first Evaluates for Gradient Descent:-
		-Iterates for 100000 times
		-While Iterating it prints Cost after each iteration for convergence test
		-Show the Decision Boudary in graph with Error and Other Info

	*Then it Iterates for Newton Raphson
		-Iterates for 10 times
		-Show the Decision boundary with Error

-------------------------------------------------------------------

*For High Degree Polynomial 

	Both methods more depends on initial parameters so I have optimized till 4 degree Polynomials more
	could be trained but by changing parameters
	
	-Some high degree are shown in Report
