# ANN-HSLFGWO
This is my implementation of a hybrid ANN model optimized by Hyperbolic secant and Levy Flights based Grey Wolf Optimization. (Main file is ANN_levyFlights_..)

The main aim of this project is to predict the scaled sound pressure level emitted by airfoils. The UCI airfoil dataset was used to train and test the model. Mealpy package was used to build the basic models such as ANN-GWO, ANN-SBO, etc. The other models such as ANN-HGWO, ANN_LevyFlights_HGWO were built from scratch using packages such as numpy and pandas. The model currently matches the performance of ANN with stochastic gradient descent. Future work will attempt at modifying the equations used to obtain values X1, X2 and X3 in classic GWO in hopes of significantly outperforming ANN-SGD. I have included my project report and the latex template for the report in case if any of you need it. 
