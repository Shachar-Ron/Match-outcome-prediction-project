# Match-outcome-prediction-project

The goal of this project is to predict match outcomes of European football matches. Based on the dataset provided on kaggle.com that includes basic match data, FIFA player statistics and bookkeeper data, I built a model to predict the probability of each match outcome – win, draw, or defeat.

In attempting to solve this problem, I apply feature transformation and dimensionality reduction techniques so as to increase the quality of the feature space. I compare multiple classification algorithms and choose the one that performs best on a separate test dataset. Also, I apply probability calibration methods based on isotonic regression to increase the quality of class probability estimates of my classifier. Lastly, I simulate making bets using my prediction model on the test set and observe the resulting return on investment. The optimal solution would be a classification algorithm with better performance than bookkeeper predictions and a betting strategy powered by said prediction algorithm that achieves positive returns on investment when betting on football matches.

![KNN](https://user-images.githubusercontent.com/57832721/85788455-aa81bb80-b735-11ea-8504-8092b1f64572.PNG)

![LogicRegression](https://user-images.githubusercontent.com/57832721/85788834-3f84b480-b736-11ea-8d6d-b2642a75c321.PNG)

![RandomForest](https://user-images.githubusercontent.com/57832721/85788865-4d3a3a00-b736-11ea-991d-91b868dcab74.PNG)

![GaussianNB](https://user-images.githubusercontent.com/57832721/85788896-588d6580-b736-11ea-8e97-59d579dd3e1a.PNG)

![precision-recall](https://user-images.githubusercontent.com/57832721/85788929-680cae80-b736-11ea-80c3-923713fea982.PNG)




