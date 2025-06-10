
# Personalized Meal Type Prediction

## Project Background

HelloFresh is a global meal kit delivery company that helps people cook at home by sending fresh ingredients and step-by-step recipes. Since starting in 2011 it has grown to serve millions of customers around the world, focusing on convenience, healthy meals, and options that fit different diets and preferences.

This project explores how to make meal recommendations more personal. Right now, suggestions are usually based on filters customers choose, but there’s room to make the experience smarter. Using recipe and review data from Food.com, the goal was to build a model that can predict what kind of meal—like breakfast, lunch, dinner, snack, or dessert a person might enjoy, helping make meal planning easier and more tailored to individual tastes.

**Insights and recommendations are provided on the following key areas:**

**1. Meal Type Preferences** - Using review and recipe data to identify trends in what types of meals users prefer.

**2. Model Performance** -  Evaluating how accurately we can predict meal types using different classification models.

**3. Personalization Strategy** - How HelloFresh could use the model to enhance customer satisfaction and increase retention.

The Python scripts used to inspect and clean the data for this analysis can be found here: [Python Cleaning Scripts](https://github.com/Abishang21/Personalized-Meal-Type-Prediction/blob/master/EDA.ipynb)

Python scripts used for feature engineering and model evaluation are available here: [Python Evaluation Scripts](https://github.com/Abishang21/Personalized-Meal-Type-Prediction/blob/master/Feature_Engineering_Model_Evaluation.ipynb)



## Data Structure &  Checks

The dataset analyzed in this project was sourced from Kaggle. It consists of  two main datasets Recipes and Reviews. The data provides detailed information about over 500,000 recipes and 1.4 million user reviews: a recipes and a reviews table.

[Kaggle Link](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews?select=reviews.csv)

The two dataset are as follows:

**Recipes Dataset (Key Columns)**

RecipeId, Name, AuthorId, CookTime, Calories, FatContent, ProteinContent.

Includes cooking/prep time, ingredient lists, and nutritional data.

**Reviews Dataset (Key Columns)**

ReviewId, RecipeId, AuthorId, Rating, Review, DateSubmitted.

Includes user ratings and text reviews linked to each recipe.


The datasets were cleaned and transformed using Python (Pandas) before being exported as a pickle file for feature engineering and model evaluation. The key preprocessing steps included:

1.  Created copies of the original files to preserve raw data.

2.  Filled missing CookTime values using the most frequent value (mode).

3.  Converted cooking time values (e.g PT30M) to minutes for consistency.

4.  Converted DatePublished to proper datetime format.

5.  Applied the IQR method to remove outliers from key numeric columns such as Calories, FatContent, and ProteinContent.

6.  Merged the cleaned recipes and reviews datasets on RecipeId.

7.  Saved the final cleaned and merged dataset as a pickle file for further analysis and modeling.


The cleaned and merged dataset.Download it from this link:
    
[Download RecipeInsights.pkl](https://drive.google.com/drive/folders/1PcjDBfoT8hmf_f6l7nJ0E91cAHiw64_9?usp=sharing)




##  Feature Engineering

Several feature engineering steps were carried out to clean, enrich, and structure the data for modeling.

The RecipeCategory column was initially too granular so categories were grouped into broader more meaningful meal types: Breakfast, Lunch, Dinner, Dessert, and Snack. Any category that didn’t fit into these five was temporarily labeled as Other.

Since the “Other” group was too large to ignore, its contents were analyzed further. Common recipe types such as Vegetable, Potato, Chicken, and Beverages were manually reassigned to appropriate meal types using logical judgment. Less frequent entries were split into:

- Other_Major – frequently occurring but uncategorized

- Other_Minor – rare and infrequent

To maintain focus on well-defined and predictable classes both "Other_Major" and "Other_Minor" were filtered out. The final dataset retained only five core meal types as the target variable (MealType):

- Breakfast, Lunch, Dinner, Dessert and Snack

The MealType labels were encoded using LabelEncoder for machine learning compatibility. The resulting mapping was:

- Breakfast: 0 , Dessert: 1, Dinner: 2, Lunch: 3, Snack: 4

Feature selection was guided by correlation analysis. Strong relationships were found between:

- Calories and FatContent (0.84)

- Calories and ProteinContent (0.67)

- CookTime and TotalTime (0.99)

Columns like Rating (constant) and ReviewCount (low correlation) were excluded. Based on these insights the final input features selected were:

- Calories, FatContent, ProteinContent, CookTime

Nutritional content and cook time were identified as key predictors of a recipe’s meal type.


## Model Development and Evaluation


The first model trained was a Logistic Regression model to predict MealType, which includes five categories:
0 = Breakfast, 1 = Brunch, 2 = Dinner, 3 = Snack, 4 = Lunch.

**Logistic Regression Insights**

- Accuracy was around 46%

- The model did a good job predicting Dinner (Class 2) with a recall of about 86%

- It struggled with Breakfast (Class 0) and Snack (Class 3) especially Snack which had only 4% recall

- Precision was highest for Lunch (Class 4) and Snack (Class 3)  so when it predicted these, it was often right

- F1 scores were strong for Dinner and Lunch, but low for Breakfast and Snack

![1.png](https://github.com/Abishang21/Personalized-Meal-Type-Prediction/blob/96e543b93da1cfc7d7977330e4de4312e21292cf/Images/1.png)

**Tackling the Class Imbalance**

The model was favoring the class with the most examples(Dinner). To fix this  both class_weight='balanced' and SMOTE (Synthetic Minority Oversampling Technique) were used.

- Both balancing techniques (class weights and SMOTE) helped improve fairness especially for the smaller classes like Snack and Breakfast but the overall accuracy dropped to 38%

![2.png](https://github.com/Abishang21/Personalized-Meal-Type-Prediction/blob/96e543b93da1cfc7d7977330e4de4312e21292cf/Images/2.png)
![3.png](https://github.com/Abishang21/Personalized-Meal-Type-Prediction/blob/96e543b93da1cfc7d7977330e4de4312e21292cf/Images/3.png)


**Recommendations**

1. With Logistic Regression showing limitations in handling class imbalance and capturing complex patterns in the data, it may not be the most suitable model. Tree-based models like Decision Trees could offer better performance by handling imbalance more effectively and modeling non-linear relationships.


**Decision Tree Insights**

Accuracy was around 61%, which is a clear improvement over Logistic Regression’s 46%, and even better compared to 38% after class balancing.

- Dinner (Class 2) had the best performance with a recall of 82%. The model correctly predicted 21,867 out of 26,789 cases.

- Snack (Class 4) also did well with a precision of 0.82 and recall of 0.67  meaning when it predicted Snack it was mostly right.

- Lunch (Class 3) improved compared to Logistic Regression but there were still many errors.

- Breakfast (Class 0) and Brunch (Class 1) had moderate results but better than before.

![4.png](https://github.com/Abishang21/Personalized-Meal-Type-Prediction/blob/96e543b93da1cfc7d7977330e4de4312e21292cf/Images/4.png)

**Improving Decision Tree Performance**

To boost performance, Hyperparameter Tuning was done using GridSearchCV testing many combinations of parameters to find the best fit.

**Tuned Decision Tree Insights**

Accuracy jumped to 94%, a huge improvement from the previous 61%.

- Breakfast (Class 0) had great performance with 94% precision and recall. The model is now very accurate at identifying breakfast items.

- Brunch (Class 1) had strong results with a 93% F1 score. Much better than before.

- Dinner (Class 2) maintains high accuracy, precision, and recall (95%+).

- Lunch (Class 3) is now much more accurately predicted. Misclassifications have dropped a lot.

- Snack (Class 4) has the highest recall (96%) meaning very few snack items were missed.

![5.png](https://github.com/Abishang21/Personalized-Meal-Type-Prediction/blob/96e543b93da1cfc7d7977330e4de4312e21292cf/Images/5.png)

The tuned model did an excellent job at identifying all classes. Misclassifications are minimal


## Recommendations

1. Push for including a wider range of meal types like Snacks and Desserts in weekly menus to encourage variety and better serve different customer preferences.

2. Continue focusing on Dinner and Lunch meals as they are the most popular choices among users based on current data trends.

3. Re-evaluate older models like Logistic Regression and prioritize tree-based models which handle complex and imbalanced data more effectively.

4. Improve customer meal suggestions by using the model to predict what meal types each user is most likely to enjoy.

4. Push for personalized meal recommendations based on user history to increase satisfaction and keep customers coming back.


