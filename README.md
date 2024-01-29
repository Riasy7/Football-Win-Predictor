# Football-Win-Predictor

Before hopping into the README, please keep in mind that this program is a basic machine learning model, as I did it I learned a lot and I plan to make it more accurate and add more predictors in the future. Note that the explanations are all things I learned through various sources, such as the pandas and scikit-learn libraries. All the concepts mentioned will be explained very precisely throughout the README instead of having a bunch of unecessary comments in the code.

## Overview

This repository contains a Python program for predicting English Premier League football match outcomes using machine learning for the 2020-2022 seasons. The primary goal is to predict the winner of a football match based on various predictors. This is done by utilizing the pandas library for data manipulation and scikit-learn for implementing Random Forest Classifier, a popular algorithm.

Note that the data used in this program is a more organized and concise version of the data that was scraped using my football data scrapping program. Visit it in this repo: https://github.com/Riasy7/Football-Stats-Data-Scraping

### Here are some key steps for this program.

**(skip to results if you don't want to have a detailed explanation of each part of the program)**

  - First begin by taking a look at our data by displaying the first 5 rows of the dataframe. This helps us understand the dataframe and how we should go about creating predictors.
  - By displaying our data, we notice that some of the data we need is an object data type. When working with machine learning, we usually need to use numerical data, so float64 and int64. In this case we need to convert some of the data, because machine learning typically doesn't work well with object data types.
  - So, we convert the date column to a datetime data type, which overrides the existing column with the date time.

      **Predictors**
    
      - Predictors: to predict the outcome of the matches, since it is a football match, we need to decide what data we use as predictors. I chose specifically things that affect a football match. The list goes: venue, opponent, hour, day of the week and target.
        - Venue: The venue is when a team is away or home. If you follow football/soccer, this is a really important factor in who is going to win.
        - Opponent: Each opponent now has a numerical code to specify who each team is playing against. This is one of the most important factors.
        - Hour: We did a simple regex string manipulation to be able to have only the hour. For example if a match starts at 16:30, removed :30 from it and kept 16.
        - Day: Each day of the week now has a code.
        - Target: If a team wins, the code for that is 1, if a team loses the code for that is 0, and the same for drawing; 0. I did this because it simplifies things since we only want to predict if a team is going to win.


  **RandomForestClassifier**
  
  - Now, we can start training our machine learning model. We use *RandomForestClassifier*, this is a model I imported from scikit-learn. This model is really powerful as is really useful.
    - Here's a few things to know about RandomForestClassifier:
      - A random forest is a number of decision trees, and each decision tree has slightly different params.
        <img src="https://miro.medium.com/v2/resize:fit:1358/1*i69vGs4AfhdhDUOlaPVLSA.png" alt="Terms to Understand">
        
      - Let's say a team has the opponent code of 18, and another is 14, this doesn't mean that it's more difficult to win if you're 18. They are just values we set for different opponents. Now this is where random forest comes in, a linear model wouldn't have picked this fact up, whereas random forest does pick it up.
      - Initializing random forest: `rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)`
        - The longer the number in `n_estimators` means the longer the algorithm might take to run, but this also means that the algorithm will be more acc.urate. So it's up to you to test it out and find the best number for your algorithm.
        - For `min_samples_split` its basically the number of samples we want to have in a leaf of the decision tree before we actually split the node
        - The higher the `min_samples_split` number is the less likely to overfit, but the lower the accuracy on the training data will be. You can play around with it to see what works for you.
        - A random forest has a lot of random params in it, so if we set a random state we can get the same results as long as the data is the same.

  **Training the model**

  -This part is the most interesting part in my opinion.
    ```
    train = matches[matches["date"] < '2022-01-01'] 
    test = matches[matches["date"] > '2022-01-01']
    ```
    - As you can see, we split up a train and test. The reason why we do this is really important:
      - Let's say we want to evaluate an algorithm and we see how well it performed or the error rate and what not on the same exact data we trained it on. This would result in a perfect algorithm that would predict the exact winning and losing. But this defeats the whole purpose. This is like taking an exam and having the answers to the exam, you will get 100% on the exam, but that doesn't mean you know the material.
      - So if we were to test our model on different data, it would do horribly.
    - Hence, this is why we split it, we want to see how well our model performs on data that it has never seen before!!! Because we want it to predict future outcomes.
    
  - Now we pass in the predictors to the random forest model and the next step would be to determine the accuracy of our model and see where to go from there. To do so we import accuracy_score from scikit-learn

  **accuracy score**

  - accuracy score is a metric that will basically tell u that if u predicted a win, what percentage of the time did the team actually win, and the same goes for losing. Bassically what % of the time did u get it right.
    to calculate the accuracy we simply run `acc = accuracy_score(test["target"], preds)` and print `acc`. If you followed the code, you will notice we get 0.6123188405797102, which is pretty good, when we predict something, 61$ of the time it actually happens.

  - However, when we check what had high accuracy and what had low accuracy. We notice that we predicted losses and draws with a great accuracy, however we predicted the wins more incorrectly then correctly. Which is ironic because we want to predict the wins more.
  - Since we predicted a win more wrong often more than right, we need to go back and see what we can fix.

  To do this we use precision score from scikit-learn

  **precision score**

  - precision score is a metric from scikit-learn that will basically tell us when we predicted a win what percentage of the time did the team actually win.
  - We notice our precision was about 47% right, which is pretty bad. So we improved this.
  - We did this by creating 1 data frae for every squad in our data, and with that we creating a rolling averages function.

  **rolling averages**
  - a rolling average is basically a way to statistically calculate flunctuations in a dataset by making calculating the average and creating a subset which results in smoother data.
  - our rolling averages function calculates rolling averages, for specific colums (specfically `["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]`), over a 3 match window. These averages aim to provide a better represnetation of a teams recent performance.
  - If you take a look at the function, it uses sorting, simple rolling averages calculations, new columns assignment, and handling all the missing values.
    - If you notice, at this part: `rolling_stats = group[cols].rolling(3, closed='left').mean()`, we assign 'left' to closed, cause otherwise pandas will make it so that the model will use the future knowledge. Assigning it left will create rolling averages for the 3 previous weeks.
    - We also drop the missing values because machine learning cannot always pass in missing values.
  - Now we call the function, and we finally have rolling averages for each match.

  **new predictors**
  - now that we have made these new predictors with rolling averages function, we can make a new set of predictions using these predictors
  - we repeat the same thing we did with the train and test split like explained before, the only difference here is the new smooth data.
  - Finally we get he result of a precision of 62.5%, which is a lot better than 47%.

  **data clean up**
  - we end the code of with a bit of data clean up like the team names are different sometimes.
  - We want to normalize team names for example wolves and wolverhampton wanderers are the same team, so we want to combine them into one team, we'll do this by creating a dictionary and use pandas map function


### Results

The machine learning model achieved a precision score improvement from 47% to 62.5%, enhancing the accuracy of match outcome predictions. The code provides a foundation for further exploration and improvement of the prediction model.

**Things I plan on doing:**
- In the near future, I'd like to improve the accuracy by doing many things such as adding more predictors, trying a different model, different params, different time period or even longer time period.
- Another really cool thing I'd love to test out is seeing how the precision and accuracy would perform compared to other leagues than the English Premier League. Or even, a completely different sport!

Thank if you read through this long README, I hope it helped you understand the basics of machine learning. I hope to learn a lot more and implement this same thought process for a lot of different datasets.
  
