# MarketGAN
#### Implementing a Generative Adversarial Network (GAN) on the stock market through a pipeline on Google Colab. Data used from 500 Companies from S&P500, downloaded by Alpha Vantage, and trained using a 3-Layer Dense Network as the Generator and a 3-Layer Convolutional Neural Network as the Discriminator.

Update 05/09/2021: Updated the notebook to remove deprecated functions, removed some code clutter, added instructions, updated stock data to sort correctly from newest to oldest to avoid biased training, and updated readme with extra instructions below.

#### Abstract

Neural networks have been advancing in capability very rapidly in recent years. One of the newest techniques with these networks is Generative Adversarial Networks. In this GAN architecture you have two neural networks pitted against each other, one trying to fool the other with noise, while the other trains on real data and responds with information on how to make that noise more realistic. After many runs, you would ideally be able to generate data that the other network wouldn't know was real or fake. We aim to implement this powerful method in the modeling of time series data, with our current medium being the stock market. A GAN that is able to work well with time series data, especially chaotic ones such as the market, would be very useful in many other areas. One is finance where you can better predict the risk in an investment, but another application might be in effectively anonymizing private sensitive data. A lot of data today is not shared because of confidentiality, so being able to generate accurate synthetic versions without loss of information would be useful.

#### Setup

Built a pipeline on Google Colab (offers a free K80 GPU for 12-hour sessions). Can be tedious to setup but works like a charm after since I could access it anywhere, change some parameters, and train a model. Some variables I played around with are different lists of companies, number of epochs, days to predict with, days to predict ahead, and threshold of percentage change to predict on. I built on top of code found on Github and added many modifications, some of which are adding methods to stop training and view confusion matrices, streamlining the process to deploy files for predictions, adapting the code to work with Google Colab, and allowing for prompt parameter changes.

#### Results

Using 30 company stocks based on highest market cap as I initially planned turned out to be completely unhelpful. Moved on to S&P 500 because it accounted for 80% of movement in the market. Shown below are the GAN results from the S&P 500 companies, 20 historical days, 5 days ahead predictions, 50k epochs, and various levels of percentage threshold. For predictions of simply up or down (0% threshold), the GAN has decent results, though a CNN I also trained alongside it (not shown below) still came out slightly better. In terms of predicting a 10% change, the GAN does quite badly. It seems to predict most of the down movements correctly but almost none of the up. Loosening the threshold to 1%, we can see that there is actually a significant change in up predictions compared to the 10% threshold. We are now predicting 14% of the true ups rather than just 1% of them, while losing very little of the accuracy in down predictions.

![Results](https://github.com/kah-ve/MarketGAN/blob/master/GANResults.PNG) 

(TODO: Add a screenshot of the model predicting a buy or sell for specific stocks)
-------------------------------------------------------------------------------------

# Update: usage notes

(TODO: Replace this colab notebook with the new cleaned and updated one once that is ready.)
#### [An Example Colab Notebook](https://github.com/kah-ve/MarketGAN/blob/master/Stock_Market_GAN_11_29_(s%26p_500%2C_50k_epochs%2C_20_history%2C_5_days_ahead%2C_1_pct_change).ipynb) 
There are step-by-step instructions below that explain how to use the notebook. The first cell of the notebook also has markdown that covers these steps. Furthermore, you can find explanations in the cell comments as well for extra details.

This google colab will take several hours to run (even with Google's provided GPU). Downloading all the stock data for the 500 companies in S&P can take 2-3 hours by itself.

Then depending on the parameters you are using for training the models, the training can take some hours more.

### What results do I get from this?

This project was mostly to help me get familiar with GANs. I modified the code so that it can run on Google Colab in a very streamlined way with the parameters that I'd like. I think this is a great way to get familiar with GANs and the training process. This code will output a confusion matrix that shows the prediction accuracy rates overall and at the end there is code that will use the model to output predictions for each stock.

### How does it work?

For our solution we are using a convolutional neural network as the discriminator
network and a 3-layered network for the generator. We are using a technique called adversarial
feature learning in which you are additionally attempting to encode the features learned by the
discriminator into the generating distribution. However, what we implemented doesn't follow this
exactly as we are training a boosted tree classifier (using XGBoost) on the weights of the
convolutional layer in the discriminator. Then to predict for different historical days, we would
input that data into the discriminator and get the flattened weights, run them through the
classifier, and finally get the prediction of the price’s movement through the backward
propagation of the weights.

During each training output we
would be saving the different trained models: including the discriminator models (at multiple
epochs), the generator model (at multiple epochs), the benchmark CNN (at multiple epochs),
and the XGBoost model. XGBoost is an open-source software library which provides a gradient
boosting framework, and has performed very well on other machine learning competitions such
as Kaggle. As mentioned before, the XGBoost would be trained on the discriminator network’s
flattened weights after we had already trained the GAN. The assumption (which has been seen
to work in other applications as well) is that the GAN will learn the feature space of the data and
the weights within the discriminator will then have predictive information for how it believes the stock prices will behave. So going forward, when we want to predict on unseen data, we would
send it through the discriminator in the GAN, then get the flattened weights, and then feed those
weights to the XGBoost model we had trained. This would then give us a prediction.

Note: Here I am using the term we since the code for training isn't sourced by me. See source at bottom of README.

# Instructions

The comments explain the steps you need to take. With some minor changes you can run this colab notebook so that it trains a model for you and makes some predictions. 

0. First you need to copy my notebook to Google Colab or locally. There are advantages to using google colab (such as access to their GPU and also being able to make modifications from any computer that has access to the colab notebook) and that's the method I used for training these models. However, you can also set this up locally for your local gpu. Leave the googlepath variable blank or point it to the folder you want for your local jupyter notebook.

### To copy the required jupyter notebook you can do one of these as you prefer. 

* Open a new google colab in google drive (New -> More -> Google Colaboratory) then File -> Open Notebook -> GitHub -> Pass in the URL of the Notebook in this repo. (TODO: Add the url link here)
* Clone this repo, then open new google colab notebook in google drive (New -> More -> Google Colaboratory). Then upload the notebook by File -> Open Notebook -> Upload -> Select the jupyter notebook you cloned from this repo.

### Instructions for getting the notebook running:

1. Run the first two cells and follow the instructions in the second cell with code in it. This will mount the google drive so you can save files and load them from your google drive folder.

2. In the 3rd cell, select a googlepath that you will use to store all your folders and files. The folder structure that you will end up with is shown at the end of these instructions. However, for now just choose a googlepath to use. (e.g. /drive/My Drive/Colab Notebook/MarketGAN

3. Modify the parameters in that same cell for training according to your wishes. These are the only modifications you need to make to be able to train the models. It's very simple to change these parameters to train different models and also change the companylist to try different stocks. (remember to delete old stock data within the stock_data folder)

4. Get an alphavantage api key at the website https://www.alphavantage.co/support/#api-key. This will allow you to download the stock data using their API. 

Note: Alphavantage has a limit of 500 requests per day and also 6 per minute for its free service.

5. Run the code until the section called [***Change the names of the files in the deployed_model folder***](https://colab.research.google.com/drive/1eSjgfS2lEjEZiD4BRvuK2l9GlNFjfNGR#scrollTo=3lPwUmniYB2k&line=1&uniqifier=1). Select the models you want to use from XGB, CNN, and GAN and place them into the deployed_model folder. Run this cell to rename them and prepare them to be used for predictions.

#### **Note:** When training the GAN or CNN models, the training will pick up from where it left off. The model will save every 10,000 steps (by default. you can modify this by changing the TRAINING_AMOUNT variable.) For this to work, you only need to leave the model in the models folder and it will select the latest step one and continue training. Thus, I recommend that when moving the trained models to the deployed_model folder, you do a copy and paste. The script under the heading 'Change the names of the files in the deployed_model folder' will rename the model for you once it's in the deployed_model folder (regardless of copied or not).

6. Run the remaining cells and wait for the output.


### Expected File Directory Layout in Folder at the End

*   cnn models *(where trained cnn models get saved)*

*   deployed_model *(This is the folder that you must MANUALLY put the trained model into after you are satisfied with the number of steps for training. XGB, CNN, and GAN models go into here. There is a script further below [here](https://colab.research.google.com/drive/1eSjgfS2lEjEZiD4BRvuK2l9GlNFjfNGR#scrollTo=3lPwUmniYB2k&line=1&uniqifier=1) which will take those files in deployed_model, rename if need be, and then use them for making predictions.)*
*   logs *(Logs will be added here)*
  *   test
  *   train

*   models *(The models will be added into this directory)*


*   stock_data *(This is the folder that all our stock data will be placed into)*


*   companylist.csv *(this is a company list that you must provide, or use the one I have provided at GitHub. We will download the data for these tickers.)*


*   stockmarketGan.ipynb *(this is this google colab document)*


# MISC Notes

Google Colab uses a K80 GPU which offers 24GB of video ram for training. Training must be limited to under 12 hour sessions or Google Colab shuts down the session. However, the training does not take more than a couple of hours and the models are being continuously saved so nothing will be lost. 

-----------------------------------------------------------------------

#### Future Work

I've barely scratched the surface with what is possible with GANs. This has mostly been setting up the framework and data pipeline. There can be a lot of improvement in terms of the type of layers and the depth of the layers that are used. I can look at different indicators to include in the training (instead of just the open, close, and volume of each stock), different parameters to train against, different selections of stocks. If I continue to apply the GAN in this field, my main goal next semester is to build it using recurrent neural networks for both the generator and discriminator. Also I want to look into the method of adversarial feature learning which is what's being used currently, and see if I can find a different way to make predictions. 

#### Sources: 

Modified and reused code from https://github.com/MiloMallo/StockMarketGAN which was sourced from https://github.com/nmharmon8/StockMarketGAN. 
