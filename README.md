# MarketGAN
#### Implementing a Generative Adversarial Network (GAN) on the stock market through a pipeline on Google Colab. Data used from 500 Companies from S&P500, downloaded by Alpha Vantage, and trained using a 3-Layer Dense Network as the Generator and a 3-Layer Convolutional Neural Network as the Discriminator.

#### [An Example Colab Notebook](https://github.com/kah-ve/MarketGAN/blob/master/Stock_Market_GAN_11_29_(s%26p_500%2C_50k_epochs%2C_20_history%2C_5_days_ahead%2C_1_pct_change).ipynb) 

#### Abstract

Neural networks have been advancing in capability very rapidly in recent years. One of the newest techniques with these networks is Generative Adversarial Networks. In this GAN architecture you have two neural networks pitted against each other, one trying to fool the other with noise, while the other trains on real data and responds with information on how to make that noise more realistic. After many runs, you would ideally be able to generate data that the other network wouldn't know was real or fake. We aim to implement this powerful method in the modeling of time series data, with our current medium being the stock market. A GAN that is able to work well with time series data, especially chaotic ones such as the market, would be very useful in many other areas. One is finance where you can better predict the risk in an investment, but another application might be in effectively anonymizing private sensitive data. A lot of data today is not shared because of confidentiality, so being able to generate accurate synthetic versions without loss of information would be useful.

#### Setup

Built a pipeline on Google Colab (offers a free K80 GPU for 12-hour sessions). Can be tedious to setup but works like a charm after since I could access it anywhere, change some parameters, and train a model. Some variables I played around with are different lists of companies, number of epochs, days to predict with, days to predict ahead, and threshold of percentage change to predict on. I built on top of code found on Github and added many modifications, some of which are adding methods to stop training and view confusion matrices, streamlining the process to deploy files for predictions, adapting the code to work with Google Colab, and allowing for prompt parameter changes.

#### Results

Using 30 company stocks based on highest market cap as I initially planned turned out to be completely unhelpful. Moved on to S&P 500 because it accounted for 80% of movement in the market. Shown below are the GAN results from the S&P 500 companies, 20 historical days, 5 days ahead predictions, 50k epochs, and various levels of percentage threshold. For predictions of simply up or down (0% threshold), the GAN has decent results, though a CNN I also trained alongside it (not shown below) still came out slightly better. In terms of predicting a 10% change, the GAN does quite badly. It seems to predict most of the down movements correctly but almost none of the up. Loosening the threshold to 1%, we can see that there is actually a significant change in up predictions compared to the 10% threshold. We are now predicting 14% of the true ups rather than just 1% of them, while losing very little of the accuracy in down predictions.

![Results](https://github.com/kah-ve/MarketGAN/blob/master/GANResults.PNG) 

#### Future Work

I've barely scratched the surface with what is possible with GANs. This has mostly been setting up the framework and data pipeline. There can be a lot of improvement in terms of the type of layers and the depth of the layers that are used. I can look at different indicators to include in the training (instead of just the open, close, and volume of each stock), different parameters to train against, different selections of stocks. If I continue to apply the GAN in this field, my main goal next semester is to build it using recurrent neural networks for both the generator and discriminator. Also I want to look into the method of adversarial feature learning which is what's being used currently, and see if I can find a different way to make predictions. 

#### Sources: 

Modified and reused code from https://github.com/MiloMallo/StockMarketGAN which was sourced from https://github.com/nmharmon8/StockMarketGAN. 
