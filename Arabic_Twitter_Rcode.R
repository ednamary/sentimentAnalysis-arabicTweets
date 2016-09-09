##### Classification using Naive Bayes --------------------

## Example: Filtering sentiment twitter messages ----
## Step 2: Exploring and preparing the data ---- 

# read the twitter data into the twitter data frame
install.packages("xlsx")
library(xlsx)
twitter<-read.xlsx("C:\\Users\\Mad Max\\Documents\\R_Homework\\Arabic_Sentiment_Twitter_Data.xlsx",1,stringsAsFactors = FALSE,header = FALSE,encoding = "UTF-16LE")

# examine the structure of the twitter data
str(twitter)
summary(twitter)

# convert Positive/Negative to factor.
twitter$X2 <- factor(twitter$X2)

# examine the X2 variable more carefully
str(twitter$X2)
table(twitter$X2)

# build a corpus using the text mining (tm) package
install.packages("NLP")

library(tm)
twitter_corpus <- VCorpus(VectorSource(twitter$X1))

# examine the twitter_corpus
print(twitter_corpus)
inspect(twitter_corpus[1:2])

#viewthe actual message text
as.character(twitter_corpus[[2]])
lapply(twitter_corpus[1:2], as.character)

# clean up the corpus using tm_map()
twitter_corpus_clean <- tm_map(twitter_corpus, content_transformer(tolower))
-------------------------------------------------------------------------------------
# show the difference between twitter_corpus and twitter_corpus_clean
as.character(twitter_corpus[[1]])
as.character(twitter_corpus_clean[[1]])
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
twitter_corpus_clean <- tm_map(twitter_corpus_clean, removeNumbers) # remove numbers
as.character(twitter_corpus_clean[[1]])
# some characters are removed, compare the before and after
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  
  # remove punctuation 
twitter_corpus_clean <- tm_map(twitter_corpus_clean, removePunctuation) 
as.character(twitter_corpus_clean[[1]])
# some characters are removed, compare the before and after
-------------------------------------------------------------------------
  
# illustration of word stemming
library(SnowballC)
wordStem(c("learn", "learned", "learning", "learns"))

twitter_corpus_clean <- tm_map(twitter_corpus_clean, stemDocument)
as.character(twitter_corpus_clean[[1]])
# eliminate unneeded whitespace
twitter_corpus_clean <- tm_map(twitter_corpus_clean, stripWhitespace) 
as.character(twitter_corpus_clean[[1]])
------------------------------------------------------------------------------
  
  # examine the final clean corpus
  lapply(twitter_corpus[1:3], as.character)
lapply(twitter_corpus_clean[1:3], as.character)  

# create a document-term sparse matrix
twitter_dtm <- DocumentTermMatrix(twitter_corpus_clean)

str(twitter_dtm)

# creating training and test datasets
#stratified sampling using CARET package that gurantees random partitions with same proportion of each class
library(caret)

in_train<-createDataPartition(twitter$X2,p=0.75,list = FALSE) #in_train vector indicates row numbers included in the training sample

twitter_dtm_train <- twitter_dtm[in_train, ]
twitter_dtm_test  <- twitter_dtm[-in_train, ]

# also save the labels
twitter_train_labels <- twitter[in_train, ]$X2
twitter_test_labels  <- twitter[-in_train, ]$X2

# check that the proportion is similar
prop.table(table(twitter_train_labels))
prop.table(table(twitter_test_labels))

# word cloud visualization
library(wordcloud)
wordcloud(twitter_corpus_clean, min.freq = 50, random.order = FALSE)

# subset the twitter data into Negative and Positive groups
Negative <- subset(twitter, X2 == "Negative")
Positive  <- subset(twitter, X2 == "Positive")

wordcloud(Negative$X2, max.words = 40, scale = c(3, 0.5))
wordcloud(Positive$X2, max.words = 40, scale = c(3, 0.5))
----------------------------------------------------------------
sms_dtm_freq_train <- removeSparseTerms(sms_dtm_train, 0.999)
sms_dtm_freq_train
-------------------------------------------------------
  
 # indicator features for frequent words
findFreqTerms(twitter_dtm_train, 5)

# save frequently-appearing terms to a character vector
twitter_freq_words <- findFreqTerms(twitter_dtm_train, 5)
str(twitter_freq_words)

# create DTMs with only the frequent terms
twitter_dtm_freq_train <- twitter_dtm_train[ , twitter_freq_words]
twitter_dtm_freq_test <- twitter_dtm_test[ , twitter_freq_words]

# convert counts to a factor
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

# apply() convert_counts() to columns of train/test data
twitter_train <- apply(twitter_dtm_freq_train, MARGIN = 2, convert_counts)
twitter_test  <- apply(twitter_dtm_freq_test, MARGIN = 2, convert_counts)

## Step 3: Training a model on the data ----
library(e1071)
twitter_classifier <- naiveBayes(twitter_train, twitter_train_labels)


## Step 4: Evaluating model performance ----
twitter_test_pred <- predict(twitter_classifier, twitter_test)
table(twitter_test_pred)

library(caret)
confusionMatrix(twitter_results$predict_type,twitter_results$actual_type,positive="Positive")
--------------------------------------------------------------------------------------------------
library(gmodels)
CrossTable(twitter_test_pred, twitter_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
----------------------------------------------------------------------

#obtain predicted probabilities

twitter_prob<- predict(twitter_classifier,twitter_test, type="raw")
str(twitter_prob)
str(twitter_test)
head(twitter_prob)

# combine the results into a data frame


twitter_results<- data.frame(actual_type=twitter_test_labels,
                                             predict_type=twitter_test_pred,
                                             prob_Negative=round(twitter_prob[,1],5),
                                             prob_Positive=round(twitter_prob[,2],5))
head(twitter_results)

# Create ROC curve. Pain in the ass. Need help. 


install.packages("ROCR")

library("ROCR")

#Creating a prediction object for airline_all_clean_model (predictor), this object/function is used to transform the input data(data frame,vector etc) into standardized format

pred<-prediction(predictions=twitter_results$prob_Positive,labels=twitter_results$actual_type)


#Creating a performance object from the prediction object

perf<-performance(pred,measure ="tpr", x.measure ="fpr")



plot(perf,main="ROC curve for Negative",col="Red",lwd=2)



abline(a=0,b=1,lwd=2,lty=2)

## Step 5: Improving model performance ----
twitter_classifier1 <- naiveBayes(twitter_train, twitter_train_labels, laplace = 1)
twitter_test_pred1 <- predict(twitter_classifier1, twitter_test)
CrossTable(twitter_test_pred1, twitter_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
---------------------------------------------------------------------
twitter_classifier2 <- naiveBayes(twitter_train, twitter_train_labels, laplace = 2)
twitter_test_pred2 <- predict(twitter_classifier1, twitter_test)
CrossTable(twitter_test_pred2, twitter_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))