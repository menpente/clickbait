if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(text2vec)) install.packages("text2vec", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(tidytext)) install.packages("tidytext", repos = "http://cran.us.r-project.org")





library(tidyverse)
library(caret)
library(stringr)
library(text2vec)
library(glmnet)
library(ggplot2)
library(tidytext)



df<-read.csv(file='clickbait_data.csv')#load data into a dataframe
df %>% group_by(clickbait) %>% summarise(n=n())#check if data is balanced between both categories
df<-df %>% mutate(clickbait = factor(clickbait))#convert dependent variable to factor

#adding linguistic features to dataframe
df$length=str_length(df$headline)#headline length in characters
df$questionmark=str_detect(df$headline,'\\?')#checking for question mark; ? is escaped with \\ as str_detect looks for regular expressions by default
df$exclamationmark=str_detect(df$headline,'!')#checking for exclamation mark
df$numbers=str_detect(df$headline,'[0-9]')#checking for numbers

#splitting dataframe into train and test
set.seed(42, sample.kind="Rounding")
y<-df$clickbait
test_index <- createDataPartition(y, times = 1, p = 0.2, list = FALSE)
test_set<-df[test_index,]
train_set<-df[-test_index,]

#random guessing baseline
set.seed(3, sample.kind="Rounding")
outcome<-factor(c(0,1))
y_random<-sample(outcome, size = 6401, replace=TRUE)
mean(test_set$clickbait==y_random)

#predicting with glm; using 3-fold cross-validation and training percentage 90%
set.seed(8, sample.kind="Rounding")
fit_glm<-train(clickbait ~ length + questionmark + exclamationmark + numbers, data=train_set, method = "glm", trControl = trainControl(method = "cv", number = 3, p = .9))
y_predict<-predict(fit_glm, test_set, type = "raw")
confusionMatrix(y_predict, test_set$clickbait)$overall['Accuracy']

#check variable importance
varImp(fit_glm)

#explore clickbait and regular news vocabulary
sentences<-train_set %>% filter(clickbait==1) %>% select(headline)#select clickbait sentences
words<-sentences %>% unnest_tokens(output=word, input=headline)#create list of words
words<-words %>% anti_join(stop_words)#remove stop words
word_counts<-words %>% count(word, sort = TRUE)#create word counts
word_counts %>% filter(n > 125) %>% #prepare chart
  mutate(word = reorder(word, n)) %>% 
  ggplot(aes(word, n)) + 
  geom_col() +
  coord_flip() +
  labs(x = "Word \n", y = "\n Count ", title = "Frequent Words In Clickbait \n") +
  geom_text(aes(label = n), hjust = 1.2, colour = "white", fontface = "bold") +
  theme(plot.title = element_text(hjust = 0.5), 
        axis.title.x = element_text(face="bold", colour="darkblue", size = 12),
        axis.title.y = element_text(face="bold", colour="darkblue", size = 12))

sentences<-train_set %>% filter(clickbait==0) %>% select(headline)#select legit news
words<-sentences %>% unnest_tokens(output=word, input=headline)
words<-words %>% anti_join(stop_words)
word_counts<-words %>% count(word, sort = TRUE)
word_counts %>% filter(n > 125) %>% 
  mutate(word = reorder(word, n)) %>% 
  ggplot(aes(word, n)) + 
  geom_col() +
  coord_flip() +
  labs(x = "Word \n", y = "\n Count ", title = "Frequent Words In Legit News \n") +
  geom_text(aes(label = n), hjust = 1.2, colour = "white", fontface = "bold") +
  theme(plot.title = element_text(hjust = 0.5), 
        axis.title.x = element_text(face="bold", colour="darkblue", size = 12),
        axis.title.y = element_text(face="bold", colour="darkblue", size = 12))

#Creating models based on string vectors

# define preprocessing function and tokenization function
prep_fun = tolower#convert to lower case
tok_fun = word_tokenizer#create word lists

it_train = itoken(train_set$headline, #preprocess
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  progressbar = FALSE)
vocab = create_vocabulary(it_train)#create vocabulary

vectorizer = vocab_vectorizer(vocab)#create vectorizer
dtm_train = create_dtm(it_train, vectorizer)#create document-term matrix for train set



it_test = tok_fun(prep_fun(test_set$headline))#preprocess test set set 
it_test = itoken(it_test, progressbar = FALSE)
dtm_test = create_dtm(it_test, vectorizer)

#bag of words
glmnet_classifier = cv.glmnet(x = dtm_train, y = train_set[['clickbait']], 
                              family = 'binomial', 
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = 3)

preds = predict(glmnet_classifier, dtm_test, type = 'class')
print ('bag of words')
confusionMatrix(factor(preds), test_set$clickbait)$overall['Accuracy']

# define tfidf model
tfidf = TfIdf$new()
# fit model to train data and transform train data with fitted model
dtm_train_tfidf = fit_transform(dtm_train, tfidf)
# tfidf modified by fit_transform() call!
# apply pre-trained tf-idf transformation to test data

dtm_test_tfidf = transform(dtm_test, tfidf)

glmnet_classifier = cv.glmnet(x = dtm_train_tfidf, y = train_set[['clickbait']], 
                              family = 'binomial', 
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = 3)

preds = predict(glmnet_classifier, dtm_test_tfidf, type = 'class')
print ('tfidf')
confusionMatrix(factor(preds), test_set$clickbait)$overall['Accuracy']

