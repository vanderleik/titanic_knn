# Titanic: Machine Learning from Disaster

# https://cran.r-project.org/web/views/MachineLearning.html
# https://medium.com/better-programming/titanic-survival-prediction-using-machine-learning-4c5ff1e3fa16

setwd("c:/FCD/PortfolioR/titanic")
getwd()

# title: 'Titanic: Machine Learning from Disaster'
# author: 'Vanderlei Kleinschmidt'
# date: '17 october 2020'

# packages
library(tidyverse)
library(Amelia)
library(patchwork)
library(class)
library(gmodels)

#1. Introdução

#2. Coletando os dados
train <- read.csv('train.csv', stringsAsFactors = F)
test  <- read.csv('test.csv', stringsAsFactors = F)

# Vai ser útil no final
survived <- train$Survived
survived <- as.factor(survived)
passengers <- test$PassengerId

#3. Explorando e preparando os dados para análise

View(train)
str(train)
View(test)
str(test)

# O dataset de treino tem 891 observações, com 12 variáveis, sendo elas:
# PassengerId: integer => é um índice ou identificador de cada passageiro
# Survived   : integer (variável target) => assume valor igual a 0 caso não tenha sobrevivido e valor igual a 1 caso tenha sobrevivido
# Pclass     : integer => é uma proxy para o status econômico e social do passageiro, assumindo valor igua a
## 1 = classe superior; 2 = classe média; e 3 = classe inferior
# Name       : character => nome do passageiro
# Sex        : character => masculino e feminino
# Age        : number => idade do passageiro em anos
# SibSp      : integer => número de irmãos, cuja relação familiar do passageiro no navio é definida como:
## Sibling= brother, sister, stepbrother, stepsister
## Spouse= husband, wife (mistresses and fiancés were ignored)
# Parch      : integer => número de pais/crianças abordo, cuja relação familiar do passageiro no navio é definida como:
## Parent= mother, father
## Child= daughter, son, stepdaughter, stepson
## Some children traveled only with a nanny, therefore parch=0 for them.
# Ticket     : character => número da passagem
# Fare       : number => tarifa do passageiro
# Cabin      : character => cabine
# Embarked   : character => porto de embarque: C = Cherbourg, Q = Queenstown, S = Southampton

# Agrupando e analisando os dados

train$isTrainSet <- TRUE
test$isTrainSet <- FALSE

names(train)
names(test)

ncol(train)
ncol(test)
test$Survived <- NA
ncol(test)

fulldata <- rbind(train, test)
round(prop.table(table(fulldata$isTrainSet)), 4)

# Criando um mapa de dados faltantes

missmap(fulldata, 
        main = "Titanic Training Data - Mapa de Dados Missing", 
        col = c("yellow", "black"), 
        legend = FALSE)

table(is.na(fulldata$Age))


pl <- ggplot(fulldata, aes(Pclass,Age)) + geom_boxplot(aes(group = Pclass, fill = factor(Pclass), alpha = 0.4))

# Vou adicionar uma escala contínua no eixo Y pra melhorar o entendimento do que está sendo visto.
pl + scale_y_continuous(breaks = seq(min(0), max(80), by = 2))

# Cada classe tem idades diferentes, sendo que na primeira classe a média das idades é maior do que nas demais, e a terceira classe é a que tem a menor média de idade.
ageMedian_1 <- fulldata %>% group_by(Pclass) %>% filter(Pclass == 1)
median_1 <- median(ageMedian_1$Age, na.rm = TRUE)

ageMedian_2 <- fulldata %>% group_by(Pclass) %>% filter(Pclass == 2)
median_2 <- median(ageMedian_2$Age, na.rm = TRUE)

ageMedian_3 <- fulldata %>% group_by(Pclass) %>% filter(Pclass == 3)
median_3 <- median(ageMedian_3$Age, na.rm = TRUE)

rotulos <- c("1a classe", "2a classe", "3a classe")
mediana <- c(median_1, median_2, median_3)
medianas <- rbind(rotulos, mediana)
medianas

impute_age <- function(age, class){
  out <- age
  for (i in 1:length(age)){
    
    if (is.na(age[i])){
      
      if (class[i] == 1){
        out[i] <- median_1
        
      }else if (class[i] == 2){
        out[i] <- median_2
        
      }else{
        out[i] <- median_3
      }
    }else{
      out[i]<-age[i]
    }
  }
  return(out)
}

fixed.ages <- impute_age(fulldata$Age, fulldata$Pclass)
fulldata$Age <- fixed.ages

table(is.na(fulldata$Age))


table(is.na(fulldata$PassengerId))
table(is.na(fulldata$Survived))
table(is.na(fulldata$Pclass))
table(is.na(fulldata$Name))
table(is.na(fulldata$Sex))
table(is.na(fulldata$Age))
table(is.na(fulldata$SibSp))
table(is.na(fulldata$Parch))
table(is.na(fulldata$Ticket))
table(is.na(fulldata$Fare))
table(is.na(fulldata$Cabin))
table(is.na(fulldata$Embarked))
table(is.na(fulldata$isTrainSet))

median(fulldata$Fare)


fareMedian_1 <- fulldata %>% group_by(Pclass) %>% filter(Pclass == 1)
median_f1 <- median(fareMedian_1$Fare)

fareMedian_2 <- fulldata %>% group_by(Pclass) %>% filter(Pclass == 2)
median_f2 <- median(fareMedian_2$Fare)

fareMedian_3 <- fulldata %>% group_by(Pclass) %>% filter(Pclass == 3)
median_f3 <- median(fareMedian_3$Fare)

rotulosf <- c("1a classe", "2a classe", "3a classe")
medianaf <- c(median_f1, median_f2, median_f3)
medianasf <- rbind(rotulosf, medianaf)
medianasf

median_f3 <- median(fareMedian_3$Fare, na.rm = TRUE)
median_f3

fulldata[is.na(fulldata$Fare), "Fare"] <- median_f3
table(is.na(fulldata$Fare))

table(fulldata$Survived)
table(fulldata$Pclass)
table(fulldata$Sex)
table(fulldata$Age)
table(fulldata$SibSp)
table(fulldata$Parch)
table(fulldata$Embarked)
table(fulldata$isTrainSet)

fulldata[fulldata$Embarked == "", "Embarked"] <- 'S'
table(fulldata$Embarked)

fulldata[fulldata$Sex == "female", "Female"] <- '1'
fulldata[fulldata$Sex == "male", "Female"] <- '0'
head(select(fulldata, Sex, Female), 30)

fulldata[fulldata$Pclass == "1", "FClasse"] <- '1'
fulldata[fulldata$Pclass == "2", "FClasse"] <- '0'
fulldata[fulldata$Pclass == "3", "FClasse"] <- '0'

fulldata[fulldata$Pclass == "1", "SClasse"] <- '0'
fulldata[fulldata$Pclass == "2", "SClasse"] <- '1'
fulldata[fulldata$Pclass == "3", "SClasse"] <- '0'

fulldata[fulldata$Pclass == "1", "TClasse"] <- '0'
fulldata[fulldata$Pclass == "2", "TClasse"] <- '0'
fulldata[fulldata$Pclass == "3", "TClasse"] <- '1'
head(select(fulldata, Pclass, FClasse, SClasse, TClasse), 30)

fulldata[fulldata$Embarked == "C", "CEmbarked"] <- '1'
fulldata[fulldata$Embarked == "Q", "CEmbarked"] <- '0'
fulldata[fulldata$Embarked == "S", "CEmbarked"] <- '0'

fulldata[fulldata$Embarked == "C", "QEmbarked"] <- '0'
fulldata[fulldata$Embarked == "Q", "QEmbarked"] <- '1'
fulldata[fulldata$Embarked == "S", "QEmbarked"] <- '0'

fulldata[fulldata$Embarked == "C", "SEmbarked"] <- '0'
fulldata[fulldata$Embarked == "Q", "SEmbarked"] <- '0'
fulldata[fulldata$Embarked == "S", "SEmbarked"] <- '1'
head(select(fulldata, Embarked, CEmbarked, QEmbarked, SEmbarked), 30)

head(select(fulldata, Survived), 30)

str(fulldata)

fulldata$Survived <- as.factor(fulldata$Survived)

fulldata$Sex <- as.factor(fulldata$Sex)
fulldata$Female <-  as.factor(fulldata$Female)

fulldata$Pclass <- as.factor(fulldata$Pclass)
fulldata$FClasse <- as.factor(fulldata$FClasse)
fulldata$SClasse <- as.factor(fulldata$SClasse)
fulldata$TClasse <- as.factor(fulldata$TClasse)

fulldata$Embarked <- as.factor(fulldata$Embarked)
fulldata$CEmbarked <- as.factor(fulldata$CEmbarked)
fulldata$QEmbarked <- as.factor(fulldata$QEmbarked)
fulldata$SEmbarked <- as.factor(fulldata$SEmbarked)

str(fulldata)

sobreviventes <- table(train$Survived) # Utilizo os dados de treino porque não tenho a informação de quantos sobreviveram nos dados de teste.
round(prop.table(sobreviventes), 4)

ggplot(train, aes(x = as.factor(Survived))) + geom_bar() + scale_x_discrete()+ theme_minimal()

gender <- table(fulldata$Sex)
round(prop.table(gender), 4)

gender_train <- table(train$Sex)
gender_test <- table(test$Sex)
round(prop.table(gender_train), 4)
round(prop.table(gender_test), 4)

ggplot(fulldata,aes(Sex)) + geom_bar(aes(fill = factor(Sex)), alpha = 0.5)

bySex <- with(train, table(Survived, Sex)) # Dados de treino porque não sabemos quantos sobreviveram nos dados de teste.
bySex

SocEconClass <- table(fulldata$Pclass)
round(prop.table(SocEconClass), 4)

ggplot(fulldata,aes(Pclass)) + geom_bar(aes(fill = factor(Pclass)), alpha = 0.5)

byClass <- with(train, table(Survived, Pclass)) # Utilizo os dados de treino porque não tenho a informação de quantos sobreviveram nos dados de teste.
byClass
round(prop.table(byClass), 4)

age_train <- ggplot(train, aes(Age)) + geom_histogram(fill = 'blue', bins = 20, alpha = 0.5) + ggtitle("Histograma para os dados de treino")
age_test <- ggplot(test,aes(Age)) + geom_histogram(fill = 'blue', bins = 20, alpha = 0.5) + ggtitle("Histograma para os dados de teste")
age_train + age_test

summary(train$Age)
summary(test$Age)
quantile(train$Age, na.rm = TRUE)
quantile(test$Age, na.rm = TRUE)
IQR(train$Age, na.rm = TRUE)
IQR(test$Age, na.rm = TRUE)

#4. Escolhendo o algoritmo de machine learning e treinando o modelo com os dados

ncol(fulldata)
ncol(train)
ncol(test)

train$Female <- NA
test$Female <- NA

train$FClasse <- NA
test$FClasse <- NA
train$SClasse <- NA
test$SClasse <- NA
train$TClasse <- NA
test$TClasse <- NA

train$CEmbarked <- NA
test$CEmbarked <- NA
train$QEmbarked <- NA
test$QEmbarked <- NA
train$SEmbarked <- NA
test$SEmbarked <- NA

ncol(fulldata)
ncol(train)
ncol(test)

train <- fulldata[fulldata$isTrainSet == TRUE,]
test <- fulldata[fulldata$isTrainSet == FALSE,]

str(train)
str(test)
names(train)
names(test)

train <- select(train, -Sex, -Pclass, -TClasse, -Embarked, -SEmbarked, -PassengerId, -Name, -Ticket, -Cabin, -isTrainSet)
test <- select(test, -Sex, -Pclass, -TClasse, -Embarked, -SEmbarked, -PassengerId, -Name, -Ticket, -Cabin, -isTrainSet)

str(train)
str(test)

head(train, 10)
head(test, 10)

names(train)
names(test)

#4.1 KNN (K-NEAREST NEIGHBOR)

summary(train[c("Age", "SibSp", "Parch", "Fare")])

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Vamos testar a função nos dados abaixo, para verificar se eles ficam na mesma escala.
normalize(c(1,2,3,4,5))
normalize(c(10,20,30,40,50))

train$Age <- normalize(train$Age)
train$SibSp <- normalize(train$SibSp)
train$Parch <- normalize(train$Parch)
train$Fare <- normalize(train$Fare)

# Faço o mesmo procedimento para os dados de teste
test$Age <- normalize(test$Age)
test$SibSp <- normalize(test$SibSp)
test$Parch <- normalize(test$Parch)
test$Fare <- normalize(test$Fare)

# Chegou a hora de rodar o KNN, do pacote "class".

View(train)
View(test)

nrow(train)
sqrt(nrow(train))

modelo <- knn(train = train[,-1], test = train[,-1], cl = train[,1], k = 30)

i=1
k.optm=1
for (i in 1:30){
  modelo <- knn(train = train[,-1], test = train[,-1], cl = train[,1], k = i)
  k.optm[i] <- 100 * sum(train[,1] == modelo)/NROW(train[,1])
  k=i
  cat(k,'=',k.optm[i],'')
}

plot(k.optm, type="b", xlab="K- Value",ylab="Accuracy level")

modelo_30 <- knn(train = train[,-1], test = train[,-1], cl = train[,1], k = 30)
modelo_27 <- knn(train = train[,-1], test = train[,-1], cl = train[,1], k = 27)
modelo_28 <- knn(train = train[,-1], test = train[,-1], cl = train[,1], k = 28)
modelo_29 <- knn(train = train[,-1], test = train[,-1], cl = train[,1], k = 29)
modelo_24 <- knn(train = train[,-1], test = train[,-1], cl = train[,1], k = 24)

round(prop.table(table(modelo_30))*100, 2)
mean(train[,1] != modelo_30)

round(prop.table(table(modelo_27))*100, 2)
mean(train[,1] != modelo_27)

round(prop.table(table(modelo_28))*100, 2)
mean(train[,1] != modelo_28)

round(prop.table(table(modelo_29))*100, 2)
mean(train[,1] != modelo_29)

round(prop.table(table(modelo_24))*100, 2)
mean(train[,1] != modelo_24)

CrossTable(train[,1], modelo_30, prop.chisq = FALSE) # 93,6% de True Negative, 59,9% de True Positive
CrossTable(train[,1], modelo_27, prop.chisq = FALSE) # 94,4% de True Negative, 59,9% de True Positive
CrossTable(train[,1], modelo_28, prop.chisq = FALSE) # 94,0% de True Negative, 59,6% de True Positive
CrossTable(train[,1], modelo_29, prop.chisq = FALSE) # 93,6% de True Negative, 60,2% de True Positive
CrossTable(train[,1], modelo_24, prop.chisq = FALSE) # 94% de True Negative, 60% de True Positive

# Em resumo, o modelo que apresenta melhores resultados tem k = 24.

modelo <- knn(train = train[,-1], test = test[,-1], cl = train[,1], k = 24)

submission <- data.frame(PassengerId = passengers, Survived = modelo)
write.csv(submission,'titanic_knn.csv', row.names = FALSE)
