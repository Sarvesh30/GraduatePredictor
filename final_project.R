#Libraries used for Project Datsa
library(plotrix)
library(ggplot2)
library(reshape2)
library(car)
library(multcomp)
library(mosaic)

#Shriram project data
#Multiple Linear Regression Model from Students data
StudentsData<-read.csv("Admission_Predict_V11.csv",header = TRUE)
StudentsData<-StudentsData[,c("Chance.of.Admit","GRE.Score","TOEFL.Score","University.Rating","SOP","LOR","CGPA","Research","Serial.No.")]
summary(StudentsData)

#Attaching Dataset and number of rows and cols
attach(StudentsData)
dim(StudentsData)

#Check name and structure of the dataset
names(StudentsData)
str(StudentsData)

#Exploratory Data Analysis

#Printing piechart of distribution of Universities

print(table(University.Rating))
print(table(Research))

data_U=c(34,126,162,105,73)
data_R=c(220,280)
labels_U=c("U1", "U2", "U3", "U4","U5")
labels_R=c("No Experience","Experience")

# Calculate percentages
percent_U=round(100*data_U / sum(data_U),1)
percent_R=round(100*data_R / sum(data_R),1)

#Create 3D pie chart
pie(data_U,labels=paste(percent_U,sep=" ","%"),col=rainbow(length(data_U)),main ="Distribution of University")
# Add legend for University
legend("topright", 
       title = "Categories", 
       legend = labels_U, 
       fill = rainbow(length(data_U)), 
       border = "black",
)

pie(data_R,labels=paste(percent_R,sep=" ","%"),col=rainbow(length(data_R)),main ="Distribution of Experience among Students")
# Add legend for Research Experience
legend("topright", 
       title = "Categories", 
       legend = labels_R, 
       fill = rainbow(length(data_R)), 
       border = "black",
)

#Create Histogram
hist(CGPA,main="Distribution of CGPA",col="green",xlab="CGPA",ylab="Count")
hist(SOP,main="Distribution of SOP",col="blue",xlab="SOP",ylab="Count")
hist(LOR,main="Distribution of LOR",col="yellow",xlab="SOP",ylab="Count")
hist(Chance.of.Admit,main="Distribution of Chance.Of.Admit",col="orange",xlab="Chance of Admit",ylab="Count")
StudentsData %>% 
  pivot_longer(
    c(GRE.Score,TOEFL.Score)
  ) %>% 
  ggplot(aes(value, fill=name))+
  geom_histogram(position = "dodge")

#Create boxplot
boxplot(Chance.of.Admit~University.Rating,main="Chance of Admit-University wise",col="green")
boxplot(CGPA~University.Rating,main="CGPA-University wise",col="orange")

#Multiple Regression Model
#start by creating blank model and then decide predictors which are significant
StudentsData.lm<-lm(Chance.of.Admit~1,data=StudentsData)
add1(StudentsData.lm,StudentsData,test='F')

StudentsData.lm=lm(Chance.of.Admit~GRE.Score+TOEFL.Score,data=StudentsData)
summary(StudentsData.lm)
anova(StudentsData.lm)

add1(StudentsData.lm,StudentsData,test='F')
StudentsData.lm=lm(Chance.of.Admit~GRE.Score+TOEFL.Score+University.Rating+SOP+LOR+CGPA+Research,data=StudentsData)
summary(StudentsData.lm)
anova(StudentsData.lm)

#Final Multiple Regression Model
StudentsData.lm=lm(Chance.of.Admit~GRE.Score+TOEFL.Score+LOR+CGPA+Research)
summary(StudentsData.lm)

#ANOVA of Overall Final Model
anova(StudentsData.lm)

#Multicollinearity checking
#VIF > 5 or 10 indicates high multicollinearity
#Tolerance < 0.1 indicates high multicollinearity
#No Multicollinearity exist between predictors

vif(admission.lm)
1 / vif(admission.lm)

#Residual plots for ValidMultiple Linear Regression
#Residual plot is showing constant variance and doesn't show any pattern and therefore
#proves validity of Multiple LinearRegression

ggplot(StudentsData.lm, aes(x = .fitted, y = .resid)) +
  geom_point() +
  geom_hline(yintercept = 0) +
  labs(title='Residual vs. Fitted Values Plot', x='Fitted Values', y='Residuals')

#qqplot for residual using ggplot

ggplot(StudentsData.lm, aes(sample=.resid)) +
  stat_qq()

#Shapiro-Wilk test for normality of residual

#Null hypothesis of the Shapiro-Wilk test is that the data is normally distributed.
#Alternative hypothesis is that the data is not normally distributed

shapiro.test(resid(StudentsData.lm))

# Create scatter plots

df_melt <- melt(StudentsData, id.vars = "Chance.of.Admit")
ggplot(df_melt, aes(x = value, y = Chance.of.Admit, color = variable)) +
  geom_point() +
  labs(x = "Predictor Values", y = "Chance.of.Admit") +
  theme(legend.title = element_blank()) +
  facet_wrap(~ variable, scales = "free_x")

#95% confidence interval and prediction interval
confint(StudentsData.lm,conf.level=0.95)
predict(StudentsData.lm,interval = "prediction", level = 0.95)
