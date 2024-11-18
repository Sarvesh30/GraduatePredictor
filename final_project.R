#Libraries used for Project Datsa
library(plotrix)
library(ggplot2)
library(reshape2)
library(car)
library(multcomp)

#Shriram project data
#Multiple Linear Regression Model from Students data
StudentsData<-read.csv("Admission_Predict_V11.csv",header = TRUE)
StudentsData<-StudentsData[,c("Chance.of.Admit","Serial.No.","GRE.Score","TOEFL.Score","University.Rating","SOP","LOR","CGPA","Research")]
summary(StudentsData)

#Attaching Dataset and number of rows and cols
attach(StudentsData)
dim(StudentsData)

#descriptive statistics of students data

#Printing piechart of distribution of Universities

print(table(University.Rating))

data=c(34,126,162,105,73)
labels=c("U1", "U2", "U3", "U4","U5") 

# Calculate percentages
piepercent=round(100*data / sum(data),1)
piepercent=paste(piepercent,"%",sep="")
#Create 3D pie chart
pie3D(data,labels=piepercent,main = "Piechart of Distribution of University",cex=0.8)
# Add legend
legend("topright", 
       title = "Categories", 
       legend = labels, 
       fill = rainbow(length(data)), 
       border = "black", 
       bty = "n")

#Create Histogram
hist(CGPA,main="Distribution of CGPA",col="green",xlab="CGPA",ylab="Count")
hist(Chance.of.Admit,main="Distribution of Chance.Of.Admit",col="orange",xlab="Chance of Admit",ylab="Count")


boxplot(Chance.of.Admit~University.Rating,main="Chance of Admit-University wise",col="green")
boxplot(CGPA~University.Rating,main="CGPA-University wise",col="orange")

#Check name and structure of the dataset

names(StudentsData)
str(StudentsData)


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

#Final Model
admission.lm=lm(Chance.of.Admit~GRE.Score+TOEFL.Score+LOR+CGPA+Research)
summary(admission.lm)
anova(admission.lm)

#Scatter plot
# (a) Create scatterplots

#admissionData2 <- melt(StudentsData[,2:6], id.vars='y') 
admissionData2=melt(StudentsData[,3:8],id.vars = 'y')
ggplot(admissionData2) + geom_jitter(aes(value,y, colour=variable),) +
  geom_smooth(aes(value,y, colour=variable), method=lm, se=FALSE) +
  facet_wrap(~variable, scales="free_x")

#95% confidence interval and prediction interval
confint(StudentsData.lm,conf.level=0.95)
predict(StudentsData.lm,interval = "prediction", level = 0.95)

# Updated ggplot section

# Ensure data preparation for ggplot
# Verify if admissionData2 is derived from StudentsData or another source
admissionData2 <- melt(StudentsData, 
                       id.vars = "Chance.of.Admit", 
                       measure.vars = c("GRE.Score", "TOEFL.Score", "CGPA", "University.Rating", "SOP", "LOR"))

ggplot(admissionData2, aes(x = value, y = Chance.of.Admit)) +
  geom_point(alpha = 0.6, colour = "blue") + # Scatterplot points
  geom_smooth(method = "lm", se = FALSE, colour = "red") + # Linear trend lines
  facet_wrap(~ variable, scales = "free_x") + # Create a plot for each variable
  labs(
    title = "Scatterplots of Predictors vs. Chance of Admit",
    x = "Predictor Value",
    y = "Chance of Admission"
  ) +
  theme_minimal()