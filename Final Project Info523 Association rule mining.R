library (arules)
library(dplyr)
library(DMwR2)
library(rpart.plot)
library(corrplot)
library(ggplot2)
library(arulesViz)

#load dataset
healthy<-read.csv('health_lifestyle_classification.csv')
dim(healthy)
summary(healthy)
str(healthy)


numeric_columns <- sapply(healthy, is.numeric)
(num_numeric_variables <- sum(numeric_columns))

char_columns <- sapply(healthy, is.character)
(char_numeric_variables <- sum(char_columns))



##Descriptive Statistics
dev.new(width = 30, height = 30,unit = "cm") 
corrplot(cor(healthy_cor,use = "pairwise.complete.obs"))

healthy_cor<-healthy %>% select(-survey_code,-electrolyte_level,-gene_marker_flag,-environmental_risk_score)
healthy_cor <- healthy_cor %>%select(where(is.numeric))%>%select(where(is.numeric))

png(file = "my_plot.png", width = 2000, height = 2000)
corr<-corrplot(cor(healthy_cor,use = "pairwise.complete.obs"),type = 'lower')
dev.off()

ggplot(healthy_cor, aes(x=daily_steps)) +
  geom_boxplot(fill = "lightskyblue")


png(file = "outlier.png", width = 2000, height = 2000)
boxplot(healthy_cor$daily_steps,healthy_cor$income,names = c("Daily Steps", "Income"),col = c("#C4DFE6","#66A5AD"))
title("Distribution of daily steps and income")

dev.off()

boxplot(healthy_cor$physical_activity,col ="#C4DFE6",xlab='Physical Activity')
title("Distribution of physical activity")



#####Association rule mining#####

#data preprocessing

healthy_asso<-healthy %>% select(-survey_code,-bmi,-bmi_estimated,-bmi_scaled,-height,-weight,-waist_size,-insulin,-calorie_intake,-sugar_intake,-environmental_risk_score,-daily_supplement_dosage,-electrolyte_level,-gene_marker_flag)
#find classes of each of the column
lapply(healthy, class) 

#deal with missing values
colSums(is.na(healthy_asso))
numeric_cols <- sapply(healthy_asso, is.numeric)
for (i in which(numeric_cols)) {
  healthy_asso[is.na(healthy_asso[, i]), i] <- mean(healthy_asso[, i], na.rm = TRUE)
}
colSums(is.na(healthy_asso))

#write.csv(healthy_test, "healthy_test.csv", row.names = FALSE)

#discretize all remaining numerical variables
healthy_asso$age <- cut(healthy$age, breaks=c(0,19,39,64,Inf), labels=c("Teen", "Young Adult", "Adult", "Senior"))
healthy_asso$bmi_corrected <- cut(healthy$bmi_corrected, breaks=c(0,18.5,24.9,29.9,Inf), labels=c("Underweight", "Healthy", "Overweight", "Obese"))
healthy_asso$blood_pressure <- cut(healthy$blood_pressure, breaks=c(0,120,129,139,180,Inf), labels=c("Normal", "Elevated", "Stage 1 Hypertension", "Stage 2 Hypertension","Hypertensive Crisis"))
healthy_asso$heart_rate <- cut(healthy$heart_rate, breaks=c(0,60,100,Inf), labels=c("Bradycardia", "Normal", "Tachycardia"))
healthy_asso$cholesterol <- cut(healthy$cholesterol, breaks=c(0,200,239,Inf), labels=c("Normal", "Borderline high", "High"))
healthy_asso$glucose <- cut(healthy$glucose, breaks=c(0,100,125,Inf), labels=c("Normal", "Prediabetes", "Diabetes"))
healthy_asso$sleep_hours <- cut(healthy$sleep_hours, breaks=c(0,7,10,Inf), labels=c("Short", "Normal", "Long"))
healthy_asso$work_hours <- cut(healthy$work_hours, breaks=c(0,8,Inf), labels=c("0-8 hours","Longer than 8 hours"))
healthy_asso$physical_activity <- cut(healthy$physical_activity, breaks=4, labels=c("low", "medLow", "medHigh", "High"))
healthy_asso$daily_steps <- cut(healthy$daily_steps, breaks=c(0,5000,7499,9999,12499,Inf), labels=c("Sedantary", "Low Active", "Somewhat Active","Active","Highly Active"))
healthy_asso$water_intake <- cut(healthy$water_intake, breaks=4, labels=c("low", "medLow", "medHigh", "High"))
healthy_asso$screen_time <- cut(healthy$screen_time, breaks=4, labels=c("low", "medLow", "medHigh", "High"))
healthy_asso$stress_level <- cut(healthy$stress_level, breaks=4, labels=c("low", "medLow", "medHigh", "High"))
healthy_asso$mental_health_score <- cut(healthy$mental_health_score, breaks=4, labels=c("low", "medLow", "medHigh", "High"))
healthy_asso <- healthy_asso %>% 
  mutate(Income=case_when(income<2000 ~ "Monthly income less than $2,000",
                          income>=2000 & income<4000 ~"Monthly income between $2,000 and $4,000",
                          income>=4000 & income<6000 ~"Monthly income between $4,000 and $6,000",
                          income>=6000 & income<8000 ~"Monthly income between $6,000 and $8,000",
                          income>=8000 ~ "Monthly income more than $8,000"))

healthy_asso <- subset(healthy_asso, select = -income)
summary(healthy_asso$physical_activity)
summary(healthy_asso$work_hours)
healthy_asso$meals_per_day <- factor(healthy$meals_per_day)

#Convert character type to factor
char_cols <- sapply(healthy_asso, is.character)
healthy_asso[char_cols] <- lapply(healthy_asso[char_cols], as.factor)


dim(healthy_asso)
summary(healthy_asso)
#write.csv(healthy_asso, "healthy_asso.csv", row.names = FALSE)



#Association rule mining 
healthy_asso_ars <- as(healthy_asso, "transactions")
colnames(healthy_asso_ars)
summary(healthy_asso_ars)
itemFrequencyPlot(healthy_asso_ars, support=0.4, cex.names=0.8)
ars <- apriori(healthy_asso_ars, parameter = list(support=0.3, confidence=0.3,maxtime = 100))
summary(ars)

plot(ars, method = "graph", engine = "htmlwidget")

rules_conf <- sort (ars, by="confidence",decreasing=TRUE)
inspect(head(rules_conf,100))

rules_lift <- sort (ars, by="lift",decreasing=TRUE)
inspect(head(rules_lift,5))



inspect(head(subset(ars, subset=rhs %in% "target=healthy"),5, by="confidence"))
inspect(head(subset(ars, subset=rhs %in% "cholesterol=Normal"),5, by="confidence"))
inspect(head(subset(ars, subset=rhs %in% "heart_rate=Normal"),5, by="confidence"))
inspect(head(subset(ars, subset=lhs %in% "glucose=Normal"),5, by="confidence"))
inspect(head(subset(ars, subset=is.maximal(ars)), 5, by="confidence"))



