# ALY6050 Module3 Project - Forecasting Financial Time Series
# Yinan Zhou 06/09/2024

install.packages("readxl")
library(readxl)
library(ggplot2)
library(dplyr)

####################################### Part 1 ########################################

data <- read_excel("ALY6050_Module3Project_Data.xlsx")
head(data)
summary(data)

AAPL_price <- data[,3]
HON_price <- data[,5]

# Transfer the data frames into Vectors
AAPL_price <-AAPL_price[[1]]
HON_price <-HON_price[[1]]

df <- data.frame(AAPL_price, HON_price)
df <- na.omit(df)
n_days <- seq_along(df[,1])

## (i) Create line plots with the number of days as x-axis and AAPL_price and HON_price as y-axis
ggplot(df, aes(x = n_days, y = df[,1])) +
  geom_line(color = "blue", linetype = "solid") +
  labs(title = "AAPL Stock Prices",
       x = "Days",
       y = "Price") +
  scale_y_continuous(labels = scales::dollar) +
  theme_minimal()

ggplot(df, aes(x = n_days, y = df[,2])) +
  geom_line(color = "blue", linetype = "solid") +
  labs(title = "HON Stock Prices",
       x = "Days",
       y = "Price") +
  scale_y_continuous(labels = scales::dollar) +
  theme_minimal()


## (ii) exponential smoothing
exponential_smoothing_calculate_next <-function (alpha, Dt, Ft){
  F_next<-alpha*Dt+(1-alpha)*Ft
  return (F_next)
}

MAPD <- function (observed_vector, predicted_vector){
  diff <- abs(observed_vector-predicted_vector)
  return (sum(diff)/sum(observed_vector))
}

# AAPL
# Initialization
D1 <- AAPL_price[1] # The first actual price in the AAPL_price series
F1 <- D1 # Initialize the first forecast value to be the same as the first actual price
alpha_list <- c(0.15, 0.35, 0.55, 0.75)
aapl_predicted_lists <- list() # Initialize an empty list to store predicted values for each alpha

# alpha <-0.15
# predicted price over the days
# predicted <- c(F1)
# for (i in 1:length(n_days)){
  # currentD <- AAPL_price[i]
  # currentF <- predicted[i]
  # predicted<-c(predicted, exponential_smoothing_calculate_next(alpha, currentD, currentF))
# }
# mapd_aapl_alpha1 <- MAPD(AAPL_price[1:length(n_days)], predicted[1:length(n_days)+1])

# Loop Through Each Alpha
for (alpha in alpha_list){
  predicted <- c(F1)
  for (i in 1:length(n_days)){
    currentD <- AAPL_price[i]
    currentF <- predicted[i]
    predicted<-c(predicted, exponential_smoothing_calculate_next(alpha, currentD, currentF))
  }
  aapl_predicted_lists[[as.character(alpha)]] <- predicted # Store the predicted values for the current alpha
  mapd_aapl_alpha <- MAPD(AAPL_price[1:length(n_days)], predicted[0:length(n_days)])  # Calculate MAPD for the current alpha
  cat("AAPL MAPD of alpha =", alpha, ":", mapd_aapl_alpha, "\n")
}

# Print the Predicted Value for Day 253
for (alpha in alpha_list) {
  predicted_values <- aapl_predicted_lists[[as.character(alpha)]]
  if (length(predicted_values) >= 253) {
    cat("Alpha =", alpha, "AAPL Predicted value at day 253:", predicted_values[253], "\n")
  } else {
    cat("Alpha =", alpha, "does not have a 253rd predicted value\n")
  }
}

# HON
# Initialization
D1 <- HON_price[1]
F1 <- D1
alpha_list <- c(0.15, 0.35, 0.55, 0.75)
hon_predicted_lists <- list()

# Loop Through Each Alpha
alpha_list <- c(0.15, 0.35, 0.55, 0.75)
hon_predicted_lists <- list()
for (alpha in alpha_list){
  predicted <- c(F1)
  for (i in 1:length(n_days)){ 
    currentD <- HON_price[i]
    currentF <- predicted[i]
    predicted<-c(predicted, exponential_smoothing_calculate_next(alpha, currentD, currentF))
  }
  hon_predicted_lists[[as.character(alpha)]] <- predicted
  mapd_hon_alpha <- MAPD(HON_price[1:length(n_days)], predicted[0:length(n_days)])
  cat("HON MAPD of alpha =", alpha, ":", mapd_hon_alpha, "\n")
}

# Print the Predicted Value for Day 253
for (alpha in alpha_list) {
  predicted_values <- hon_predicted_lists[[as.character(alpha)]]
  if (length(predicted_values) >= 253) {
    cat("Alpha =", alpha, "HON Predicted value at day 253:", predicted_values[253], "\n")
  } else {
    cat("Alpha =", alpha, "does not have a 253rd predicted value\n")
  }
}

##(iii) 
cal_trend <- function (beta, Ft, Ft_1, Tt_1){
  Tt <- beta*(Ft-Ft_1)+(1-beta)*Tt_1
  return (Tt)
}

MAPE <- function (observed_vector, predicted_vector){
  diff <- abs(observed_vector-predicted_vector)/observed_vector
  return (sum(diff)/length(observed_vector))
}

# AAPL
predicted <- aapl_predicted_lists['0.55']
predicted <- predicted[[1]] # change data to value 
beta_list <- c(0.15, 0.25, 0.45, 0.85)
aapl_adjusted_lists <- list()

for (beta in beta_list){
  trend <- c(0)
  for (i in 1:length(n_days)){
    currentF <- predicted[i]
    nextF <- predicted[i+1]
    currentT <- trend[i]
    trend<-c(trend, cal_trend(beta, nextF, currentF, currentT))
  }
  adjusted_predict = predicted + trend
  aapl_adjusted_lists[[as.character(beta)]] <- adjusted_predict
  mape_aapl <- MAPE(AAPL_price[1:length(n_days)], adjusted_predict[0:length(n_days)])
  cat("AAPL MAPE of beta =", beta, ":", mape_aapl, "\n")
}

# Print the Predicted Value for Day 253
for (beta in beta_list) {
  adjusted_predict <- aapl_adjusted_lists[[as.character(beta)]]
  if (length(adjusted_predict) >= 253) {
    cat("Beta =", beta, "AAPL Predicted value at day 253:", adjusted_predict[253], "\n")
  } else {
    cat("Beta =", beta, "does not have a 253rd predicted value\n")
  }
}

aapl_predicted_beta4 <- aapl_adjusted_lists['0.85']
aapl_predicted_beta4 <-aapl_predicted_beta4[[1]]
ggplot(df, aes(x = n_days, y = df[,1])) +
  geom_line(color = "blue", linetype = "solid") +
  geom_line(aes(y=aapl_predicted_beta4[1:252]), color = "red", linetype = "dashed") +
  labs(title = "AAPL Stock Prices",
       x = "Days",
       y = "Price") +
  scale_y_continuous(labels = scales::dollar) +
  theme_minimal()

# HON
predicted <- hon_predicted_lists['0.55']
predicted <- predicted[[1]]
beta_list <- c(0.15, 0.25, 0.45, 0.85)
hon_adjusted_lists <- list()

for (beta in beta_list){
  trend <- c(0)
  for (i in 1:length(n_days)){
    currentF <- predicted[i]
    nextF <- predicted[i+1]
    currentT <- trend[i]
    trend<-c(trend, cal_trend(beta, nextF, currentF, currentT))
  }
  adjusted_predict = predicted + trend
  hon_adjusted_lists[[as.character(beta)]] <- adjusted_predict
  mape_hon <- MAPE(HON_price[1:length(n_days)], adjusted_predict[0:length(n_days)])
  cat("HON MAPE of beta =", beta, ":", mape_hon, "\n")
}

# Print the Predicted Value for Day 253
for (beta in beta_list) {
  adjusted_predict <- hon_adjusted_lists[[as.character(beta)]]
  if (length(adjusted_predict) >= 253) {
    cat("Beta =", beta, "AAPL Predicted value at day 253:", adjusted_predict[253], "\n")
  } else {
    cat("Beta =", beta, "does not have a 253rd predicted value\n")
  }
}

hon_predicted_beta4 <- hon_adjusted_lists['0.85']
hon_predicted_beta4 <-hon_predicted_beta4[[1]]
ggplot(df, aes(x = n_days, y = df[,2])) +
  geom_line(color = "blue", linetype = "solid") +
  geom_line(aes(y=hon_predicted_beta4[1:252]), color = "red", linetype = "dashed") +
  labs(title = "HON Stock Prices",
       x = "Days",
       y = "Price") +
  scale_y_continuous(labels = scales::dollar) +
  theme_minimal()

####################################### Part 2 ########################################

# AAPL
# (i)
moving_average <-function (x1, x2, x3){
  return (0.5*x3+0.3*x2+0.2*x1)
}
aapl_100 <- AAPL_price[1:3]

for (i in 3:100){
  x1<-AAPL_price[i-2]
  x2<-AAPL_price[i-1]
  x3<-AAPL_price[i]
  aapl_100 <- c(aapl_100, moving_average(x1,x2,x3))
}

sample_days <- c(101:252)
sample_AAPL_price <- AAPL_price[101:252]

# Fit a linear model
model <- lm(sample_AAPL_price ~ sample_days)
# Get the coefficients
coefficients <- coef(model)
coefficients

# Calculate the trend values
calculate_days <- c(101:257)
aapl_predicted_trend_values <- coefficients[1] + coefficients[2] * calculate_days
aapl_predicted <- c(aapl_100[1:100],aapl_predicted_trend_values)

ggplot(df, aes(x = n_days, y = df[,1])) +
  geom_line(color = "blue", linetype = "solid") +
  geom_line(aes(y=aapl_predicted[1:252]), color = "red", linetype = "dashed") +
  labs(title = "AAPL Stock Prices",
       x = "Days",
       y = "Price") +
  scale_y_continuous(labels = scales::dollar) +
  theme_minimal()

observed_aapl_price<-c(AAPL_price[1:252], 116.32,115.97,119.49,119.21,119.26)
for (i in c(253:257)){
  cat ("period ", i, ": linear trend predicted price = ", aapl_predicted[i], ", observed price = ", observed_aapl_price[i],'\n')
}

# (ii)
mape_aapl<-MAPE(observed_aapl_price[4:252],aapl_predicted[4:252])
cat ("AAPL MAPE = ", mape_aapl, '\n')

################################################################################

#HON
observed_hon_price<-c(HON_price[1:252], 196.99,201.98,199.29,197.24,201.54)
# (i)
moving_average <-function (x1, x2, x3){
  return (0.5*x3+0.3*x2+0.2*x1)
}
hon_100 <- HON_price[1:3]

for (i in 3:100){
  x1<-HON_price[i-2]
  x2<-HON_price[i-1]
  x3<-HON_price[i]
  hon_100 <- c(hon_100, moving_average(x1,x2,x3))
}

sample_days <- c(101:252)
sample_HON_price <- HON_price[101:252]

# Fit a linear model
model <- lm(sample_HON_price ~ sample_days)
# Get the coefficients
coefficients <- coef(model)
# Calculate the trend values
calculate_days <- c(101:257)
hon_predicted_trend_values <- coefficients[1] + coefficients[2] * calculate_days
hon_predicted <- c(hon_100[1:100],hon_predicted_trend_values)

ggplot(df, aes(x = n_days, y = df[,2])) +
  geom_line(color = "blue", linetype = "solid") +
  geom_line(aes(y=hon_predicted[1:252]), color = "red", linetype = "dashed") +
  labs(title = "HON Stock Prices",
       x = "Days",
       y = "Price") +
  scale_y_continuous(labels = scales::dollar) +
  theme_minimal()

observed_hon_price<-c(HON_price[1:252], 196.99,201.98,199.29,197.24,201.54)
for (i in c(253:257)){
  cat ("period ", i, ": linear trend predicted price = ", hon_predicted[i], ", observed price = ", observed_hon_price[i],'\n')
}

# (ii)
mape_hon<-MAPE(observed_hon_price[4:252],hon_predicted[4:252])
cat ("HON MAPE = ", mape_hon, '\n')


####################################### Part 3 ########################################

# AAPL
#(i)
sample_days <- 1:252
sample_AAPL_price <- AAPL_price[1:252]

# Fit a linear model
model <- lm(sample_AAPL_price ~ sample_days)
# Get the coefficients
coefficients <- coef(model)
coefficients

# Predict values for periods 1 through 257
future_days <- 1:257
aapl_predicted <- predict(model, newdata = data.frame(sample_days = future_days))
mape_aapl<-MAPE(observed_aapl_price[1:252],aapl_predicted[1:252])
cat ("AAPL MAPE = ", mape_aapl, '\n')

ggplot(df, aes(x = n_days, y = df[,1])) +
  geom_line(color = "blue", linetype = "solid") +
  geom_line(aes(y=aapl_predicted[1:252]), color = "red", linetype = "dashed") +
  labs(title = "AAPL Stock Prices",
       x = "Days",
       y = "Price") +
  scale_y_continuous(labels = scales::dollar) +
  theme_minimal()

#(ii)
aapl_residual <- observed_aapl_price[1:252] - aapl_predicted[1:252]

# Independence (plot the residual vs independent variable)
plot(1:252, aapl_residual)

# Homoscedastic (plot residual vs predicted y)
plot(aapl_predicted[1:252], aapl_residual)

# normal probability plot
qqnorm(aapl_residual)
qqline(aapl_residual, col = "red")

# Chi-square test
# Define the number of bins
hist(aapl_residual,breaks = 5 )
num_bins <- 5
# Create breaks for binning
breaks <- quantile(aapl_residual, probs = seq(0, 1, length.out = num_bins + 1))
# Bin the residual
obs_freq <- table(cut(aapl_residual, breaks = breaks, include.lowest = TRUE))
# Calculate expected frequencies assuming a normal distribution
expected_freq <- diff(pnorm(breaks, mean = mean(aapl_residual), sd = sd(aapl_residual))) * length(aapl_residual)
# Perform the Chi-squared test
chi_sq_test_aaple <- chisq.test(obs_freq, p = expected_freq / sum(expected_freq))
# Print the results of the Chi-squared test
print(chi_sq_test_aaple)

# HON
#(i)
sample_days <- 1:252
sample_HON_price <- HON_price[1:252]

# Fit a linear model
model <- lm(sample_HON_price ~ sample_days)
# Get the coefficients
coefficients <- coef(model)
coefficients

# Predict values for periods 1 through 257
future_days <- 1:257
hon_predicted <- predict(model, newdata = data.frame(sample_days = future_days))
mape_hon<-MAPE(observed_hon_price[1:252],hon_predicted[1:252])
cat ("HON MAPE = ", mape_hon, '\n')

ggplot(df, aes(x = n_days, y = df[,2])) +
  geom_line(color = "blue", linetype = "solid") +
  geom_line(aes(y=hon_predicted[1:252]), color = "red", linetype = "dashed") +
  labs(title = "HON Stock Prices",
       x = "Days",
       y = "Price") +
  scale_y_continuous(labels = scales::dollar) +
  theme_minimal()

#(ii)
hon_residual <- observed_hon_price[1:252] - hon_predicted[1:252]

# Independence (plot the residual vs independent variable)
plot(1:252, hon_residual)

# Homoscedastic (plot residual vs predicted y)
plot(hon_predicted[1:252], aapl_residual)

# normal probability plot
qqnorm(hon_residual)
qqline(hon_residual, col = "red")

# Chi-square test
# Define the number of bins
hist(hon_residual,breaks = 5 )
num_bins <- 5
# Create breaks for binning
breaks <- quantile(hon_residual, probs = seq(0, 1, length.out = num_bins + 1))
# Bin the residual
obs_freq <- table(cut(hon_residual, breaks = breaks, include.lowest = TRUE))
# Calculate expected frequencies assuming a normal distribution
expected_freq <- diff(pnorm(breaks, mean = mean(hon_residual), sd = sd(hon_residual))) * length(hon_residual)
# Perform the Chi-squared test
chi_sq_test_hon<- chisq.test(obs_freq, p = expected_freq / sum(expected_freq))
# Print the results of the Chi-squared test
print(chi_sq_test_hon)

#################################Question#######################################

# Remove rows with NA values
data <- na.omit(data)

# Extract AAPL and HON stock prices
aapl_prices <- data[, 3]
aapl_prices <- aapl_prices[[1]]
hon_prices <- data[, 5]
hon_prices <- hon_prices[[1]]

# Calculate daily returns
aapl_returns <- diff(log(aapl_prices))
hon_returns <- diff(log(hon_prices))

# Combine returns into a single data frame
returns <- data.frame(Date = data$Date[-1], AAPL_Returns = aapl_returns, HON_Returns = hon_returns)

# Calculate mean daily returns
mean_aapl_returns <- mean(returns$AAPL_Returns)
mean_aapl_returns
mean_hon_returns <- mean(returns$HON_Returns)
mean_hon_returns

# Calculate standard deviation of daily returns
sd_aapl_returns <- sd(returns$AAPL_Returns)
sd_aapl_returns
sd_hon_returns <- sd(returns$HON_Returns)
sd_hon_returns

# Calculate Sharpe Ratio
risk_free_rate <- 0.0525  # Use the three-month Treasury bond rate 5.25% in this case
sharpe_aapl <- (mean_aapl_returns - risk_free_rate) / sd_aapl_returns
sharpe_aapl 
sharpe_hon <- (mean_hon_returns - risk_free_rate) / sd_hon_returns
sharpe_hon

# Calculate optimal allocation weights
sharpe_weights <- c(sharpe_aapl, sharpe_hon) / sum(c(sharpe_aapl, sharpe_hon))
aapl_allocation <- sharpe_weights[1]
hon_allocation <- sharpe_weights[2]

# Print the results
print(paste("AAPL allocation percentage:", round(aapl_allocation * 100, 2), "%"))
print(paste("HON allocation percentage:", round(hon_allocation * 100, 2), "%"))


