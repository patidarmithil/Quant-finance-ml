from src.data_handler import prepare_data
from src.model_builder import train_logistic_regression, train_xgboost
from src.backtester import apply_strategy, backtest, performance_metrics, model_accuracy, plot_results

def main():
    print("Loading and preparing data...")
    train, test, features = prepare_data(ticker="RELIANCE.NS", split_date="2021-01-01")
    
    print("Training Logistic Regression...")
    train_lr, test_lr = train_logistic_regression(train, test, features)
    
    print("Training XGBoost...")
    train_xgb, test_xgb = train_xgboost(train, test, features)
    
    print("Applying trading strategies and backtesting...")
    # Logistic Regression
    test_lr = apply_strategy(test_lr, "prob_lr")
    test_lr = backtest(test_lr)
    
    # XGBoost
    test_xgb = apply_strategy(test_xgb, "prob_xgb")
    test_xgb = backtest(test_xgb)
    
    # Output Metrics
    print("\n--- Performance Metrics ---")
    print(f"Logistic Regression: {performance_metrics(test_lr)}")
    print(f"XGBoost: {performance_metrics(test_xgb)}")
    
    print("\n--- Model Accuracy ---")
    print(f"Logistic Regression Accuracy: {model_accuracy(test_lr, 'prob_lr')}%")
    print(f"XGBoost Accuracy: {model_accuracy(test_xgb, 'prob_xgb')}%")
    
    print("\nGenerating Plots...")
    plot_results(test_lr, test_xgb)

if __name__ == "__main__":
    main()
