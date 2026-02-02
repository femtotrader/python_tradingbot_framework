from utils.core import Bot
from utils.portfolio import TRADEABLE, sharpe_compute_weights


class SharpePortfolioOptWeeklyBot(Bot):
    """
    Bot that rebalances weekly to a portfolio optimized for maximum Sharpe ratio.
    Uses PyPortfolioOpt to calculate optimal weights and rebalancePortfolio to execute trades.
    """
    
    def __init__(self):
        super().__init__("SharpePortfolioOptWeeklyBot", symbol=None)
        self.tradeable_symbols = TRADEABLE
    
    def makeOneIteration(self):
        """
        Execute weekly rebalancing based on Sharpe ratio optimization.
        
        Returns:
            0: Rebalancing completed (no traditional buy/sell signal)
        """
        print("Fetching market data for portfolio optimization...")
        
        # Download data for all tradeable symbols using base class method
        try:
            # Get data in long format (symbol, timestamp, open, high, low, close, volume)
            data_long = self.getYFDataMultiple(
                self.tradeable_symbols, 
                interval="1d", 
                period="3mo", 
                saveToDB=True
            )
            
            if data_long.empty:
                print("Warning: No data fetched for optimization")
                return 0
            
            # Convert to wide format for PyPortfolioOpt using base class method
            df = self.convertToWideFormat(data_long, value_column="close", fill_method="both")
            
            if df.empty:
                print("Warning: No valid data after cleaning")
                return 0
            
            print(f"Calculating optimal portfolio weights for {len(df.columns)} assets...")
            weights = sharpe_compute_weights(df)
            if not weights:
                print("Warning: Optimization returned no non-zero weights")
                return 0

            print(f"Optimized portfolio contains {len(weights)} assets")
            print("Rebalancing portfolio to target weights...")
            print(
                f"Top 5 holdings: {dict(list(sorted(weights.items(), key=lambda x: x[1], reverse=True))[:5])}"
            )

            # Rebalance portfolio using base class method with onlyOver50USD=True
            # This will filter out assets with target value <= $50 and redistribute weights
            self.rebalancePortfolio(weights, onlyOver50USD=True)
            
            print("Weekly rebalancing completed successfully")
            return 0
        
        except Exception as e:
            print(f"Error during portfolio optimization: {e}")
            import traceback
            traceback.print_exc()
            raise


bot = SharpePortfolioOptWeeklyBot()

# bot.local_development()
bot.run()
