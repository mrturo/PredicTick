"""Backtesting simulator for evaluating financial market prediction models."""

import datetime
import os
from typing import List, Optional

import joblib
import pandas as pd
from dateutil.relativedelta import relativedelta
from pytz import timezone
from sklearn.metrics import accuracy_score

from prediction__predict_core import MarketDataHandler, PredictionResult
from prediction__predict_model import SymbolPredictor
from updater.data_updater import FileManager
from utils.feature_engineering import enrich_with_common_features
from utils.indicators import IndicatorCalculator
from utils.logger import Logger
from utils.parameters import ParameterLoader

# Global parameter instance
parameters = ParameterLoader(FileManager.last_update())

# === Global Parameters ===
features = parameters["features"]
model_path = parameters["model_filepath"]
scaler_path = parameters["scaler_filepath"]
market_tz = parameters["market_tz"]
cutoff_minutes = parameters["cutoff_minutes"]
up_threshold = parameters["up_threshold"]
down_threshold = parameters["down_threshold"]
historical_window_days = parameters["historical_window_days"]
backtesting_output_folder = parameters["backtesting_output_folder"]

indicator_params = IndicatorCalculator.get_indicator_parameters()
rsi_window = indicator_params["rsi_window"]
macd_fast = indicator_params["macd_fast"]
macd_slow = indicator_params["macd_slow"]
macd_signal = indicator_params["macd_signal"]
bollinger_window = indicator_params["bollinger_window"]
stoch_rsi_window = indicator_params["stoch_rsi_window"]
atr_window = indicator_params["atr_window"]
williams_r_window = indicator_params["williams_r_window"]
stoch_rsi_min_periods = indicator_params["stoch_rsi_min_periods"]
bollinger_band_method = indicator_params["bollinger_band_method"]
obv_fill_method = indicator_params["obv_fill_method"]

stoch_rsi_min_periods = parameters["stoch_rsi_min_periods"]
bollinger_band_method = parameters["bollinger_band_method"]
obv_fill_method = parameters["obv_fill_method"]

training_symbols = parameters["training_symbols"]
mercado_pago_symbols = parameters["mercado_pago_symbols"]
xtb_symbols = parameters["xtb_symbols"]


# === Utility Functions ===
def _scale_features(scaler, df: pd.DataFrame):
    """Scale features using the provided scaler."""
    return scaler.transform(df[features])


def _build_filename(filename: str, suffix: str = ""):
    base, ext = os.path.splitext(filename)
    suffix = suffix.strip()
    if len(suffix) > 0:
        suffix = f"_{suffix}"
    return f"{base}{suffix}{ext}"


def buid_real_direction(target: int):
    """Map integer target value to market direction label."""
    return "UP" if target == 2 else ("NEUTRAL" if target == 1 else "DOWN")


def _load_model_and_scaler():
    """Load model and scaler artifacts."""
    model = joblib.load(_build_filename(model_path))
    scaler = joblib.load(_build_filename(scaler_path))
    return model, scaler


# === Symbol Processor ===
# pylint: disable=too-few-public-methods
class SymbolProcessor:
    """Processes raw symbol data for backtesting."""

    @staticmethod
    def process(symbol: str, raw_data: List[dict]):
        """Log prediction details for the given trading day."""
        df = pd.DataFrame(raw_data)
        if df.empty:
            Logger.warning(f"No data found for {symbol}.")
            return pd.DataFrame()

        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
        df.set_index("Datetime", inplace=True)
        df = df.tz_convert(market_tz).sort_index()
        df["date"] = df.index.date

        now_ny = datetime.datetime.now(timezone(market_tz))
        cutoff_sec = cutoff_minutes * 60

        valid_indices = [
            idx
            for day, group in df.groupby("date")
            if (now_ny - group.index.max()).total_seconds() > cutoff_sec
            for idx in group.index
        ]
        df = df.loc[valid_indices]
        df = enrich_with_common_features(df, symbol)

        return df.dropna(subset=features + ["target"])


# === Backtest Simulator ===
class BacktestSimulator:
    """Simulates the backtest based on model predictions."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.capital = {
            "last_year": 1.0,
            "last_nine_months": 1.0,
            "last_six_months": 1.0,
            "last_three_months": 1.0,
            "last_one_month": 1.0,
            "last_three_weeks": 1.0,
            "last_two_weeks": 1.0,
            "last_one_week": 1.0,
        }
        self.y_true: List[int] = []
        self.y_pred: List[int] = []
        self.daily_returns = []  # To track daily returns for profit factor calculation
        self.num_operations = 0  # To count number of operations (UP predictions)
        self.accurate_predictions = 0
        self.max_drawdown = 0  # To track the max drawdown
        self.max_capital = (
            1.0  # To track the maximum capital value for drawdown calculation
        )
        self.predictor = SymbolPredictor()
        self.predictions_log = []

    def _process_data_for_backtest(self, df: pd.DataFrame):
        """Pre-process data to get ready for backtesting simulation."""
        df["date"] = df.index.date
        last_available = df["date"].max()
        unique_dates = sorted(
            d
            for d in df["date"].unique()
            if d >= last_available - datetime.timedelta(days=historical_window_days)
        )
        return unique_dates, df[df["date"].isin(unique_dates)]

    def save_predictions_to_csv(self):
        """Save all backtest predictions to a timestamped CSV file."""
        if not self.predictions_log:
            Logger.warning("No predictions to save.")
            return

        os.makedirs(backtesting_output_folder, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = os.path.join(
            backtesting_output_folder, f"backtest_predictions_{timestamp}.csv"
        )

        df_predictions = pd.DataFrame(self.predictions_log)
        df_predictions.to_csv(output_path, index=False)

        Logger.success(f"Backtesting saved to {output_path}")

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def print_day_details(
        self, date, best_symbol, prediction, score, target, daily_return
    ):
        """Log prediction details for the given trading day."""
        predicted = "UP" if prediction == 1 else "DOWN"
        real = buid_real_direction(target)
        accurate = predicted == real if prediction == 1 else "None"
        if best_symbol is not None:
            if predicted == "DOWN":
                Logger.debug(
                    f"  📈 {date} → No symbol is predicted to perform positively."
                )
            else:
                Logger.debug(
                    f"  📈 {date} → Best Symbol: {best_symbol} → Predicted: {predicted} "
                    f"(Score: {score*100:.2f}) → Real: {real} | Accurate: {accurate} "
                    f"| Daily Return: {daily_return*100:.2f}%"
                )
        else:
            Logger.debug(
                f"  📈 {date} → Predicted: {predicted} (Score: {score*100:.2f}) "
                f"→ Real: {real} | Accurate: {accurate} | Daily Return: {daily_return*100:.2f}%"
            )

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def simulate_same_symbol_all_days(self, df: pd.DataFrame):
        """Simulate backtesting by applying predictions on the same symbol across multiple days."""
        # Preprocess the data
        unique_dates, processed_data = self._process_data_for_backtest(df)
        symbol = processed_data["symbol"].iloc[0]

        date_nine_months_ago = max(unique_dates) - relativedelta(months=9)
        date_six_months_ago = max(unique_dates) - relativedelta(months=6)
        date_three_months_ago = max(unique_dates) - relativedelta(months=3)
        date_one_month_ago = max(unique_dates) - relativedelta(months=1)
        date_three_weeks_ago = max(unique_dates) - relativedelta(weeks=3)
        date_two_weeks_ago = max(unique_dates) - relativedelta(weeks=2)
        date_one_week_ago = max(unique_dates) - relativedelta(weeks=1)

        for current_date in unique_dates:
            result: PredictionResult = self.predictor.predict_next_day(
                [symbol], current_date
            )
            if result is None:
                Logger.debug(
                    f"  📈 {result.next_date} → No symbol is predicted to perform positively."
                )
                continue

            sample = processed_data[processed_data["date"] == result.next_date].iloc[
                -1:
            ]
            if sample.empty or pd.isna(sample["target"].values[0]):
                continue

            last_row = sample.iloc[0]
            prediction = 1 if result.direction == "UP" else 0
            score = result.expected_return_score

            self.y_true.append(last_row["target"])
            self.y_pred.append(prediction)

            # Evaluar retorno diario
            if prediction == 1:
                self.num_operations += 1
                if last_row["target"] == 2:
                    self.accurate_predictions += 1
                daily_return = (
                    last_row["today_close"] - last_row["close_yesterday"]
                ) / last_row["close_yesterday"]
            else:
                daily_return = 0

            # Capital y drawdown
            self.capital["last_year"] *= 1 + daily_return
            if result.next_date >= date_nine_months_ago:
                self.capital["last_nine_months"] *= 1 + daily_return
            if result.next_date >= date_six_months_ago:
                self.capital["last_six_months"] *= 1 + daily_return
            if result.next_date >= date_three_months_ago:
                self.capital["last_three_months"] *= 1 + daily_return
            if result.next_date >= date_one_month_ago:
                self.capital["last_one_month"] *= 1 + daily_return
            if result.next_date >= date_three_weeks_ago:
                self.capital["last_three_weeks"] *= 1 + daily_return
            if result.next_date >= date_two_weeks_ago:
                self.capital["last_two_weeks"] *= 1 + daily_return
            if result.next_date >= date_one_week_ago:
                self.capital["last_one_week"] *= 1 + daily_return

            self.daily_returns.append(daily_return)

            # Update max capital for drawdown calculation
            if self.capital["last_year"] > self.max_capital:
                self.max_capital = self.capital["last_year"]
            self.max_capital = max(self.max_capital, self.capital["last_year"])
            self.max_drawdown = max(
                self.max_drawdown,
                (self.max_capital - self.capital["last_year"]) / self.max_capital,
            )

            # Logging
            self.print_day_details(
                result.next_date,
                None,
                prediction,
                score,
                last_row["target"],
                daily_return,
            )
            real_direction = buid_real_direction(last_row["target"])
            self.predictions_log.append(
                {
                    "symbol": result.symbol,
                    "Date": result.next_date,
                    "Predicted": result.direction,
                    "Score": result.expected_return_score,
                    "Real": real_direction,
                    "Accurate": (
                        prediction == real_direction if prediction == 1 else None
                    ),
                    "Return": daily_return * 100,
                }
            )

    def summarize(self):
        """Output the final backtest summary."""
        acc = accuracy_score(self.y_true, self.y_pred)

        # Profit Factor Calculation
        total_gains = sum(d for d in self.daily_returns if d > 0)
        total_losses = -sum(d for d in self.daily_returns if d < 0)
        profit_factor = total_gains / total_losses if total_losses != 0 else 0

        Logger.success("Annual Statistics:")
        Logger.simple(f"   * Accuracy: {acc:.2%}")
        Logger.simple(f"   * Total operations (UP predictions): {self.num_operations}")
        Logger.simple(
            f"   * Accurate Predictions (UP predictions & UP real): {self.accurate_predictions}"
        )
        Logger.simple(f"   * Max Drawdown: {self.max_drawdown * 100:.2f}%")
        Logger.simple(f"   * Profit Factor: {profit_factor:.2f}")
        Logger.success("Return Summary:")
        Logger.simple(
            f"   * Last year: {(self.capital['last_year'] - 1.0) * 100:.2f}% "
            f"(Monthly Avg: {(self.capital['last_year'] - 1.0) * 100 / 12:.2f}%)"
        )
        Logger.simple(
            f"   * Last nine months: {(self.capital['last_nine_months'] - 1.0) * 100:.2f}% "
            f"(Monthly Avg: {(self.capital['last_nine_months'] - 1.0) * 100 / 9:.2f}%)"
        )
        Logger.simple(
            f"   * Last six months: {(self.capital['last_six_months'] - 1.0) * 100:.2f}% "
            f"(Monthly Avg: {(self.capital['last_six_months'] - 1.0) * 100 / 6:.2f}%)"
        )
        Logger.simple(
            f"   * Last three months: {(self.capital['last_three_months'] - 1.0) * 100:.2f}% "
            f"(Monthly Avg: {(self.capital['last_three_months'] - 1.0) * 100 / 3:.2f}%)"
        )
        Logger.simple(
            f"   * Last one month: {(self.capital['last_one_month'] - 1.0) * 100:.2f}% "
            f"(Weekly Avg: {(self.capital['last_one_month'] - 1.0) * 100 / 4:.2f}%)"
        )
        Logger.simple(
            f"   * Last three weeks: {(self.capital['last_three_weeks'] - 1.0) * 100:.2f}% "
            f"(Weekly Avg: {(self.capital['last_three_weeks'] - 1.0) * 100 / 3:.2f}%)"
        )
        Logger.simple(
            f"   * Last two weeks: {(self.capital['last_two_weeks'] - 1.0) * 100:.2f}% "
            f"(Weekly Avg: {(self.capital['last_two_weeks'] - 1.0) * 100 / 2:.2f}%)"
        )
        Logger.simple(
            f"   * Last one weeks: {(self.capital['last_one_week'] - 1.0) * 100:.2f}% "
            f"(Dayly Avg: {(self.capital['last_one_week'] - 1.0) * 100 / 7:.2f}%)"
        )

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def simulate_best_symbol_per_day(self, market_data_dict: dict):
        """Backtesting by dynamically selecting the best symbol per day using raw market data."""
        symbols = list(market_data_dict.keys())

        # Load and process all symbols
        all_frames = []
        for symbol, historical in market_data_dict.items():
            df = SymbolProcessor.process(symbol, historical)
            if not df.empty:
                all_frames.append(df)

        if not all_frames:
            Logger.error("No valid data processed from input market_data_dict.")
            return

        full_data = pd.concat(all_frames)
        full_data["date"] = full_data.index.date
        unique_dates = sorted(full_data["date"].unique())
        last_date = max(unique_dates)
        unique_dates = [
            d
            for d in unique_dates
            if d >= last_date - datetime.timedelta(days=historical_window_days)
        ]

        date_nine_months_ago = last_date - relativedelta(months=9)
        date_six_months_ago = last_date - relativedelta(months=6)
        date_three_months_ago = last_date - relativedelta(months=3)
        date_one_month_ago = last_date - relativedelta(months=1)
        date_three_weeks_ago = last_date - relativedelta(weeks=3)
        date_two_weeks_ago = last_date - relativedelta(weeks=2)
        date_one_week_ago = last_date - relativedelta(weeks=1)

        for current_date in unique_dates:
            result: PredictionResult = self.predictor.predict_next_day(
                symbols=symbols, current_date=current_date
            )
            if result is None:
                Logger.debug(
                    f"  📈 {result.next_date} → No symbol is predicted to perform positively."
                )
                continue

            sample = full_data[
                (full_data["symbol"] == result.symbol)
                & (full_data["date"] == result.next_date)
            ].iloc[-1:]

            if sample.empty or pd.isna(sample["target"].values[0]):
                continue

            last_row = sample.iloc[0]
            prediction = 1 if result.direction == "UP" else 0
            score = result.expected_return_score

            self.y_true.append(last_row["target"])
            self.y_pred.append(prediction)

            # Evaluate return
            if prediction == 1:
                self.num_operations += 1
                if last_row["target"] == 2:
                    self.accurate_predictions += 1
                daily_return = (
                    last_row["today_close"] - last_row["close_yesterday"]
                ) / last_row["close_yesterday"]
            else:
                daily_return = 0

            # Update capital
            self.capital["last_year"] *= 1 + daily_return
            if result.next_date >= date_nine_months_ago:
                self.capital["last_nine_months"] *= 1 + daily_return
            if result.next_date >= date_six_months_ago:
                self.capital["last_six_months"] *= 1 + daily_return
            if result.next_date >= date_three_months_ago:
                self.capital["last_three_months"] *= 1 + daily_return
            if result.next_date >= date_one_month_ago:
                self.capital["last_one_month"] *= 1 + daily_return
            if result.next_date >= date_three_weeks_ago:
                self.capital["last_three_weeks"] *= 1 + daily_return
            if result.next_date >= date_two_weeks_ago:
                self.capital["last_two_weeks"] *= 1 + daily_return
            if result.next_date >= date_one_week_ago:
                self.capital["last_one_week"] *= 1 + daily_return
            self.daily_returns.append(daily_return)

            if self.capital["last_year"] > self.max_capital:
                self.max_capital = self.capital["last_year"]
            self.max_drawdown = max(
                self.max_drawdown,
                (self.max_capital - self.capital["last_year"]) / self.max_capital,
            )
            self.print_day_details(
                result.next_date,
                result.symbol,
                prediction,
                score,
                last_row["target"],
                daily_return,
            )
            real_direction = buid_real_direction(last_row["target"])
            self.predictions_log.append(
                {
                    "symbol": result.symbol,
                    "Date": result.next_date,
                    "Predicted": result.direction,
                    "Score": result.expected_return_score,
                    "Real": real_direction,
                    "Accurate": (
                        result.direction == real_direction if prediction == 1 else None
                    ),
                    "Return": daily_return * 100,
                }
            )


# === Main Execution ===
def backtest_same_symbol_all_days(symbol: str, market_data=None):
    """Main function to execute the backtest for a given symbol."""
    market_data_dict = (
        MarketDataHandler.load_market_data() if market_data is None else market_data
    )

    if symbol not in market_data_dict:
        return

    model, scaler = _load_model_and_scaler()
    df_symbol = SymbolProcessor.process(symbol, market_data_dict[symbol])

    if df_symbol.empty:
        Logger.error(f"No valid data to backtest for symbol: {symbol}")
        return

    simulator = BacktestSimulator(model, scaler)
    simulator.simulate_same_symbol_all_days(df_symbol)
    simulator.save_predictions_to_csv()
    simulator.summarize()


def backtest_best_symbol_per_day(symbols: Optional[List[str]] = None, market_data=None):
    """Backtest predictions by dynamically selecting the best-performing symbol per day."""
    market_data_dict = (
        MarketDataHandler.load_market_data() if market_data is None else market_data
    )
    if symbols is None:
        symbols = list(market_data_dict.keys())
    else:
        market_data_dict = {
            symbol: market_data_dict[symbol]
            for symbol in symbols
            if symbol in market_data_dict
        }
    if not market_data_dict:
        Logger.error("No valid symbols available for backtesting.")
        return

    model, scaler = _load_model_and_scaler()
    simulator = BacktestSimulator(model, scaler)
    simulator.simulate_best_symbol_per_day(market_data_dict)
    simulator.save_predictions_to_csv()
    simulator.summarize()


if __name__ == "__main__":
    local_market_data = MarketDataHandler.load_market_data()
    # backtest_same_symbol_all_days("TSLA", local_market_data)
    # backtest_best_symbol_per_day(xtb_symbols, local_market_data)
    # backtest_best_symbol_per_day(mercado_pago_symbols, local_market_data)
    backtest_same_symbol_all_days("TSLA", local_market_data)
