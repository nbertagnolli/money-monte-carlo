import numpy as np
from functools import partial
from typing import List
import dataclasses


def conservative_average_withdrawl(
    floor: int,
    ceil_withdrawl_rate: float,
    year_index: int,
    portfolio,
    performance: np.array,
) -> np.array:
    """Calculate vector of withdrawl rates over samples based on yearly performance.

    If the rate would put you under a specified minimum set the rate to achieve that
    minimum. This method will take in an index, portfolio, and performance matrix and
    return a vector of withdrawl rates for this position in time.

    :param floor: _description_
    :param ceil_withdrawl_rate: _description_
    :param year_index: The year index to withdrawl.
    :param portfolio: The (runs, years)  matrix of portfolio values.
    :param performance: The (runs, years) matrix of market performance.
    :return: The array (runs,) of withdrawl rates for year year_index.
    """
    # Create a vector of constant withdraw rates
    withdrawl_rates = (np.ones((portfolio.shape[0], 1)) * ceil_withdrawl_rate).flatten()

    # Calculate the average performance up until the current year for all years
    average_cumulative_performance = np.cumsum(performance, axis=1) / np.cumsum(
        np.ones(performance.shape), axis=1
    )

    # Find the x,y coordinates of all instances where the cumulative average performance
    # is less than the ceil_withdrawl_rate
    rows = np.where(
        (average_cumulative_performance[:, year_index - 1] < ceil_withdrawl_rate)
        | (ceil_withdrawl_rate * portfolio[:, year_index - 1] < floor)
    )

    # Fill those in with the actual minimum
    for row in rows[0]:
        withdrawl_rates[row] = floor / portfolio[row, year_index - 1]

    return abs(withdrawl_rates)


def constant_withdrawl_rate(
    rate: float, year_index: int, portfolio: np.array, performance: np.array
) -> np.array:
    """_summary_

    :param rate: The constant rate to withdrawl at.
    :param year_index: The year index to withdrawl.
    :param portfolio: The (runs, years)  matrix of portfolio values.
    :param performance: The (runs, years) matrix of market performance.
    :return: The array (runs,) of withdrawl rates for year year_index.
    """
    return abs((np.ones((portfolio.shape[0], 1)) * rate).flatten())


def constant_withdrawl_amount(
    floor, year_index: int, portfolio, performance: np.array
) -> np.array:
    """Return the rate of withdrawl which yields a cash return of floor.

    :param floor: The minimum cash amount to return.
    :param year_index: The year index to withdrawl.
    :param portfolio: The (runs, years)  matrix of portfolio values.
    :param performance: The (runs, years) matrix of market performance.
    :return: The array (runs,) of withdrawl rates for year year_index.
    """
    # Create a vector of constant withdraw rates
    withdrawl_rates = (np.ones((portfolio.shape[0], 1)) * floor).flatten()

    return abs(withdrawl_rates / portfolio[:, year_index - 1])


def vanguard_withdrawl(
    absolute_floor: int,
    target_withdrawl_rate: float,
    floor_withdrawl_rate: float,
    ceil_withdrawl_rate: float,
    year_index: int,
    portfolio,
    performance: np.array,
) -> np.array:
    """Calculate a vector of withdrawl rates over samples based on yearly performance.

    - Set a ceiling and a floor for changes to a retired client's spending amount each
        year.
    - Keep withdrawals within a manageable window of variability relative to the previous
        year, based on parameters you and the client agree on.
    - Increase spending after high performance years and decrease it after low performance
        years up to the limits set.

    :param floor: _description_
    :param ceil_withdrawl_rate: _description_
    :param year_index: The year index to withdrawl.
    :param portfolio: The (runs, years)  matrix of portfolio values.
    :param performance: The (runs, years) matrix of market performance.
    :return: The array (runs,) of withdrawl rates for year year_index.
    """
    # Create a vector of constant withdraw rates
    withdrawl_rates = (
        np.ones((portfolio.shape[0], 1)) * target_withdrawl_rate
    ).flatten()

    # Get the performance of the previous year
    previous_performance = performance[:, year_index - 1]

    for row in range(0, portfolio.shape[0]):
        previous_performance = performance[row, year_index - 1]
        if previous_performance < target_withdrawl_rate:
            withdrawl_rates[row] = max(
                floor_withdrawl_rate, absolute_floor / portfolio[row, year_index - 1]
            )
        elif previous_performance * 0.9 > target_withdrawl_rate:
            withdrawl_rates[row] = min(ceil_withdrawl_rate, previous_performance * 0.9)

    return withdrawl_rates


def save_constant_amount(amount, year_index, portfolio, perfomrance):
    """Return the withdrawl rate required to save the fixed amount.

    :param amount: The amount to save each year.
    :param year_index: The year index to withdrawl.
    :param portfolio: The (runs, years)  matrix of portfolio values.
    :param performance: The (runs, years) matrix of market performance.
    :return: The array (runs,) of withdrawl rates for year year_index.
    """
    save_amounts = np.ones(portfolio.shape[0]) * amount
    return -1 * save_amounts / portfolio[:, year_index - 1]


def sample_annual_return(
    sigma: float, distribution: List[float], num_runs: int, num_years: int
) -> np.array:
    """Sample a random annual return from the discreate distrbution of returns.

    :param sigma: The STD by which to permute the random sample. It affects the
        randomness of the returned values.
    :param distribution: The distribution of historical returns used to sample from.
    :param num_runs: The number of desired Monte Carlo runs.
    :param num_years: The number of years to experiment over.
    :return: The (num_runs, num_years) matrix of yearls stock market performance.
    """
    samples = np.random.choice(distribution, num_runs * num_years)
    random_offset = sigma * np.random.randn(num_runs * num_years)
    return (samples + random_offset).reshape((num_runs, num_years))


def count_busts(simulated_balances: np.array, minimum_balance: int = 120000) -> int:
    """Count the number of yearly returns less than minimum_balance across runs.

    :param simulated_balances: The (runs, years) matrix of spending.
    :param minimum_balance: The minimum amount that is acceptable to spend per year.
    :return: The count of runs which experienced at least one failure year.
    """
    n_busts_per_run = np.sum(np.round(simulated_balances) < minimum_balance, axis=1)
    return np.sum(n_busts_per_run > 0)


# Slightly more concise and extensible version of what you had before : )
def monte_carlo(
    sample_f, num_runs: int, num_years: int, initial_assets: float, withdrawl_rate_fn
):
    """Run monte carlo simulation of monetary growth over time.

    Thie method can run motne carlo simulations of financial assets over time. It can
    be used to run withdrawl experiments for retirment or predictions of total earnings
    over time. If the returned withdrawl rates from the withdrawl_rate_fn are negative
    then that will simulate saving over time instead of withdrawing.

    :param sample_f: The sampling function which takes in a number of runs and a number
        of years and returns a matrix of yearly stock market performances.
    :param num_runs: The number of Monte Carlo experiments to run.
    :param num_years: The number of years that the experiment should be run over.
    :param initial_assets: The starting balance.
    :param withdrawl_rate_fn: The function which takes the current year, portfolio,
        and performance and returns a vector of withdrawl rates for the next time stamp.
    :return portfolio:
    :return spend:
    :return performance:
    """
    # The total amount of simulated money in the portfolio.
    # Rows are trials and columns represent years.
    portfolio = np.ones((num_runs, num_years), dtype=np.float64) * initial_assets

    # The amount spent in that simulated year
    spend = np.zeros((num_runs, num_years), dtype=np.float64)

    # The performance sample each year
    performance = sample_f(num_runs, num_years)

    for j in range(1, num_years):
        # Vectorize the computation and do all runs simultaneously :)
        spend[:, j] = withdrawl_rate_fn(j, portfolio, performance) * portfolio[:, j - 1]
        # TODO:: SAMPLES SHOULD NEVER BE NEGATIVE WE END UP WITH NEGATIVE PORTFOLIO VALUES
        portfolio[:, j] = (portfolio[:, j - 1] - spend[:, j]) * (1 + performance[:, j])

    return portfolio, spend, performance


@dataclasses.dataclass
class MCEvaluation:
    spending_mean: float
    spending_std: float
    average_withdrawl_rate: float
    average_ending_value: float
    minimum_spend: float
    probability_of_failure: float

    def __str__(self):
        """Pretty printing of the dataclass."""
        return "\n".join(
            [
                f"{field.name}: {getattr(self, field.name)}"
                for field in dataclasses.fields(self)
            ]
        )

    def to_dict(self):
        return {k: str(v) if k == "message_id" else v for k, v in self.__dict__.items()}


def evaluate_run(
    portfolio: np.array,
    spend: np.array,
    performance: np.array,
    min_balance: float,
) -> np.array:
    """Create nice summary statistics from a Monte Carlo experiment."""
    num_runs = portfolio.shape[0]
    return MCEvaluation(
        spending_mean=np.mean(spend),
        # spending_median=np.median(spend),
        spending_std=np.std(spend),
        average_withdrawl_rate=np.mean(spend[:, 1:] / portfolio[:, :-1]),
        average_ending_value=np.mean(portfolio[:, -1]),
        minimum_spend=np.nanmin(spend) if np.nanmin(spend) >= 0 else 0,
        probability_of_failure=count_busts(spend[:, 1:], min_balance) / num_runs,
    ).to_dict()


def run_monte_carlo(
    past_performance: str,
    num_years: str,
    initial_assets: str,
    withdrawl_rate: str,
    minimum_withdrawl: str,
    withdrawl_strategy_name: str,
):
    num_runs = 80000
    num_years = int(num_years)
    initial_assets = int(initial_assets)
    withdrawl_rate = float(withdrawl_rate)
    minimum_withdrawl = int(minimum_withdrawl)
    past_performance = np.array(past_performance)
    withdrawl_strategies = {
        "Constant Saving Amount": partial(save_constant_amount, minimum_withdrawl),
        "Constant Withdrawl Rate": partial(constant_withdrawl_rate, withdrawl_rate),
        "Constant Withdrawl Amount": partial(
            constant_withdrawl_amount, minimum_withdrawl
        ),
        "Vanguard Method": partial(
            vanguard_withdrawl,
            minimum_withdrawl + 1,
            withdrawl_rate,
            0.04,
            0.06,
        ),
        "Dynamic Minimum Withdrawl": partial(
            conservative_average_withdrawl,
            minimum_withdrawl + 1,
            withdrawl_rate,
        ),
    }

    # Run the experiment
    portfolio, spend, performance = monte_carlo(
        partial(sample_annual_return, 0.02, past_performance.tolist()),
        num_runs,
        num_years,
        initial_assets,
        withdrawl_strategies[withdrawl_strategy_name],
    )

    # Evaluate the results
    results = evaluate_run(portfolio, spend, performance, minimum_withdrawl)

    # Remove nonsensical aggregate statistics if saving and add in better stats.
    if "Saving" in withdrawl_strategy_name:
        to_remove = [
            "spending_mean",
            "probability_of_failure",
            "spending_std",
            "average_withdrawl_rate",
            "minimum_spend",
        ]
        for name in to_remove:
            results.pop(name)
        results["min_portfolio_value"] = np.min(portfolio[:, -1])
        results["max_portfolio_value"] = np.max(portfolio[:, -1])
        results["median_portfolio_value"] = np.median(portfolio[:, -1])
        results["percentile_10"] = np.percentile(portfolio[:, -1], 10)
        results["percentile_50"] = np.percentile(portfolio[:, -1], 50)
        results["percentile_90"] = np.percentile(portfolio[:, -1], 90)

    results["portfolio"] = portfolio.tolist()
    results["spend"] = spend.tolist()
    results["performance"] = performance.tolist()
    return results
