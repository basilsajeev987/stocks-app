import yfinance as yf
import numpy as np
import pandas as pd

# =========================
# RL TRADING AGENT
# =========================

ACTIONS = [

    "BUY",

    "SELL",

    "HOLD"
]

# =========================
# SIMPLE Q TABLE
# =========================

q_table = {}

# =========================
# GET STATE
# =========================

def get_state(
    price_change
):

    if price_change > 0.02:

        return "UP"

    elif price_change < -0.02:

        return "DOWN"

    else:

        return "SIDEWAYS"

# =========================
# RL TRAIN
# =========================

def train_rl_agent(
    ticker
):

    try:

        stock = yf.Ticker(
            ticker
        )

        hist = stock.history(
            period="1y"
        )

        if hist.empty:

            return {
                "error":
                    "No Data"
            }

        hist["Returns"] = (

            hist["Close"]

            .pct_change()
        )

        hist.dropna(inplace=True)

        alpha = 0.1

        gamma = 0.9

        epsilon = 0.1

        total_reward = 0

        # =========================
        # TRAIN LOOP
        # =========================

        for i in range(
            1,
            len(hist)
        ):

            current_return = float(

                hist["Returns"]
                .iloc[i]
            )

            state = get_state(
                current_return
            )

            # =========================
            # INIT STATE
            # =========================

            if state not in q_table:

                q_table[state] = {

                    action: 0

                    for action
                    in ACTIONS
                }

            # =========================
            # EPSILON GREEDY
            # =========================

            if np.random.rand() < epsilon:

                action = np.random.choice(
                    ACTIONS
                )

            else:

                action = max(

                    q_table[state],

                    key=q_table[state].get
                )

            # =========================
            # REWARD
            # =========================

            reward = 0

            if action == "BUY":

                reward = current_return * 100

            elif action == "SELL":

                reward = (
                    -current_return
                ) * 100

            elif action == "HOLD":

                reward = 0

            total_reward += reward

            # =========================
            # NEXT STATE
            # =========================

            if i < len(hist) - 1:

                next_return = float(

                    hist["Returns"]
                    .iloc[i + 1]
                )

                next_state = get_state(
                    next_return
                )

            else:

                next_state = state

            if next_state not in q_table:

                q_table[next_state] = {

                    action: 0

                    for action
                    in ACTIONS
                }

            # =========================
            # Q UPDATE
            # =========================

            old_value = q_table[
                state
            ][action]

            next_max = max(

                q_table[
                    next_state
                ].values()
            )

            new_value = (

                old_value

                + alpha * (

                    reward

                    + gamma * next_max

                    - old_value
                )
            )

            q_table[state][action] = (
                new_value
            )

        # =========================
        # FINAL SIGNAL
        # =========================

        latest_return = float(

            hist["Returns"]
            .iloc[-1]
        )

        latest_state = get_state(
            latest_return
        )

        best_action = max(

            q_table[
                latest_state
            ],

            key=q_table[
                latest_state
            ].get
        )

        confidence = abs(

            q_table[
                latest_state
            ][best_action]
        )

        confidence = min(
            confidence,
            99
        )

        return {

            "ticker":
                ticker,

            "signal":
                best_action,

            "state":
                latest_state,

            "confidence":
                round(
                    confidence,
                    2
                ),

            "reward":
                round(
                    total_reward,
                    2
                ),

            "model":
                "Q-Learning RL Agent",
        }

    except Exception as e:

        print(
            "RL ERROR:",
            e
        )

        return {
            "error":
                str(e)
        }