# 💰 VibeTrading Hackathon: Machine Learning Trading Challenge

Welcome to the **VibeTrading Hackathon**! Your mission is to develop a machine learning strategy that correctly captures the market's "vibe" (momentum and reversals) to generate **Alpha** returns greater than the simple Buy and Hold benchmark.
## Setup Instructions

If you already have the required tools installed you can skip this step.
- **Python:** The programming language you'll be using to write your AI.
- **VSCode (or other code editor):** Editor to write and edit your Python code.
- **Git:** Version control to download the hackathon code and save changes.
Choose your operating system for full setup instructions:

- [**Windows Setup Guide**](read_me/Windows_setup.md)
- [**macOS Setup Guide**](read_me/Mac_setup.md)
- [**Linux Setup Guide**](read_me/Linux_setup.md)

## 🏆 The Challenge: Maximize Alpha and Control Risk
- **Primary Metric:** **Alpha** = (Bot Return) - (Buy and Hold Return). Higher is better.

All code editing will be in this repo. The competition dashboard is available here: https://quant-trading.lovable.app/
## 🐍 The 5 Starter Bots (The VibeTrading Team)

You have five pre-built Machine Learning bots, each using a distinct core algorithm to beat the market. Bot files live under the `bot_*` folders.
| Bot Name | Technique | Approach | Key sklearn Model |
| :--- | :--- | :--- | :--- |
| **Hyperplane Hippie** 🔄 | Multi-Layer Perceptron | Regression (Predicts return magnitude) | `MLPRegressor` |
| **Lord of the Bins** 🗑️ | Logistic Regression | Classification (Predicts up/down probability) | `LogisticRegression` |
| **The Baguette Booster** 🥖 | Random Forest | Classification (Ensemble voting) | `RandomForestClassifier` |
| **KNN Cash Cow** 🐄 | K-Nearest Neighbors | Classification (Proximity / historical match) | `KNeighborsClassifier` |
| **"My cool bot"** 📐 | Up to you | Classification / Regression | (To be implemented) |

---
## 🛠️ How to Improve the Bots (The Three Levers)

To generate Alpha, modify the strategy templates using these three levers.
1. Prediction Horizon & Target Tuning
    - **Location:** Look for `N_DAYS_PREDICT` in the `--- CONFIGURATION ---` section.
    - **Goal:** Adjust horizon (short-term flips vs long-term trends).
    - **Classification bots:** Tune the confidence threshold (e.g., only act if predicted "Up" probability ≥ 55%).

2. Feature Engineering (The Alpha Source)
    - **Location:** See `SECTION A: FEATURE ENGINEERING` in each bot. Use functions from `technical_indicators.py`.
    - **Goal:** Create predictive inputs (X) for the model. Starter bots use simple moving-average differences.
    - **Ideas:** RSI, MACD components, ATR, volume-adjusted volatility, market-relative features (e.g., SP500 comparison).
    - **Important:** If you add features, update the `FEATURE_COLS` list inside the strategy file.

3. Model Hyperparameter Tuning
    - **Location:** See `SECTION B: TRAIN MODEL` in each bot.
    - **Goal:** Adjust algorithm settings to fit data and avoid overfitting.
    - **Examples:** MLP `hidden_layer_sizes` / `max_iter`; Random Forest `n_estimators` / `max_depth`; KNN `n_neighbors`.
---

## ⚙️ How to Run Your Strategy

1. Open the bot file (for example: `bot_knn_cash_cow/knn_cash_cow.py`).
2. Edit the code below the "EDIT DOWNWARDS FROM THIS POINT" marker to implement changes.
3. Run the script from the project root:

```bash
python bot_knn_cash_cow/knn_cash_cow.py
```

The script prints metrics and saves an analysis plot. Final `.joblib` submission files are placed in the `/submissions` folder.
### 🖼️ Interpreting the Alpha Plot

- **Top Panel:** Stock price, Strategy Equity (red), and Buy & Hold Equity (black dashed). Green triangles show buy signals.
- **Bottom Panel (Alpha Spread):** Strategy equity minus Buy & Hold. If the purple line is above zero, the strategy is outperforming.
## Tips and Tricks

Read more tips here: [**Tips and Tricks**](read_me/Tips_N_Tricks.md)
# 💰 VibeTrading Hackathon: Machine Learning Trading Challenge

Welcome to the **VibeTrading Hackathon**! Your mission is to develop a machine learning strategy that correctly captures the market's "vibe" (momentum and reversals) to generate **Alpha** returns greater than the simple Buy and Hold benchmark (Return better than simple holding the Stock).

## Setup Instructions

Here are the prerequisites for the hackaton. If you already have this installed you can skip this step.

**Python:** The programming language you'll be using to write your AI. 

**VSCode (or other code editor):** The code editor you'll use to write and edit your Python code.  

**Git:** Version control tool you'll use to download the hackaton code and save your changes.

Choose your operating system for complete setup instructions:

- [**Windows Setup Guide**](read_me/Windows_setup.md) - Complete installation guide for Windows
- [**macOS Setup Guide**](read_me/Mac_setup.md) - Complete installation guide for Mac
- [**Linux Setup Guide**](read_me/Linux_setup.md) - Complete installation guide for Linux

## 🏆 The Challenge: Maximize Alpha and Control Risk

The winner will be the team that creates the most profitable bot that successfully manages risk.

* **Primary Metric:** **Alpha** (The bot's return - Buy and Hold Return). (Higher is better).

All code editing will be in this repo. 
**The competition will take place here** :  [Dashboard](https://quant-trading.lovable.app/)

## 🐍 The 5 Starter Bots (The VibeTrading Team)

You have five pre-built Machine Learning bots, each using a distinct core algorithm to beat the market. The bots have the prefix bot_ in the begging.

| Bot Name | Technique | Approach | Key $\text{sklearn}$ Model |
| :--- | :--- | :--- | :--- |
| **The MLP MVP** 🔄 | Multi-Layer Perceptron | **Regression** (Predicts Return Magnitude) | `MLPRegressor` |
| **Lord of the Bins** 🗑️ | Logistic Regression | **Classification** (Predicts Up/Down Probability) | `LogisticRegression` |
| **The Baguette Booster** 🥖 | Random Forest | **Classification** (Ensemble Voting) | `RandomForestClassifier` |
| **KNN Cash Cow** 🐄 | K-Nearest Neighbors | **Classification** (Proximity/Historical Match) | `KNeighborsClassifier` |
| **"My cool bot"** 📐 | Up to the user | **Classification/Regression** | (To be implemented) |

---

## 🛠️ How to Improve the Bots

To generate Alpha, you must modify the strategy templates using the three primary levers listed below.
### 1. Prediction Horizon & Target Tuning
* **Location:** Look for the `N_DAYS_PREDICT` variable in the `--- CONFIGURATION ---` section.
* **Goal:** Adjust how far into the future the model tries to predict (e.g., short-term flips or long-term trends).
* **Classification Bots Only:** Tune the **confidence threshold** (e.g., in `Lord of the Bins`) to only buy if the model is highly confident (e.g., $55\%$ probability of "Up" or higher).
* 
### 2. Feature Engineering (The Alpha Source)
* **Location:** Look for the section `SECTION A: FEATURE ENGINEERING` in each bot. New features can be imported by using the technical_indicators.py.
* **Goal:** Create new, predictive input features ($\text{X}$) that the $\text{ML}$ model can learn from. The current bots only use a simple Moving Average Difference.
* **Ideas:**
    * **Momentum:** Relative Strength Index ($\text{RSI}$), $\text{MACD}$ components.
    * **Volatility:** Average True Range ($\text{ATR}$), volume-adjusted standard deviations.
    * **Market Context:** Compare the stock's performance to the overall $\text{SP 500}$ market.
* **Critical Step:** If you add a feature, **you MUST update the `FEATURE_COLS` list** inside the strategy file.

### 3. Model Hyperparameter Tuning
* **Location:** Look for the section `SECTION B: TRAIN MODEL`.
* **Goal:** Adjust the core settings of the $\text{ML}$ algorithm to fit the data better and reduce overfitting.
* **Examples:**
    * **MLP:** Change `hidden_layer_sizes`, `max_iter`.
    * **Random Forest:** Change `n_estimators` (number of trees), `max_depth`.
    * **KNN:** Change `n_neighbors` (the 'K' value).

---

## ⚙️ How to Run Your Strategy

1.  Open the corresponding Python file (e.g., `/bot_knn_cash_cow/knn_cash_cow.py`).
2.  Make your changes to in the suitable places (**EDIT DOWNWARDS FROM THIS POINT**.
3.  Run the file from your terminal (from the project root):
    ```bash
    python bot_knn_cash_cow/knn_cash_cow.py
    ```
4.  The script will print metrics to the console and save the visual analysis plot.
5.  The final submission file with ending .joblib will be located in the /submissions folder.

### 🖼️ Interpreting the Alpha Plot

The output plot is now split into two panels for clearer analysis:

* **Top Panel:** Shows the Stock Price, your **Strategy Equity** (Red), and the **Buy and Hold Equity** (Black Dashed Line). Green triangles show your bot's **Buy Signals**.
* **Bottom Panel (The Alpha Spread):** Shows the difference between your strategy's equity and Buy and Hold ($\text{Strategy} - \text{B and H}$).
    * If the purple line is **above the zero axis**, your bot is currently **outperforming** the market.
    * If the purple line is **trending upwards**, your bot is successfully generating Alpha during that period.
 
## Tips and Tricks
 
Read here for more tips and tricks towards a winning strategy: - [**Tips and Tricks**](read_me/Tips_N_Tricks.md)
