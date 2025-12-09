# Bitcoin Transaction Fee Prediction Model
## Slideshow Architecture & Presentation Structure

---

## **SLIDE 1: Title Slide**
**Bitcoin Transaction Fee Prediction: A Structural Model Approach**

- Subtitle: "Two-Stage Estimation with 15M Transaction Dataset"
- Date: November 2025
- Author/Institution
- Key visual: Bitcoin blockchain visualization or fee rate distribution

---

## **SLIDE 2: Executive Summary / Key Findings**
**High-Level Results**

- **Dataset**: ~15 million transactions from November 2025
- **Approach**: Two-stage structural model (Equation 4)
- **Key Drivers**: Transaction value, blockspace utilization, user impatience, exchange activity
- **Model Performance**: R² ~0.11 (median regression), captures structural relationships
- **Main Insight**: Fee dynamics driven by congestion, transaction characteristics, and user behavior

**Visual**: Key metrics dashboard or summary statistics

---

## **SLIDE 3: Research Questions & Objectives**
**What We Set Out to Answer**

1. How well does the structural model (Equation 4) predict Bitcoin fees with new data?
2. What are the key drivers of transaction fees?
3. How do different transaction types affect fees?
4. How does performance compare to previous datasets?

**Visual**: Research question icons or flowchart

---

## **SLIDE 4: Data Overview**
**The Dataset**

- **Source**: SQLite database (`11-24-2025-15m-data-lake.db`)
- **Size**: 14,984,540 transactions
- **Sampling**: 2,000 blocks (~5M transactions analyzed)
- **Time Period**: August - November 2025
- **Key Variables**:
  - Transaction fees (fee_rate)
  - Transaction characteristics (weight, size, value)
  - Mempool state (mempool_size, mempool_tx_count)
  - Timing (found_at, mined_at, waittime)
  - Exchange flags (FromExchange, ToExchange)

**Visual**: Data pipeline diagram or sample transaction statistics

---

## **SLIDE 5: Two-Stage Estimation Framework**
**Methodology Overview**

### **Stage 1: Feature Engineering (Phase 2)**
- **Wait Time Prediction** (W_hat): Random Forest model predicting expected wait time
- **Congestion Metrics** (ρ_t): Time-weighted mempool congestion per epoch
- **User Impatience** (time_cost): Proxy from min_respend_blocks
- **Quantile Distribution** (F_tq): Upper-tail probability distribution

### **Stage 2: Fee Prediction (Phase 3)**
- **Structural Model**: Equation 4 with derived features from Stage 1
- **Multiple Specifications**: OLS, Quantile Regression, Gamma GLM, Splines

**Visual**: Two-stage pipeline diagram with data flow

---

## **SLIDE 6: Equation 4 - The Structural Model**
**The Core Specification**

```
fee_it = α₁ + α₂·ρ̂_t + α₃·riemann_sum + α₄·V_it + α₅·Weight_it + 
         α₆·FromExchange + α₇·ToExchange + α₈·Blockspace_t + 
         α₉·NFT_it + wallet_dummies + ε_it
```

**Variable Definitions**:
- **ρ̂_t**: Mempool congestion (time-weighted)
- **riemann_sum**: Structural term capturing expected wait for similar/higher impatience users
- **V_it**: Transaction value (total output amount)
- **Weight_it**: Transaction weight (vbytes)
- **FromExchange/ToExchange**: Exchange transaction flags
- **Blockspace_t**: Block space utilization at confirmation
- **NFT_it**: NFT transaction indicator

**Visual**: Equation with color-coded components

---

## **SLIDE 7: Feature Engineering - Phase 2**
**Derived Features**

### **1. Wait Time Prediction (W_hat)**
- **Model**: Random Forest Regressor
- **Features**: log(ρ_t), time_cost, has_child, rbf_flag
- **Output**: Predicted wait time in seconds
- **Performance**: R² ~0.40-0.50

### **2. Congestion Metric (ρ_t)**
- **Definition**: Time-weighted mempool transaction count per epoch
- **Epoch**: 30-minute windows
- **Purpose**: Captures mempool state at transaction entry

### **3. User Impatience (time_cost)**
- **Proxy**: 1 / (min_respend_blocks + ε)
- **Quantile Bins**: 1-10 per epoch
- **Interpretation**: Higher quantile = more impatient users

### **4. Tail Distribution (F_tq)**
- **Definition**: Upper-tail probability for each quantile
- **Use**: Component of Riemann sum calculation

**Visual**: Feature engineering pipeline with example calculations

---

## **SLIDE 8: Structural Terms - Riemann Sum**
**The Key Innovation**

**Riemann Sum**: Captures expected wait time for users with similar or higher impatience levels

**Computation**:
1. Group transactions by epoch and time_cost_quantile
2. Compute mean wait time (W_hat_q) per quantile
3. Calculate probability mass per quantile
4. Cumulative sum from quantile q to 10: Σ(prob_q × W_hat_q)

**Interpretation**: 
- Higher riemann_sum → More competition from impatient users
- Structural term that doesn't require fee data (avoids endogeneity)

**Visual**: Riemann sum calculation diagram or formula breakdown

---

## **SLIDE 9: Exchange Detection**
**Identifying Exchange Transactions**

### **FromExchange (α₆)**
- **Method**: Proxy based on transaction structure
- **Indicators**:
  - High transaction value (95th percentile)
  - Large transaction weight (90th percentile)
  - Round output amounts (divisible by 0.001 BTC)
- **Rationale**: Exchange withdrawals have characteristic patterns

### **ToExchange (α₇)**
- **Method**: Direct address matching
- **Source**: Known exchange addresses from Bithypha.com
- **Coverage**: 50+ exchanges, 100+ addresses
- **Detection**: Parse tx_data hex → extract output addresses → match against database

**Visual**: Exchange detection flowchart or address matching example

---

## **SLIDE 10: Blockspace Utilization**
**Competition for Block Space**

**Definition**: Cumulative weight of transactions confirmed before transaction i in the same block

**Computation**:
1. Sort transactions by block (conf_block_hash) and entry time (found_at)
2. Calculate cumulative weight within each block
3. Normalize by total block weight

**Interpretation**:
- Higher blockspace_t → Transaction confirmed later in block
- Captures intra-block competition
- Proxy for transaction priority

**Visual**: Block structure diagram showing cumulative weight

---

## **SLIDE 11: Data Quality & Preprocessing**
**Data Cleaning Steps**

### **Quality Checks**:
- Missing values: <1% for key variables
- Outlier removal: Top 0.1% fee rates (99.9th percentile)
- Block structure: 2,000 blocks, avg 2,500 txs/block
- Block fullness: ~60% of max block weight

### **Preprocessing**:
- Log transformations: Applied to skewed variables (ρ_t, V_it, weight)
- Standardization: Features scaled for model training
- Train/Test Split: 80/20 random split

**Visual**: Data quality dashboard or missing value heatmap

---

## **SLIDE 12: Exploratory Data Analysis**
**Understanding the Data**

### **Fee Rate Distribution**:
- Highly right-skewed
- Median: ~2-5 sat/vB
- Mean: ~10-20 sat/vB
- Max: Clipped at 99.9th percentile

### **Key Correlations**:
- **has_child**: -0.19 (CPFP transactions pay lower fees)
- **absolute_fee**: +0.27 (expected)
- **W_hat**: -0.05 (counterintuitive - needs explanation)
- **rho_t**: -0.01 (weak, may need transformation)

**Visual**: Correlation heatmap, distribution plots, scatter plots

---

## **SLIDE 13: Model Specifications**
**Four Modeling Approaches**

### **Model 1: OLS Regression**
- **Specification**: Linear model with log-transformed target
- **Features**: All Equation 4 variables + interactions
- **Use Case**: Baseline, interpretable coefficients

### **Model 2: Quantile Regression**
- **Quantiles**: 50th (median), 90th (upper tail)
- **Advantage**: Robust to outliers, captures tail behavior
- **Use Case**: Understanding fee dynamics at different percentiles

### **Model 3: Gamma GLM**
- **Distribution**: Gamma (right-skewed, positive)
- **Link**: Log link (multiplicative effects)
- **Advantage**: Natural fit for fee data, handles heteroscedasticity
- **Use Case**: Structural interpretation with proper distributional assumptions

### **Model 4: Spline Regression**
- **Method**: B-splines with cubic basis
- **Knots**: 8 knots (7 segments)
- **Features**: Non-linear for key variables (ρ_t, V_it, W_hat, blockspace)
- **Advantage**: Captures regime-switching behavior
- **Use Case**: Non-linear relationships, congestion regimes

**Visual**: Model comparison table or flowchart

---

## **SLIDE 14: Model Results - OLS**
**Baseline Linear Model**

### **Performance**:
- **R²**: ~0.11-0.12
- **MAE**: ~2.0 sat/vB
- **Median AE**: ~0.9 sat/vB

### **Key Coefficients** (significant):
- **has_child**: -0.71 (CPFP pays less)
- **log_V_it**: +0.52 (value premium)
- **log_blockspace_t**: +0.22 (competition premium)
- **log_W_hat**: -0.12 (wait time discount)

### **Interpretation**:
- Model captures ~11% of fee variation
- Structural terms significant
- Exchange effects present but small

**Visual**: Coefficient plot with confidence intervals

---

## **SLIDE 15: Model Results - Quantile Regression**
**Median vs Upper Tail**

### **Median Regression (50th percentile)**:
- **Pseudo-R²**: 0.11
- **MAE**: 1.96 sat/vB
- **Key Drivers**: has_child (-0.71), log_V_it (+0.52)

### **90th Percentile Regression**:
- **Pseudo-R²**: 0.08
- **MAE**: 3.95 sat/vB
- **Key Drivers**: has_child (-1.27), log_V_it (+1.47)

### **Insights**:
- Upper tail shows stronger effects (2x coefficient magnitudes)
- Value premium increases in high-fee regime
- CPFP discount larger for high-fee transactions

**Visual**: Coefficient comparison (median vs 90th percentile) bar chart

---

## **SLIDE 16: Model Results - Gamma GLM**
**Multiplicative Effects**

### **Performance**:
- **R²**: 0.06
- **MAE**: 2.31 sat/vB
- **Deviance**: 108,787

### **Key Coefficients** (exp(coef) = multiplicative effect):
- **has_child**: 0.65 (35% fee reduction)
- **log_V_it**: 1.46 (46% increase per log unit)
- **log_blockspace_t**: 1.14 (14% increase per log unit)
- **log_W_hat**: 0.89 (11% reduction per log unit)

### **Advantages**:
- Natural distributional fit
- Multiplicative interpretation
- Handles heteroscedasticity

**Visual**: Multiplicative effect visualization or coefficient plot

---

## **SLIDE 17: Model Results - Spline Regression**
**Non-Linear Relationships**

### **Performance**:
- **R²**: ~0.10-0.12
- **MAE**: ~2.0 sat/vB
- **Features**: 4 spline variables × 10 basis functions = 40+ features

### **Key Findings**:
- Non-linear effects in congestion (ρ_t)
- Value effects vary by regime
- Blockspace competition shows threshold effects

### **Regime Switching**:
- Low congestion: Linear fee dynamics
- High congestion: Non-linear, steeper fee increases
- Captures "fee market breakdown" scenarios

**Visual**: Spline plots showing non-linear relationships for key variables

---

## **SLIDE 18: Feature Importance Analysis**
**What Drives Fees?**

### **Top 10 Features** (by absolute coefficient):

1. **has_child** (-0.71): CPFP transactions pay less
2. **log_V_it** (+0.52): Value premium (high-value txs pay more)
3. **V_it** (+0.29): Linear value effect
4. **log_time_cost_quantile** (+0.26): Impatience premium
5. **log_blockspace_t** (+0.22): Blockspace competition
6. **log_weight** (-0.14): Size discount (larger txs pay less per vB)
7. **log_W_hat** (-0.12): Wait time discount
8. **log_rho_t** (+0.07): Congestion effect (weak)
9. **rbf_flag** (-0.06): RBF transactions pay slightly less
10. **log_riemann_sum** (+0.05): Structural competition term

**Visual**: Feature importance bar chart or waterfall plot

---

## **SLIDE 19: Exchange Effects**
**Exchange Transaction Analysis**

### **FromExchange (α₆)**:
- **Prevalence**: ~5-10% of transactions
- **Effect**: Small positive (exchange withdrawals pay premium)
- **Interpretation**: Exchanges prioritize speed, pay higher fees

### **ToExchange (α₇)**:
- **Prevalence**: ~3-5% of transactions
- **Effect**: Small negative (deposits pay less)
- **Interpretation**: Deposits less time-sensitive, can wait

### **Exchange Types Detected**:
- Major: Binance, Coinbase, Kraken, Bitfinex, OKX
- Mid-size: Bybit, HTX, KuCoin, Gemini
- Regional: Cash App, Robinhood, Upbit, Bithumb

**Visual**: Exchange transaction statistics or fee comparison

---

## **SLIDE 20: CPFP & RBF Effects**
**Transaction Type Analysis**

### **CPFP (Child Pays for Parent)**:
- **Indicator**: has_child = 1
- **Prevalence**: ~27% of transactions
- **Fee Effect**: -0.71 (significant discount)
- **Interpretation**: CPFP transactions pay less because parent already paid

### **RBF (Replace-by-Fee)**:
- **Indicator**: rbf_flag = 1
- **Prevalence**: ~0.9% of transactions
- **Fee Effect**: -0.06 (small discount)
- **Interpretation**: RBF users can adjust fees, pay less initially

**Visual**: Transaction type comparison or fee distribution by type

---

## **SLIDE 21: Blockspace Competition**
**Intra-Block Dynamics**

### **Blockspace Utilization Distribution**:
- Mean: ~60% of max block weight
- Distribution: Right-skewed (some blocks full, many partially full)
- Competition: Higher utilization → higher fees

### **Effect on Fees**:
- **log_blockspace_t**: +0.22 coefficient
- **Interpretation**: 10% increase in blockspace → 2.2% increase in fees
- **Non-linear**: Spline model shows threshold effects

### **Block Structure**:
- Transactions confirmed later in block pay more
- Captures miner's transaction ordering strategy
- Proxy for transaction priority

**Visual**: Blockspace utilization histogram or fee vs blockspace scatter

---

## **SLIDE 22: Congestion Effects**
**Mempool State Impact**

### **Congestion Metric (ρ_t)**:
- **Definition**: Time-weighted mempool transaction count
- **Distribution**: Highly variable (100s to 100,000s)
- **Effect**: Weak positive (+0.07), but significant

### **Riemann Sum**:
- **Definition**: Structural competition term
- **Effect**: Small positive (+0.05)
- **Interpretation**: Captures competition from impatient users

### **Insights**:
- Direct congestion effect weak (may need better proxy)
- Structural terms capture competition better
- Non-linear effects in spline model

**Visual**: Congestion time series or fee vs congestion scatter

---

## **SLIDE 23: Value Premium**
**Transaction Value Effects**

### **Value Distribution**:
- Mean: ~178M sats (~1.78 BTC)
- Highly right-skewed
- Log transformation essential

### **Fee Effects**:
- **log_V_it**: +0.52 (strong positive)
- **V_it**: +0.29 (linear component)
- **Interpretation**: High-value transactions pay premium for security/speed

### **Quantile Analysis**:
- Median: +0.52 effect
- 90th percentile: +1.47 effect (3x stronger)
- Upper tail shows much stronger value premium

**Visual**: Value distribution or fee vs value scatter (log scale)

---

## **SLIDE 24: Model Comparison**
**Which Model Performs Best?**

### **Performance Metrics**:

| Model | R² | MAE | Median AE | Notes |
|-------|----|-----|-----------|-------|
| OLS | 0.11-0.12 | 2.0 | 0.9 | Baseline, interpretable |
| Quantile (50th) | 0.11 | 2.0 | 0.9 | Robust, median focus |
| Quantile (90th) | 0.08 | 4.0 | - | Captures tail |
| Gamma GLM | 0.06 | 2.3 | 1.3 | Distributional fit |
| Spline | 0.10-0.12 | 2.0 | 0.9 | Non-linear, flexible |

### **Trade-offs**:
- **OLS**: Best interpretability, good performance
- **Quantile**: Robust, captures tail behavior
- **Gamma GLM**: Proper distribution, multiplicative effects
- **Spline**: Captures non-linearities, regime switching

**Visual**: Model comparison radar chart or performance bar chart

---

## **SLIDE 25: Residual Analysis**
**Model Diagnostics**

### **Residual Patterns**:
- **Distribution**: Right-skewed (underpredicts high fees)
- **Heteroscedasticity**: Variance increases with predicted fees
- **Outliers**: High-fee transactions poorly predicted

### **Improvements Needed**:
- Better handling of extreme fees
- Non-linear transformations
- Interaction terms
- Time-of-day effects

**Visual**: Residual plots (residuals vs fitted, Q-Q plot, histogram)

---

## **SLIDE 26: Limitations & Challenges**
**What We Can't Explain**

### **Model Limitations**:
- **R² = 0.11**: Only captures 11% of fee variation
- **Missing Factors**: Miner preferences, time-of-day, network events
- **Endogeneity**: Some features may be endogenous
- **Data Quality**: Exchange detection proxy imperfect

### **Challenges**:
- **High Variance**: Fee market highly volatile
- **Non-Stationarity**: Fee dynamics change over time
- **Complex Interactions**: Many unmodeled interactions
- **Outliers**: Extreme fees hard to predict

**Visual**: Limitations checklist or challenges diagram

---

## **SLIDE 27: Comparison to Previous Dataset**
**Performance Evolution**

### **Previous Dataset** (if available):
- **Size**: [Previous size]
- **R²**: [Previous R²]
- **Key Differences**: [What changed]

### **New Dataset Improvements**:
- **Larger Sample**: 15M vs [previous] transactions
- **Better Features**: Exchange detection, blockspace utilization
- **Structural Terms**: Riemann sum, improved congestion metrics
- **Model Diversity**: Multiple specifications tested

### **Consistent Findings**:
- CPFP discount robust across datasets
- Value premium consistent
- Congestion effects weak but present

**Visual**: Comparison table or performance evolution chart

---

## **SLIDE 28: Policy Implications**
**What This Means for Bitcoin**

### **Fee Market Insights**:
1. **CPFP Works**: Child transactions pay less (fee efficiency)
2. **Value Premium**: High-value transactions pay more (security premium)
3. **Exchange Behavior**: Exchanges pay premiums for speed
4. **Congestion Weak**: Direct congestion effects small (structural terms better)

### **User Recommendations**:
- Use CPFP for fee efficiency
- High-value transactions: Pay premium for security
- Exchange withdrawals: Expect higher fees
- Monitor blockspace utilization

**Visual**: Policy implications diagram or recommendations list

---

## **SLIDE 29: Future Research Directions**
**Next Steps**

### **Model Improvements**:
1. **Better Congestion Metrics**: Improve ρ_t proxy
2. **Time Effects**: Add time-of-day, day-of-week
3. **Miner Behavior**: Model miner transaction selection
4. **Network Events**: Incorporate network congestion events

### **Feature Engineering**:
1. **Better Exchange Detection**: Improve FromExchange proxy
2. **NFT Detection**: Improve NFT identification
3. **Wallet Clustering**: Add wallet-level features
4. **Transaction Graph**: Network effects

### **Methodology**:
1. **Causal Inference**: Address endogeneity concerns
2. **Machine Learning**: Try XGBoost, Neural Networks
3. **Ensemble Methods**: Combine multiple models
4. **Real-time Prediction**: Deploy for live fee estimation

**Visual**: Research roadmap or future work diagram

---

## **SLIDE 30: Conclusion**
**Key Takeaways**

### **What We Learned**:
1. **Structural Model Works**: Equation 4 captures key fee drivers
2. **CPFP Discount**: Significant fee reduction for child transactions
3. **Value Premium**: High-value transactions pay more
4. **Exchange Effects**: Present but small
5. **Congestion Weak**: Direct effects small, structural terms better

### **Model Performance**:
- **R² = 0.11**: Modest but meaningful
- **MAE = 2.0 sat/vB**: Reasonable for median transactions
- **Robust Findings**: Consistent across model specifications

### **Contribution**:
- First large-scale structural fee model
- Novel structural terms (Riemann sum)
- Comprehensive feature engineering
- Multiple model specifications

**Visual**: Summary diagram or key findings visualization

---

## **SLIDE 31: Acknowledgments & References**
**Credits & Sources**

### **Data Sources**:
- SQLite database: `11-24-2025-15m-data-lake.db`
- Exchange addresses: Bithypha.com
- Bitcoin blockchain data

### **Tools & Libraries**:
- Python: pandas, numpy, scikit-learn, statsmodels
- Visualization: matplotlib, seaborn
- Database: SQLite

### **References**:
- [Academic papers on Bitcoin fees]
- [Previous fee prediction models]
- [Bitcoin fee market research]

**Visual**: Logo wall or reference list

---

## **APPENDIX SLIDES (Optional)**

### **A1: Technical Details**
- Database schema
- Feature engineering code snippets
- Model hyperparameters

### **A2: Additional Visualizations**
- Distribution plots
- Time series plots
- Correlation matrices
- Residual diagnostics

### **A3: Robustness Checks**
- Different train/test splits
- Subset analyses
- Sensitivity analysis

### **A4: Code Repository**
- GitHub link
- Documentation
- Reproducibility guide

---

## **VISUALIZATION RECOMMENDATIONS**

### **Key Charts Needed**:
1. **Data Pipeline Diagram**: Two-stage estimation flow
2. **Equation 4 Visualization**: Color-coded components
3. **Feature Importance**: Bar chart or waterfall
4. **Model Comparison**: Performance metrics table/chart
5. **Residual Diagnostics**: Multiple plots
6. **Distribution Plots**: Fee rates, key features
7. **Correlation Heatmap**: Feature relationships
8. **Time Series**: Fee evolution, congestion
9. **Scatter Plots**: Fee vs key predictors
10. **Spline Plots**: Non-linear relationships

### **Design Guidelines**:
- **Color Scheme**: Consistent palette (e.g., Bitcoin orange + complementary colors)
- **Typography**: Clear, readable fonts
- **Layout**: Clean, uncluttered slides
- **Data Density**: Balance detail with clarity
- **Accessibility**: High contrast, readable text

---

## **PRESENTATION FLOW**

### **Part 1: Introduction** (Slides 1-6)
- Title, summary, questions, data, methodology, equation

### **Part 2: Methodology** (Slides 7-12)
- Feature engineering, structural terms, exchange detection, data quality, EDA

### **Part 3: Results** (Slides 13-24)
- Model specifications, results for each model, feature importance, effects analysis

### **Part 4: Discussion** (Slides 25-30)
- Diagnostics, limitations, comparisons, implications, future work, conclusion

### **Part 5: Closing** (Slide 31)
- Acknowledgments, references

---

## **NOTES FOR PRESENTERS**

1. **Slide Timing**: ~2-3 minutes per slide (60-90 min total)
2. **Key Messages**: Emphasize structural model innovation, CPFP discount, value premium
3. **Technical Depth**: Adjust based on audience (academic vs industry)
4. **Q&A Prep**: Be ready to discuss R² = 0.11, endogeneity, model limitations
5. **Visuals**: Use plots from `plots/` folder, generate new ones as needed
6. **Code**: Reference notebook cells for technical details

---

**Document Version**: 1.0  
**Last Updated**: [Current Date]  
**Based on**: `phase3_fee_model_new_data.ipynb` analysis

