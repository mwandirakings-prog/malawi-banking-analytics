# ============================================================
# Malawi Banking Analytics Project
# File: src/banking_analysis.R
# Author: Kings Mwandira
# Description: Statistical Analysis of Malawian Banking Sector
#              Panel Data Regression + NPL Drivers + ggplot2
# ============================================================

# ── 1. Load Libraries ────────────────────────────────────────
library(tidyverse)
library(ggplot2)
library(dplyr)
library(readr)
library(corrplot)
library(RColorBrewer)
library(gridExtra)
library(scales)
library(plm)
library(lmtest)

cat("==========================================================\n")
cat("  Malawi Banking Sector — R Statistical Analysis\n")
cat("==========================================================\n\n")

# ── 2. Set Working Directory & Load Data ─────────────────────
setwd("C:/Users/MR MATUMBA/MalawiBANKING")

cat("[1/6] Loading banking data...\n")
df <- read_csv("data/processed/malawi_banking_data.csv",
               show_col_types = FALSE)

cat(paste("      Records loaded  :", nrow(df), "\n"))
cat(paste("      Columns         :", ncol(df), "\n"))
cat(paste("      Banks           :", n_distinct(df$bank_name), "\n"))
cat(paste("      Years           :", min(df$year), "to", max(df$year), "\n\n"))

# ── 3. Descriptive Statistics ─────────────────────────────────
cat("[2/6] Generating descriptive statistics...\n")

stats_summary <- df %>%
  summarise(
    Avg_ROA          = round(mean(return_on_assets_pct, na.rm=TRUE), 2),
    Avg_ROE          = round(mean(return_on_equity_pct, na.rm=TRUE), 2),
    Avg_NPL          = round(mean(npl_ratio_pct, na.rm=TRUE), 2),
    Avg_CAR          = round(mean(capital_adequacy_ratio_pct, na.rm=TRUE), 2),
    Avg_CostIncome   = round(mean(cost_to_income_ratio_pct, na.rm=TRUE), 2),
    Avg_Liquidity    = round(mean(liquidity_ratio_pct, na.rm=TRUE), 2),
    Max_NPL          = round(max(npl_ratio_pct, na.rm=TRUE), 2),
    Min_ROA          = round(min(return_on_assets_pct, na.rm=TRUE), 2),
    Max_ROA          = round(max(return_on_assets_pct, na.rm=TRUE), 2)
  )

cat("\n  CAMELS Summary Statistics:\n")
cat(paste("  Average ROA              :", stats_summary$Avg_ROA, "%\n"))
cat(paste("  Average ROE              :", stats_summary$Avg_ROE, "%\n"))
cat(paste("  Average NPL Ratio        :", stats_summary$Avg_NPL, "%\n"))
cat(paste("  Average Capital Adequacy :", stats_summary$Avg_CAR, "%\n"))
cat(paste("  Average Cost-to-Income   :", stats_summary$Avg_CostIncome, "%\n"))
cat(paste("  Average Liquidity Ratio  :", stats_summary$Avg_Liquidity, "%\n"))
cat(paste("  Highest NPL Ratio        :", stats_summary$Max_NPL, "%\n"))
cat(paste("  Best ROA                 :", stats_summary$Max_ROA, "%\n\n"))

# ── 4. Bank Performance Rankings ─────────────────────────────
cat("[3/6] Ranking banks by performance...\n")

bank_rankings <- df %>%
  group_by(bank_name) %>%
  summarise(
    Avg_ROA    = round(mean(return_on_assets_pct), 2),
    Avg_ROE    = round(mean(return_on_equity_pct), 2),
    Avg_NPL    = round(mean(npl_ratio_pct), 2),
    Avg_CAR    = round(mean(capital_adequacy_ratio_pct), 2),
    Avg_Cost   = round(mean(cost_to_income_ratio_pct), 2),
    Risk_Score = round(mean(npl_ratio_pct) - mean(return_on_assets_pct), 2)
  ) %>%
  arrange(Risk_Score)

cat("\n  Bank Performance Rankings (Best to Worst Risk Score):\n")
print(as.data.frame(bank_rankings))
cat("\n")

# ── 5. Panel Data Regression — ROA Drivers ───────────────────
cat("[4/6] Running panel data regression (ROA drivers)...\n")

# Convert to panel data format
panel_data <- pdata.frame(df,
                          index = c("bank_name", "year"))

# Fixed Effects Model
fe_model <- plm(
  return_on_assets_pct ~
    npl_ratio_pct +
    cost_to_income_ratio_pct +
    capital_adequacy_ratio_pct +
    liquidity_ratio_pct +
    loan_to_deposit_ratio_pct,
  data   = panel_data,
  model  = "within",
  effect = "individual"
)

cat("\n  Fixed Effects Panel Regression Results:\n")
cat("  Dependent Variable: Return on Assets (ROA %)\n\n")
print(summary(fe_model))

# ── 6. NPL Driver Analysis ────────────────────────────────────
cat("\n[5/6] Analysing NPL drivers...\n")

npl_model <- plm(
  npl_ratio_pct ~
    cost_to_income_ratio_pct +
    capital_adequacy_ratio_pct +
    liquidity_ratio_pct +
    loan_to_deposit_ratio_pct,
  data   = panel_data,
  model  = "within",
  effect = "individual"
)

cat("\n  NPL Driver Regression Results:\n")
cat("  Dependent Variable: NPL Ratio (%)\n\n")
print(summary(npl_model))

# ── 7. Visualizations ─────────────────────────────────────────
cat("\n[6/6] Creating visualizations...\n")

dir.create("data/processed/r_plots", showWarnings = FALSE)

# Plot 1 — ROA by Bank Over Time
p1 <- ggplot(df, aes(x = year,
                     y = return_on_assets_pct,
                     color = bank_name,
                     group = bank_name)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  labs(
    title    = "Return on Assets (ROA) by Bank — 2018 to 2023",
    subtitle = "Malawi Banking Sector Analysis",
    x        = "Year",
    y        = "ROA (%)",
    color    = "Bank"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", color = "#1B4F8A"),
    plot.subtitle = element_text(color = "#555555"),
    legend.position = "bottom"
  ) +
  scale_color_brewer(palette = "Set2")

ggsave("data/processed/r_plots/plot1_roa_trends.png",
       p1, width = 10, height = 6, dpi = 150)

# Plot 2 — NPL Ratio by Bank
p2 <- ggplot(df, aes(x = year,
                     y = npl_ratio_pct,
                     color = bank_name,
                     group = bank_name)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  geom_hline(yintercept = 5, linetype = "dashed",
             color = "red", size = 1) +
  annotate("text", x = 2018.3, y = 6,
           label = "RBM Threshold 5%",
           color = "red", size = 3.5) +
  labs(
    title    = "NPL Ratio by Bank — 2018 to 2023",
    subtitle = "Red dashed line = RBM maximum threshold (5%)",
    x        = "Year",
    y        = "NPL Ratio (%)",
    color    = "Bank"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", color = "#1B4F8A"),
    plot.subtitle = element_text(color = "#CC0000")
  ) +
  scale_color_brewer(palette = "Set2")

ggsave("data/processed/r_plots/plot2_npl_trends.png",
       p2, width = 10, height = 6, dpi = 150)

# Plot 3 — Capital Adequacy Bar Chart
p3 <- ggplot(df, aes(x = reorder(bank_name,
                                 capital_adequacy_ratio_pct),
                     y = capital_adequacy_ratio_pct,
                     fill = bank_name)) +
  geom_col(show.legend = FALSE) +
  geom_hline(yintercept = 10, linetype = "dashed",
             color = "red", size = 1) +
  annotate("text", x = 1.5, y = 11,
           label = "Minimum CAR 10%",
           color = "red", size = 3.5) +
  coord_flip() +
  facet_wrap(~year) +
  labs(
    title = "Capital Adequacy Ratio by Bank and Year",
    subtitle = "Red line = RBM minimum requirement (10%)",
    x = "Bank",
    y = "CAR (%)"
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", color = "#1B4F8A")) +
  scale_fill_brewer(palette = "Set2")

ggsave("data/processed/r_plots/plot3_capital_adequacy.png",
       p3, width = 14, height = 10, dpi = 150)

# Plot 4 — Correlation Heatmap
cat("      Generating correlation heatmap...\n")

cor_vars <- df %>%
  select(return_on_assets_pct, return_on_equity_pct,
         npl_ratio_pct, capital_adequacy_ratio_pct,
         cost_to_income_ratio_pct, liquidity_ratio_pct,
         loan_to_deposit_ratio_pct)

cor_matrix <- cor(cor_vars, use = "complete.obs")

png("data/processed/r_plots/plot4_correlation_heatmap.png",
    width = 900, height = 800, res = 120)
corrplot(cor_matrix,
         method  = "color",
         type    = "upper",
         tl.col  = "#1B4F8A",
         tl.srt  = 45,
         addCoef.col = "black",
         number.cex  = 0.8,
         col    = colorRampPalette(c("#D73027","white","#1B4F8A"))(200),
         title  = "CAMELS Correlation Matrix — Malawi Banking Sector",
         mar    = c(0,0,2,0))
dev.off()

# Plot 5 — Risk Flag Summary
p5 <- ggplot(df, aes(x = year,
                     fill = risk_flag)) +
  geom_bar(position = "fill") +
  labs(
    title    = "Risk Flag Distribution Over Time",
    subtitle = "Proportion of bank-years by risk category",
    x        = "Year",
    y        = "Proportion",
    fill     = "Risk Category"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", color = "#1B4F8A")) +
  scale_fill_manual(values = c(
    "HIGH RISK" = "#D73027",
    "WATCH"     = "#FC8D59",
    "HEALTHY"   = "#1A9850"
  )) +
  scale_y_continuous(labels = percent)

ggsave("data/processed/r_plots/plot5_risk_distribution.png",
       p5, width = 10, height = 6, dpi = 150)

# ── 8. Export Results ─────────────────────────────────────────
write_csv(bank_rankings,
          "data/processed/bank_rankings.csv")

cat("\n==========================================================\n")
cat("  R STATISTICAL ANALYSIS COMPLETE!\n")
cat("==========================================================\n")
cat("  Bank Rankings    : data/processed/bank_rankings.csv\n")
cat("  Plot 1 — ROA Trends           : r_plots/plot1\n")
cat("  Plot 2 — NPL Trends           : r_plots/plot2\n")
cat("  Plot 3 — Capital Adequacy     : r_plots/plot3\n")
cat("  Plot 4 — Correlation Heatmap  : r_plots/plot4\n")
cat("  Plot 5 — Risk Distribution    : r_plots/plot5\n")
cat("==========================================================\n")
