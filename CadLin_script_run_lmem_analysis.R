3) scripts/run_lmem_analysis.R
# Minimal reproducible script for the inferential stage
# Mixed-effects analysis for PB vs. SP dialect differentiation

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(lme4)
  library(lmerTest)
  library(performance)
})

# -----------------------------
# Input
# -----------------------------
input_file <- "data/lmem_dataset.csv"
output_file <- "outputs/lmem_results.csv"

# -----------------------------
# Read data
# -----------------------------
df <- read_csv(input_file, show_col_types = FALSE)

# Required columns
required_cols <- c(
  "sample_id", "speaker", "dialect",
  "f0sd", "f0SAQ", "df0mean_pos", "df0sd_pos",
  "sl_LTAS_alpha", "cvint",
  "pause_sd", "pause_meandur", "pause_rate"
)

missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
}

df <- df %>%
  mutate(
    speaker = as.factor(speaker),
    dialect = as.factor(dialect)
  )

acoustic_vars <- c(
  "f0sd", "f0SAQ", "df0mean_pos", "df0sd_pos",
  "sl_LTAS_alpha", "cvint",
  "pause_sd", "pause_meandur", "pause_rate"
)

results <- list()

for (var in acoustic_vars) {
  formula_str <- paste0(var, " ~ dialect + (1|speaker)")
  model <- lmer(as.formula(formula_str), data = df, REML = TRUE)
  
  coefs <- summary(model)$coefficients
  # assumes second row = dialect effect relative to reference level
  effect_row <- rownames(coefs)[grepl("^dialect", rownames(coefs))][1]
  
  r2_vals <- performance::r2_nakagawa(model)
  
  results[[var]] <- data.frame(
    acoustic_variable = var,
    effect = effect_row,
    beta = unname(coefs[effect_row, "Estimate"]),
    se = unname(coefs[effect_row, "Std. Error"]),
    t_value = unname(coefs[effect_row, "t value"]),
    p_value = unname(coefs[effect_row, "Pr(>|t|)"]),
    r2_marginal = unname(r2_vals$R2_marginal),
    r2_conditional = unname(r2_vals$R2_conditional)
  )
}

results_df <- bind_rows(results)

dir.create("outputs", showWarnings = FALSE, recursive = TRUE)
write_csv(results_df, output_file)

cat("LMEM analysis completed.\n")
cat("Results saved to:", output_file, "\n")