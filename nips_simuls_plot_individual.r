library(tidyverse)

##############################################################################
# Experiment 1 data

noise_exp <- read.csv("./outputs/simuls_nips/individual_spatial/scores.csv",
                      nrows = 6, header = F)
for  (ii in c(2, 3, 4, 5, 6, 1)){
  noise_exp[ii,] <- noise_exp[ii,] / noise_exp[1,]
}
noise_exp <- noise_exp[2:6,] %>%
  gather(key = "estimator", value = "score")

# Identity + logDiag
# Supervised + Logdiag
# Identity + Wasserstein
# Unsupervised + Geom
# Supervised + Geom

estimator_levels <- c("log-diag", "sup. log-diag", "Wasserstein", "geometric",
                      "sup. geometric")
estimator <- estimator_levels %>% factor(levels = estimator_levels)
noise_exp$estimator <- rep(estimator, times = 10)

noises <- read.csv("./outputs/simuls_nips/individual_spatial/noises_A.csv", header = F)
noise_exp$xaxis <- rep(noises[["V1"]], each = 5)
is_supervised <- noise_exp$estimator %>% grepl("sup", .)
noise_exp$supervised <- ifelse(is_supervised, "sup", "unsup") %>%
    as.factor()

color_cats <- c(
  "#009D79",# blueish green
  "#009D79",# blueish green
  "#E36C2F",  #vermillon
  "#EEA535", # orange
  "#EEA535"  # orange
  # "#56B4E9",# sky blue
  # "#F0E442", #yellow
  # "#0072B2", #blue
  # "#CC79A7" #violet
)

ggplot(
  data = noise_exp,
  mapping = aes(y = score, x = xaxis, group = estimator,
                color = estimator, fill = estimator, linetype=supervised)) +
  geom_hline(yintercept = 1., color = "black", linetype = "dotted",
             size = 1) +
  geom_line(size = 1.5, alpha = 0.8) +
  geom_point(fill = "white", size = 3, shape = 21) +
  theme_minimal() +
  scale_x_log10() +
  scale_y_continuous(limits = c(0, 1.05)) +
  scale_color_manual(values = color_cats, name = NULL) +
  scale_fill_manual(values = color_cats, name = NULL) +
  scale_linetype_manual(values=c("dotdash", "solid"), name = NULL) +
  annotate(geom = "text", x = 0.0045, y = 1.04, label = "chance level") +
  labs(x = expression(sigma),
       y = "Normalized MAE") +
  guides(
      color = guide_legend(
          nrow = 2,
          override.aes = list(linetype = c("solid", "dotdash", "solid", "solid", "dotdash")))) +
  guides(linetype = F) +
  theme(text = element_text(family = "Helvetica", size = 16),
        legend.position = "top",
        legend.margin = margin(t = 0, r = 0, b = -.4, l = -1, unit="cm"),
        legend.title = element_text(size = 14))

ggsave("./outputs/fig_1c_individual_noise.png", width = 5, height = 3.5, dpi = 300)
ggsave("./outputs/fig_1c_individual_noise.pdf", width = 5, height = 3.5, dpi = 300)
