library(tidyverse)

##############################################################################
# Experiment 1 data
da_exp <- read.csv("./outputs/simuls_nips/synth_da/scores.csv", nrows = 4, header = F)
for (ii in c(2, 3, 4, 1)){
  da_exp[ii,] <- da_exp[ii,] / da_exp[1,]
}
da_exp <- da_exp[2:4,] %>%
  gather(key = "estimator", value = "score")

estimator <- c("log-diag", "Wasserstein", "geometric") %>%
  factor(, levels = c("log-diag", "Wasserstein", "geometric"))
da_exp$estimator <- rep(estimator, times = 10)

distance <- read.csv("./outputs/simuls_nips/synth_da/distance_a.csv", header = F)
da_exp$xaxis <- rep(distance[["V1"]], each = 3)

color_cats <- c(
  "#009D79",# blueish green
  "#EEA535",  # orange
  "#E36C2F"  #vermillon
  # "#56B4E9",# sky blue
  # "#F0E442", #yellow
  # "#0072B2", #blue
  # "#CC79A7" #violet
)

ggplot(
  data = da_exp,
  mapping = aes(y = score, x = xaxis, group = estimator,
                color = estimator, fill = estimator)) +
  geom_hline(yintercept = 1., color = "black", linetype = "dotted",
             size = 1) +
  geom_line(size = 1.5, alpha = 0.8) +
  geom_point(fill = "white", size = 3, shape = 21) +
    theme_minimal() +
  scale_x_continuous(breaks = seq(0, 3, 0.5)) +
  scale_y_continuous(limits = c(0, 1.05)) +
  scale_color_manual(values = color_cats, name = NULL) +
  scale_fill_manual(values = color_cats, name = NULL) +
  annotate(geom = "text", x = 0.35, y = 1.04, label = "chance level") +
  labs(x = expression(mu),
       y = "Normalized MAE") +
  theme(text = element_text(family = "Helvetica", size = 16),
        legend.position = "top", legend.title = element_text(size = 14))
ggsave("./outputs/fig_1a_distance.png", width = 5, height = 3.5, dpi = 300)
ggsave("./outputs/fig_1a_distance.pdf", width = 5, height = 3.5, dpi = 300)


##############################################################################
# Experiment 2 data

snr_exp <- read.csv("./outputs/simuls_nips/synth_snr/scores.csv", nrows = 4, header = F)
for (ii in c(2, 3, 4, 1)){
  snr_exp[ii,] <- snr_exp[ii,] / snr_exp[1,]
}

snr_exp <- snr_exp %>%
  gather(key = "estimator", value = "score")

estimator2 <- c("chance", "log-diag", "Wasserstein", "geometric") %>%
  factor(, levels = c("chance", "log-diag", "Wasserstein", "geometric"))

snr_exp$estimator <- rep(estimator2, times = 10)

sigmas <- read.csv("./outputs/simuls_nips/synth_snr/sigmas.csv", header = F)
snr_exp$xaxis <- rep(sigmas[["V1"]], each = 4)

ggplot(
  data = snr_exp %>% subset(estimator != "chance"),
  mapping = aes(y = score, x = xaxis, group = estimator,
                color = estimator)) +
  geom_line(size = 1.5, alpha = 0.8) +
  geom_point(fill = "white", size = 3, shape = 21) +
  theme_minimal() +
  scale_y_continuous(limits = c(0, 1.05)) +
  scale_x_log10(breaks =  10^(-10:10),
                minor_breaks = rep(1:9, 21) * (10 ^ rep(-10:10, each=9))) +
  scale_color_manual(values = color_cats, name = NULL) +
  # labs(x = TeX("distance between A and $I_p$"),
  #      y = "normalized M.A.E.") +
  geom_hline(yintercept = 1, color = "black", linetype = "dotted",
             size = 1) +
  annotate(geom = "text", x = 0.02, y = 1.04, label = "chance level") +
  labs(x = expression(sigma),
       y = "Normalized MAE") +
  theme(text = element_text(family = "Helvetica", size = 16),
        legend.position = "top", legend.title = element_text(size = 14))
ggsave("./outputs/fig_1b_snr.png", width = 5, height = 3.5, dpi = 300)
ggsave("./outputs/fig_1b_snr.pdf", width = 5, height = 3.5, dpi = 300)
