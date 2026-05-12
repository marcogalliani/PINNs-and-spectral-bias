# =============================================================================
# Spectral Bias Analysis – AdjointForwardSolver (WELL-SPECIFIED model)
#
# Counterpart to spectral_analysis_misspec.R:
#   - True forcing: superposition of sinusoids at frequencies [1, 3, 5, 10] Hz
#   - Base ODE model: TRUE model, i.e. includes the exact forcing
#   - Solver: AdjointForwardSolver estimates u(t) = residual correction
#   - Analysis: spectrum of the SOLUTION y(t), not of the residual u(t)
#   - Prediction: y(t) converges to the true solution uniformly across
#     frequencies; no spectral-bias ordering expected
#
# Contrast with the misspecified case where y(t) builds up low-frequency
# content first (spectral bias) because u(t) must reconstruct the entire
# forcing from scratch.  Here the base model already carries the signal, so
# the optimiser only adjusts a residual and the solution is correct from the
# start.
#
# Sections
#   1. Experiment parameters          (identical to misspec script)
#   2. Synthetic data generation
#   3. Build AdjointForwardSolver     (base ODE = TRUE model)
#   4. Snapshot-recording optimisation + reconstruct y(t) per snapshot
#   5. Spectral dynamics (FFT of y(t) snapshots)
#   6. Normalise by true solution amplitude
#   7. Plots: heatmap, final spectrum, state fit, residual u(t)
# =============================================================================

library(ggplot2)
library(reshape2)

setwd("ode-fit")
source("src/solvers/forward-solvers/load_forward_solvers.R")
source("examples/ode_models.R")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Experiment Parameters  (kept identical to misspec script)
# ─────────────────────────────────────────────────────────────────────────────

K_freqs <- c(1, 3, 5, 10)
A_amps  <- rep(1, length(K_freqs))

T_end  <- 2.0
t_sim  <- seq(0, T_end, length.out = 200)
dt_sim <- diff(t_sim)[1]
t_obs  <- t_sim[seq(1, length(t_sim), by = 4)]

alpha   <- 1.0
lambda  <- 1e0
seed    <- 42

REC_FRQ  <- 5L
max_iter <- 10000L

# ─────────────────────────────────────────────────────────────────────────────
# 2. Synthetic Data
# ─────────────────────────────────────────────────────────────────────────────

rhs_truth <- function(y, t, p) {
  -p$alpha * y + sum(p$A * sin(2 * pi * p$K * t))
}

y_gen    <- euler_solve(0, t_sim, rhs_truth, list(alpha = alpha, A = A_amps, K = K_freqs))
u_true_t <- sapply(t_sim, function(t) sum(A_amps * sin(2 * pi * K_freqs * t)))

obs_idx  <- match(round(t_obs, 10), round(t_sim, 10))
set.seed(seed)
noise_sd <- 0.05
obs_data <- y_gen[obs_idx, , drop = FALSE] #+ matrix(rnorm(length(obs_idx), 0, noise_sd), ncol = 1)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Build AdjointForwardSolver (base ODE = TRUE model, including forcing)
#
# u(t) is a residual correction; its optimum is u*(t) = 0.
# ─────────────────────────────────────────────────────────────────────────────

base_rhs <- function(y, t, p) {
  -p$alpha * y + sum(p$A * sin(2 * pi * p$K * t))
}

solver <- AdjointForwardSolver$new(
  model      = ODEModel$new(rhs = base_rhs),
  times_sim  = t_sim,
  obs_times  = t_obs,
  obs_values = obs_data,
  params     = list(alpha = alpha, A = A_amps, K = K_freqs),
  lambda     = lambda
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Snapshot-Recording Optimisation + Reconstruct y(t) Per Snapshot
#
# We record u(t) at every REC_FRQ cost calls, then reconstruct the state
# trajectory y(t) for each snapshot post-hoc.  Spectral analysis targets y(t).
# ─────────────────────────────────────────────────────────────────────────────

ns <- solver$n_steps
nv <- solver$n_vars
y0_val <- array(0, nv)

u0_flat    <- rep(0, ns * nv)
frames     <- list(list(call = 0L, u = u0_flat))   # initial snapshot at u = 0
call_count <- 0L

tracked_cost <- function(u_flat, y0) {
  val        <- solver$cost_function(u_flat, y0)
  call_count <<- call_count + 1L
  if (call_count %% REC_FRQ == 0L) {
    frames[[length(frames) + 1L]] <<- list(call = call_count, u = u_flat)
  }
  val
}

cat("Starting optimisation (well-specified model)...\n")
res <- optim(
  par     = rep(0, ns * nv),
  fn      = tracked_cost,
  gr      = solver$gradient_function,
  y0      = y0_val,
  method  = "BFGS",
  control = list(maxit = max_iter, reltol = sqrt(.Machine$double.eps))
)

solver$u <- matrix(res$par, ns, nv)
final    <- solver$solve_state_adjoint(solver$u, y0_val)
solver$y <- final$y
solver$p <- final$p

cat(sprintf("Done: %d snapshots | convergence code %d | final value %.4e\n",
            length(frames), res$convergence, res$value))

# Reconstruct y(t) for every recorded snapshot
y_frames <- lapply(frames, function(fr) {
  u_mat <- matrix(fr$u, ns, nv)
  solver$solve_state_adjoint(u_mat, y0_val)$y
})

# ─────────────────────────────────────────────────────────────────────────────
# 5. Spectral Dynamics  (FFT of y(t), not u(t))
# ─────────────────────────────────────────────────────────────────────────────

dt_val <- diff(t_sim)[1]

compute_fft <- function(signal, dt) {
  n    <- length(signal)
  half <- seq_len(floor(n / 2))
  frqs <- (seq_len(n) - 1L) / (n * dt)
  Yf   <- fft(signal) / n
  list(frq = frqs[half], amp = Mod(Yf[half]))
}

spectra <- lapply(y_frames, function(y_snap)
  compute_fft(y_snap[, 1], dt_val)
)

frq     <- spectra[[1]]$frq
n_snaps <- length(spectra)

dyn_mat <- matrix(0, nrow = n_snaps, ncol = length(frq))
for (i in seq_len(n_snaps)) dyn_mat[i, ] <- spectra[[i]]$amp

# ─────────────────────────────────────────────────────────────────────────────
# 6. Normalise by True Solution Amplitude
#
# Reference: FFT of y_gen at each forcing frequency.
# Both misspec and wellspec heatmaps then share a [0, 1] colour scale where
# 1 = true solution amplitude recovered.
# ─────────────────────────────────────────────────────────────────────────────

y_ref_spec  <- compute_fft(y_gen[, 1], dt_val)
sel_cols    <- sapply(K_freqs, function(kf) which.min(abs(frq - kf)))
y_ref_amps  <- 2 * y_ref_spec$amp[sel_cols]   # factor 2: one-sided spectrum

norm_mat <- sweep(2 * dyn_mat[, sel_cols, drop = FALSE], 2, y_ref_amps, "/")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Plots
# ─────────────────────────────────────────────────────────────────────────────

iter_labels <- sapply(frames, `[[`, "call")

# 7a. Spectral dynamics: relative amplitude of y(t) at each forcing frequency vs iteration
#     Expect all lines to start at ~ 1 and stay there — no spectral-bias ordering.
line_df <- reshape2::melt(norm_mat)
colnames(line_df) <- c("snapshot", "freq_idx", "norm_amp")
line_df$iter <- iter_labels[line_df$snapshot]
line_df$freq <- factor(K_freqs[line_df$freq_idx],
                       labels = paste0(K_freqs, " Hz"))

p_spectral <- ggplot(line_df, aes(x = iter, y = norm_amp, colour = freq)) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 1, linetype = "dashed", colour = "grey50", linewidth = 0.5) +
  scale_y_continuous(limits = c(0, NA)) +
  labs(title = "Spectral Dynamics of Solution  y(t)  [Well-Specified]",
       subtitle = "Relative amplitude at each forcing frequency vs. optimisation calls\n(dashed = target; expect uniform convergence at 1, no spectral-bias ordering)",
       x = "Cost-function calls", y = "Relative amplitude  |FFT| / |true|",
       colour = "Frequency") +
  theme_minimal(base_size = 12)

print(p_spectral)

# 7b. Final spectrum: estimated y(t) vs true y(t)
y_est_spec <- compute_fft(solver$y[, 1], dt_val)
y_ref_spec2 <- compute_fft(y_gen[, 1],   dt_val)

spec_df <- data.frame(
  freq   = rep(y_est_spec$frq, 2),
  amp    = c(y_est_spec$amp, y_ref_spec2$amp),
  source = rep(c("Estimated y(t)", "True y(t)"), each = length(y_est_spec$frq))
)
spec_df <- spec_df[spec_df$freq <= max(K_freqs) * 1.5, ]

p_spectrum <- ggplot(spec_df, aes(x = freq, y = amp, colour = source)) +
  geom_line(linewidth = 0.8) +
  geom_vline(xintercept = K_freqs, linetype = "dashed",
             colour = "grey60", linewidth = 0.4) +
  labs(title = "Final Spectrum: Estimated vs True Solution  [Well-Specified]",
       x = "Frequency [Hz]", y = "Amplitude", colour = NULL) +
  theme_minimal(base_size = 12) +
  scale_colour_manual(values = c("Estimated y(t)" = "#2166ac",
                                 "True y(t)"      = "#d6604d"))

print(p_spectrum)

# 7c. State trajectory: true vs fitted
fit_df <- data.frame(time  = t_sim,
                     y_fit = solver$y[, 1],
                     y_gen = y_gen[, 1])
obs_df <- data.frame(time = t_obs, obs = obs_data[, 1])

p_fit <- ggplot(fit_df, aes(x = time)) +
  geom_line(aes(y = y_gen, colour = "True y(t)")) +
  geom_line(aes(y = y_fit, colour = "Fitted y(t)")) +
  geom_point(data = obs_df, aes(x = time, y = obs),
             size = 0.8, colour = "grey40") +
  labs(title = "State Trajectory: True vs Fitted  [Well-Specified]",
       x = "Time", y = "y(t)", colour = NULL) +
  theme_minimal(base_size = 12) +
  scale_colour_manual(values = c("True y(t)"   = "#d6604d",
                                 "Fitted y(t)" = "#2166ac"))

print(p_fit)

# 7d. Residual correction u(t) in the time domain (sanity check: should be ≈ 0)
u_df <- data.frame(time  = t_sim,
                   u_hat = solver$u[, 1])

p_residual <- ggplot(u_df, aes(x = time, y = u_hat)) +
  geom_line(colour = "#2166ac", alpha = 0.85) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
  labs(title = "Residual Correction u(t)  [Well-Specified]",
       subtitle = "Should be ≈ 0 everywhere — the base model already carries the signal",
       x = "Time", y = "u(t)") +
  theme_minimal(base_size = 12)

print(p_residual)
