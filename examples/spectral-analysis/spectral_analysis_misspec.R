# =============================================================================
# Spectral Bias Analysis – AdjointForwardSolver
#
# R analogue of ICML_SpectralBias.ipynb applied to AdjointForwardSolver:
#   - True forcing: superposition of sinusoids at known frequencies K
#   - Base ODE model: simple exponential decay (no forcing)
#   - Solver: AdjointForwardSolver estimates u(t) to close model-data gap
#   - Question: does the estimated forcing recover low frequencies first?
#
# Sections
#   1. Experiment parameters
#   2. Synthetic data generation (base ODE + known multi-freq forcing)
#   3. Build AdjointForwardSolver
#   4. Snapshot-recording optimisation (record u(t) every REC_FRQ calls)
#   5. Spectral dynamics (FFT of u(t) snapshots)
#   6. Normalise by true forcing amplitude
#   7. Plots: heatmap, final spectrum, time-domain fit / forcing
# =============================================================================

library(ggplot2)
library(reshape2)

setwd("ode-fit")
source("src/solvers/forward-solvers/load_forward_solvers.R")
source("examples/ode_models.R")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Experiment Parameters
# ─────────────────────────────────────────────────────────────────────────────

# True-forcing frequencies (Hz) and amplitudes – equal-amplitude case
K_freqs <- c(2, 4, 6, 8, 10, 12, 14, 16)
A_amps  <- rep(1, length(K_freqs))

T_end  <- 2.0
dt_sim <- 1e-2               # 201 pts, fs = 100 Hz, Nyquist = 50 Hz
t_sim  <- seq(0, T_end, by = dt_sim)
t_obs  <- seq(0, T_end, by = 4*dt_sim)   # coarser observation grid

alpha   <- 1.0      # decay rate of the base ODE
lambda  <- 1e-7
seed    <- 42

REC_FRQ  <- 5L      # record a snapshot every N cost-function calls
max_iter <- 1000L

# ─────────────────────────────────────────────────────────────────────────────
# 2. Synthetic Data
# ─────────────────────────────────────────────────────────────────────────────

# Data-generating RHS: dy/dt = -alpha*y + u_true(t)
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
# 3. Build AdjointForwardSolver (base ODE without forcing)
# ─────────────────────────────────────────────────────────────────────────────

base_rhs <- function(y, t, p) -p$alpha * y

solver <- AdjointForwardSolver$new(
  model      = ODEModel$new(rhs = base_rhs),
  times_sim  = t_sim,
  obs_times  = t_obs,
  obs_values = obs_data,
  params     = list(alpha = alpha),
  lambda     = lambda
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Snapshot-Recording Optimisation
# ─────────────────────────────────────────────────────────────────────────────

ns <- solver$n_steps
nv <- solver$n_vars
y0_val <- array(0, nv)

frames     <- list()
call_count <- 0L

# Wraps cost_function: records u(t) every REC_FRQ calls.
# Signature matches optim() expectation: fn(par, y0).
tracked_cost <- function(u_flat, y0) {
  val        <- solver$cost_function(u_flat, y0)
  call_count <<- call_count + 1L
  if (call_count %% REC_FRQ == 0L) {
    frames[[length(frames) + 1L]] <<- list(call = call_count, u = u_flat)
  }
  val
}

cat("Starting optimisation...\n")
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

# ─────────────────────────────────────────────────────────────────────────────
# 5. Spectral Dynamics
# ─────────────────────────────────────────────────────────────────────────────

dt_val <- diff(t_sim)[1]

compute_fft <- function(signal, dt) {
  n    <- length(signal)
  half <- seq_len(floor(n / 2))
  frqs <- (seq_len(n) - 1L) / (n * dt)
  Yf   <- fft(signal) / n
  list(frq = frqs[half], amp = Mod(Yf[half]))
}

spectra <- lapply(frames, function(fr)
  compute_fft(fr$u[seq_len(ns)], dt_val)   # var 1 (only 1 var)
)

frq     <- spectra[[1]]$frq
n_snaps <- length(spectra)

dyn_mat <- matrix(0, nrow = n_snaps, ncol = length(frq))
for (i in seq_len(n_snaps)) dyn_mat[i, ] <- spectra[[i]]$amp

# ─────────────────────────────────────────────────────────────────────────────
# 6. Normalise by True Forcing Amplitude
# ─────────────────────────────────────────────────────────────────────────────

sel_cols <- sapply(K_freqs, function(kf) which.min(abs(frq - kf)))

# Factor 2: one-sided spectrum has half the power of the two-sided one
norm_mat <- sweep(2 * dyn_mat[, sel_cols], 2, A_amps, "/")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Plots
# ─────────────────────────────────────────────────────────────────────────────

iter_labels <- sapply(frames, `[[`, "call")

# 7a. Spectral dynamics heatmap (freq x iter)
hm_df <- reshape2::melt(norm_mat)
colnames(hm_df) <- c("snapshot", "freq_idx", "norm_amp")
hm_df$iter <- iter_labels[hm_df$snapshot]
hm_df$freq <- K_freqs[hm_df$freq_idx]

p_heatmap <- ggplot(hm_df, aes(x = factor(freq), y = iter,
                                fill = pmin(norm_amp, 1))) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "#2166ac", limits = c(0, 1),
                      name = "Relative\namplitude") +
  scale_y_reverse() +
  labs(title = "Spectral Dynamics of Estimated Forcing  u(t)",
       subtitle = "Rows = optimisation cost calls (early at top); cols = forcing frequency",
       x = "Frequency [Hz]", y = "Cost-function calls") +
  theme_minimal(base_size = 12)

print(p_heatmap)

# 7b. Final spectrum: estimated vs true forcing
u_est_spec  <- compute_fft(solver$u[, 1], dt_val)
u_true_spec <- compute_fft(u_true_t,      dt_val)

spec_df <- data.frame(
  freq   = rep(u_est_spec$frq, 2),
  amp    = c(u_est_spec$amp, u_true_spec$amp),
  source = rep(c("Estimated u(t)", "True u(t)"), each = length(u_est_spec$frq))
)
spec_df <- spec_df[spec_df$freq <= max(K_freqs) * 1.5, ]

p_spectrum <- ggplot(spec_df, aes(x = freq, y = amp, colour = source)) +
  geom_line(linewidth = 0.8) +
  geom_vline(xintercept = K_freqs, linetype = "dashed",
             colour = "grey60", linewidth = 0.4) +
  labs(title = "Final Spectrum: Estimated vs True Forcing",
       x = "Frequency [Hz]", y = "Amplitude", colour = NULL) +
  theme_minimal(base_size = 12) +
  scale_colour_manual(values = c("Estimated u(t)" = "#2166ac",
                                 "True u(t)"      = "#d6604d"))

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
  labs(title = "State Trajectory: True vs Fitted",
       x = "Time", y = "y(t)", colour = NULL) +
  theme_minimal(base_size = 12) +
  scale_colour_manual(values = c("True y(t)"   = "#d6604d",
                                 "Fitted y(t)" = "#2166ac"))

print(p_fit)

# 7d. True vs estimated forcing (time domain)
u_df <- data.frame(time  = t_sim,
                   u_hat = solver$u[, 1],
                   u_tru = u_true_t)

p_forcing <- ggplot(u_df, aes(x = time)) +
  geom_line(aes(y = u_tru, colour = "True u(t)")) +
  geom_line(aes(y = u_hat, colour = "Estimated u(t)"), alpha = 0.85) +
  labs(title = "Forcing: True vs Estimated",
       x = "Time", y = "u(t)", colour = NULL) +
  theme_minimal(base_size = 12) +
  scale_colour_manual(values = c("True u(t)"      = "#d6604d",
                                 "Estimated u(t)" = "#2166ac"))

print(p_forcing)