#!/usr/bin/env Rscript
#################################################
# Title: Paired tumor - normal per feature      #
# Date: November 10, 2026                       #
# Notes: Step 4. For one assay                  #
#   (protein or RNA), takes the primary log2    #
#   matrix (features x samples) and, for every  #
#   patient with a tumor/normal pair,           #
#   computes                                    #
#     delta     = log2 T - log2 N, averaged     #
#                 over the patient's pairs that #
#                 are non-missing for the       #
#                 feature (n_pairs_used says    #
#                 how many)                     #
#     delta_pc  = delta minus the patient's     #
#                 median delta over the         #
#                 centering set                 #
#     delta_dc  = delta minus the feature's     #
#                 mean delta over patients,     #
#                 then minus the patient's      #
#                 median of that over the       #
#                 centering set (double-        #
#                 centered)                     #
#   The centering set is the expressed features #
#   without a strong cohort-wide T-N change     #
#   (column centring_set, kept under this       #
#   name as published).                         #
#   Which version to use is left to the user:   #
#   delta_dc removes each feature's average     #
#   tumor effect, which includes any            #
#   systematic tumor-specific effect shared by  #
#   the cohort.                                 #
#   Sample sheet (TSV): patient, prot_tumour,   #
#   prot_normal (sample = matrix column name;   #
#   ";" between replicate pairs, same number of #
#   names in both lists), optional has_wgs      #
#   (TRUE/FALSE, only colors the QC plot). For  #
#   RNA: rna_tumour, rna_normal.                #
#   Outputs (under --outdir):                   #
#     <assay>_tn.tsv.gz  patient x feature T-N  #
#     <assay>_feature_stats.tsv                 #
#     <assay>_patients.tsv (QC per patient)     #
#     <assay>_excluded.tsv (skipped pairs and   #
#       patients, with the reason)              #
#     plots/<assay>_tn_by_patient.pdf           #
#   The T-N table and the feature statistics    #
#   are saved before the QC and the plots.      #
#################################################

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "utils.R"))
suppressPackageStartupMessages(library(cowplot))

##############
# Parameters #
##############

opt <- parse_args_required(list(
  make_option("--assay", type = "character", help = "protein or rna"),
  make_option("--matrix", type = "character", help = "features x samples matrix (CSV/TSV, first column = feature id)"),
  make_option("--samplesheet", type = "character", help = "sample sheet TSV (patient, prot_tumour, prot_normal, optional has_wgs)"),
  make_option("--min_ave", type = "double", default = NA_real_,
              help = "minimum mean log2 level for a feature to count as expressed [no filter]"),
  make_option("--strong_q", type = "double", default = 0.01, help = "BH q for a strong cohort-wide T-N change"),
  make_option("--strong_lfc", type = "double", default = 1, help = "|mean T-N| for a strong change"),
  make_option("--clamp", type = "double", default = NA_real_, help = "clamp centered values to +/- this [none]"),
  make_option("--outdir", type = "character", default = ".", help = "output directory")
), required = c("assay", "matrix", "samplesheet"))

assay_name <- match.arg(opt$assay, c("protein", "rna"))
sample_prefix <- c(protein = "prot", rna = "rna")[[assay_name]]
plots_dir <- file.path(opt$outdir, "plots")
dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)

# Constants of the cohort-signature resemblance QC (see "Patient summary").
signature_q <- 0.01          # BH q that defines a cohort-wide change
signature_min_features <- 20 # with fewer such features, all expressed features are used
min_complete_obs <- 3        # fewest complete feature pairs needed for a correlation

#############
# Functions #
#############

# Read the features x samples matrix. The first column (feature ids) is read as
# text so that ids such as "00123" or "10000-28" keep their exact spelling; all
# other columns must be numbers. Stops on parsing problems, missing or
# duplicated ids, duplicated sample names and infinite values.
read_expression_matrix <- function(file) {
  delim <- if (grepl("\\.csv(\\.gz)?$", file)) "," else "\t"
  header <- names(read_delim(file, delim = delim, n_max = 0, col_types = cols(.default = "c"),
                             name_repair = "minimal", show_col_types = FALSE, progress = FALSE))
  if (length(header) < 2) stop("matrix needs a feature id column and at least one sample column")
  if (any(is.na(header) | header == "")) stop("matrix has an empty column name")
  if (anyDuplicated(header)) {
    stop("duplicated column names in the matrix: ", paste(unique(header[duplicated(header)]), collapse = ", "))
  }
  matrix.df <- read_delim(file, delim = delim, col_types = paste0("c", strrep("d", length(header) - 1)),
                          name_repair = "minimal", show_col_types = FALSE, progress = FALSE)
  stop_on_parse_problems(matrix.df, "matrix")
  feature_ids <- matrix.df[[1]]
  if (any(is.na(feature_ids) | feature_ids == "")) stop("the matrix has rows without a feature id")
  if (anyDuplicated(feature_ids)) {
    stop("duplicated feature ids in the matrix: ",
         paste(head(unique(feature_ids[duplicated(feature_ids)]), 5), collapse = ", "))
  }
  values <- as.matrix(matrix.df[, -1])
  if (any(is.infinite(values))) stop("the matrix contains infinite values")
  rownames(values) <- feature_ids
  values
}

# Read the sample sheet and check the patient keys: every patient needs a name,
# and each patient may appear in one row only.
read_samplesheet <- function(file, tumor_col, normal_col) {
  samples.df <- read_tsv(file, col_types = cols(.default = "c"), show_col_types = FALSE)
  missing <- setdiff(c("patient", tumor_col, normal_col), names(samples.df))
  if (length(missing) > 0) stop("sample sheet lacks column(s): ", paste(missing, collapse = ", "))
  if (!"has_wgs" %in% names(samples.df)) samples.df$has_wgs <- "FALSE"   # only used to color the QC plot
  if (any(is.na(samples.df$patient))) stop("sample sheet has rows without a patient name")
  if (anyDuplicated(samples.df$patient)) {
    stop("patient listed in more than one row of the sample sheet: ",
         paste(unique(samples.df$patient[duplicated(samples.df$patient)]), collapse = ", "),
         ". Put all replicate pairs of a patient in one row, separated by ';'.")
  }
  samples.df %>%
    transmute(patient, has_wgs = toupper(has_wgs) %in% "TRUE",
              tumor_names = .data[[tumor_col]], normal_names = .data[[normal_col]])
}

# Turn the sample sheet into one row per tumor/normal pair. The ";"-separated
# tumor and normal lists of a patient pair up by position, so they must have
# the same length. Returns the pairs and a table of patients without any pair.
build_pairs <- function(samples.df) {
  has_names.df <- samples.df %>% mutate(has_both = !is.na(tumor_names) & !is.na(normal_names))
  no_names.df <- has_names.df %>%
    filter(!has_both) %>%
    transmute(patient, pair = NA_integer_, reason = "tumor or normal sample name missing in the sample sheet")

  split.df <- has_names.df %>%
    filter(has_both) %>%
    mutate(tumor_list = str_split(tumor_names, ";"), normal_list = str_split(normal_names, ";"))
  unequal <- lengths(split.df$tumor_list) != lengths(split.df$normal_list)
  if (any(unequal)) {
    stop("tumor and normal lists have different lengths for patient(s): ",
         paste(split.df$patient[unequal], collapse = ", "),
         ". Replicates pair up by position, so both lists need the same number of names.")
  }

  pairs.df <- split.df %>%
    select(patient, has_wgs, tumor_list, normal_list) %>%
    unnest(c(tumor_list, normal_list)) %>%
    rename(tumor_sample = tumor_list, normal_sample = normal_list) %>%
    mutate(tumor_sample = str_trim(tumor_sample), normal_sample = str_trim(normal_sample)) %>%
    group_by(patient) %>%
    mutate(pair = row_number()) %>%
    ungroup()

  empty_name <- pairs.df$tumor_sample == "" | pairs.df$normal_sample == ""
  if (any(empty_name)) {
    stop("empty sample name (stray ';'?) for patient(s): ", paste(unique(pairs.df$patient[empty_name]), collapse = ", "))
  }
  duplicated_pair <- duplicated(pairs.df[c("tumor_sample", "normal_sample")])
  if (any(duplicated_pair)) {
    stop("the same tumor/normal pair is listed more than once (patient(s): ",
         paste(unique(pairs.df$patient[duplicated_pair]), collapse = ", "), ")")
  }
  both_roles <- intersect(pairs.df$tumor_sample, pairs.df$normal_sample)
  if (length(both_roles) > 0) {
    in_both <- pairs.df$tumor_sample %in% both_roles | pairs.df$normal_sample %in% both_roles
    stop("a sample name is used as both tumor and normal (patient(s): ",
         paste(unique(pairs.df$patient[in_both]), collapse = ", "), ")")
  }
  list(pairs.df = pairs.df, no_names.df = no_names.df)
}

# Per patient, the mean over the pairs that are non-missing for each feature.
# Returns the mean T-N matrix (features x patients; NA if no pair is usable)
# and the matching number of pairs used.
mean_over_pairs <- function(pair_delta, patient_of_pair, patient_order) {
  delta <- matrix(NA_real_, nrow = nrow(pair_delta), ncol = length(patient_order),
                  dimnames = list(rownames(pair_delta), patient_order))
  n_used <- matrix(0L, nrow = nrow(pair_delta), ncol = length(patient_order),
                   dimnames = dimnames(delta))
  for (patient_key in patient_order) {
    patient_pairs <- pair_delta[, patient_of_pair == patient_key, drop = FALSE]
    n_used[, patient_key] <- as.integer(rowSums(!is.na(patient_pairs)))
    delta[, patient_key] <- if_else(n_used[, patient_key] > 0,
                                    rowSums(patient_pairs, na.rm = TRUE) / pmax(n_used[, patient_key], 1L),
                                    NA_real_)
  }
  list(delta = delta, n_used = n_used)
}

# Row means that ignore NA and give NA (not NaN) when a row has no values.
row_mean <- function(m) {
  out <- rowMeans(m, na.rm = TRUE)
  out[is.nan(out)] <- NA_real_
  out
}

# A statistic of every column of m over the rows chosen by the logical vector
# `rows`. NA for every column when no row is chosen (an empty set).
column_stat <- function(m, rows, stat_fn) {
  if (sum(rows) == 0) return(rep(NA_real_, ncol(m)))
  apply(m[rows, , drop = FALSE], 2, stat_fn, na.rm = TRUE)
}

# Correlation of two vectors over their complete pairs; NA when there are too
# few pairs or one vector does not vary (instead of an error or a warning).
safe_cor <- function(x, y) {
  complete <- !is.na(x) & !is.na(y)
  if (sum(complete) < min_complete_obs) return(NA_real_)
  if (sd(x[complete]) == 0 || sd(y[complete]) == 0) return(NA_real_)
  cor(x[complete], y[complete])
}

# Matrix (features x patients) to a long table with one value column.
to_long <- function(m, value_name) {
  as_tibble(m, rownames = "feature_id") %>%
    pivot_longer(-feature_id, names_to = "patient", values_to = value_name)
}

#############
# Read data #
#############

expr <- read_expression_matrix(opt$matrix)
samples.df <- read_samplesheet(opt$samplesheet, paste0(sample_prefix, "_tumour"), paste0(sample_prefix, "_normal"))

###########
# Pairs   #
###########

built <- build_pairs(samples.df)
pairs.df <- built$pairs.df %>%
  mutate(in_matrix = tumor_sample %in% colnames(expr) & normal_sample %in% colnames(expr))

# Pairs with a sample that is not a matrix column are skipped and listed. A
# patient with no remaining pair is reported as excluded. Sample names stay
# internal: only patient keys are printed.
skipped_pairs.df <- pairs.df %>%
  filter(!in_matrix) %>%
  transmute(patient, pair, reason = "sample not found in the matrix; pair skipped")
pairs.df <- pairs.df %>% filter(in_matrix)
no_pair.df <- samples.df %>%
  filter(!patient %in% pairs.df$patient, !patient %in% built$no_names.df$patient) %>%
  transmute(patient, pair = NA_integer_, reason = "patient excluded: no pair with both samples in the matrix")
excluded.df <- bind_rows(built$no_names.df, skipped_pairs.df, no_pair.df)
if (nrow(excluded.df) > 0) {
  message("pairs or patients skipped (see ", assay_name, "_excluded.tsv): ",
          paste(unique(excluded.df$patient), collapse = ", "))
}
if (nrow(pairs.df) == 0) stop("no tumor/normal pair found in the matrix")
write_tsv(excluded.df, file.path(opt$outdir, paste0(assay_name, "_excluded.tsv")), na = "")

####################
# Tumor - normal   #
####################

# T-N for every pair and feature, then the mean over the patient's pairs that
# are non-missing for that feature. A feature missing in one replicate pair is
# averaged over the other pair(s) (n_pairs_used records how many).
pair_delta <- expr[, pairs.df$tumor_sample, drop = FALSE] - expr[, pairs.df$normal_sample, drop = FALSE]
patients <- unique(pairs.df$patient)
averaged <- mean_over_pairs(pair_delta, pairs.df$patient, patients)
delta <- averaged$delta
n_pairs_used <- averaged$n_used
n_pat <- length(patients)

#################
# Feature stats #
#################

# One-sample t-test of each feature's T-N over patients, BH-adjusted. The test
# is used only to find features with a strong cohort-wide change, not for
# inference.
used_samples <- unique(c(pairs.df$tumor_sample, pairs.df$normal_sample))
feature_stats.df <- tibble(
  feature_id = rownames(delta),
  n_patients = rowSums(!is.na(delta)),
  ave_expr = row_mean(expr[, used_samples, drop = FALSE]),
  mean_delta = row_mean(delta),
  sd_delta = apply(delta, 1, sd, na.rm = TRUE)
) %>%
  mutate(t = mean_delta / (sd_delta / sqrt(n_patients)),
         p = if_else(n_patients > 1, 2 * pt(-abs(t), df = n_patients - 1), NA_real_),
         across(c(t, p), ~ if_else(is.nan(.x), NA_real_, .x)),
         q = p.adjust(p, method = "BH"),
         expressed = coalesce(is.na(opt$min_ave) | ave_expr >= opt$min_ave, FALSE),
         strong_tn = !is.na(q) & q < opt$strong_q & abs(mean_delta) > opt$strong_lfc,
         centring_set = expressed & !strong_tn)

#############
# Centering #
#############

# Patient medians over the centering set (features expressed and without a
# strong cohort-wide change). An empty set gives NA centered values and a warning.
centering_set <- feature_stats.df$centring_set
if (sum(centering_set) == 0) {
  warning("the centering set is empty: delta_pc and delta_dc are NA. ",
          "Relax --strong_q / --strong_lfc or --min_ave.", call. = FALSE)
}
clamp <- function(m) if (is.na(opt$clamp)) m else pmin(pmax(m, -opt$clamp), opt$clamp)

delta_pc <- clamp(sweep(delta, 2, column_stat(delta, centering_set, median)))
delta_minus_feature_mean <- delta - row_mean(delta)
delta_dc <- clamp(sweep(delta_minus_feature_mean, 2, column_stat(delta_minus_feature_mean, centering_set, median)))

tn.df <- to_long(delta, "delta") %>%
  left_join(to_long(delta_pc, "delta_pc"), by = c("feature_id", "patient")) %>%
  left_join(to_long(delta_dc, "delta_dc"), by = c("feature_id", "patient")) %>%
  left_join(to_long(n_pairs_used, "n_pairs_used"), by = c("feature_id", "patient")) %>%
  left_join(pairs.df %>% count(patient, name = "n_pairs"), by = "patient") %>%
  mutate(assay = assay_name) %>%
  select(assay, patient, feature_id, delta, delta_pc, delta_dc, n_pairs, n_pairs_used)

###############
# Save output #
###############

# The T-N table and the feature statistics come first: nothing below (QC,
# plots) can prevent them from being written.
write_tsv(tn.df, file.path(opt$outdir, paste0(assay_name, "_tn.tsv.gz")), na = "")

####################
# Patient summary  #
####################

# Cohort-signature resemblance (column tumour_signature_r, name kept as
# published): for each patient, the Pearson correlation between the patient's
# T-N and the cohort mean T-N over the "signature" features, those with a
# clear cohort-wide change (q < 0.01). With fewer than 20 such features, all
# expressed features are used. Read it with two limits in mind:
#   - it is scale-invariant: a patient whose tumor signal is weak but has the
#     same pattern still scores close to 1, so it is no measure of tumor content;
#   - it is self-inclusive: the patient is part of the cohort mean, which
#     matters little at n > 10 but inflates r in small cohorts.
# It is NA when the signature set is empty or a patient has fewer than 3
# complete observations.
signature_set <- feature_stats.df$expressed & !is.na(feature_stats.df$q) & feature_stats.df$q < signature_q
if (sum(signature_set) < signature_min_features) signature_set <- feature_stats.df$expressed
if (sum(signature_set) == 0) {
  warning("the signature feature set is empty: tumour_signature_r is NA", call. = FALSE)
}
feature_stats.df$tumour_signature <- signature_set

write_tsv(feature_stats.df, file.path(opt$outdir, paste0(assay_name, "_feature_stats.tsv")), na = "")

patients.df <- tibble(patient = patients) %>%
  left_join(pairs.df %>% group_by(patient) %>% summarise(n_pairs = n(), has_wgs = first(has_wgs)),
            by = "patient") %>%
  mutate(
    median_delta = column_stat(delta, centering_set, median),
    sd_delta_pc = column_stat(delta_pc, centering_set, sd),
    tumour_signature_r = apply(delta[signature_set, , drop = FALSE], 2, function(x)
      safe_cor(x, feature_stats.df$mean_delta[signature_set])),
    n_signature_features = sum(signature_set)
  ) %>%
  arrange(patient)
write_tsv(patients.df, file.path(opt$outdir, paste0(assay_name, "_patients.tsv")), na = "")

#########
# Plots #
#########

# One QC figure: patient-centered T-N boxplots over the centering set and the
# cohort-signature resemblance per patient. A plotting failure only warns.
make_qc_plot <- function() {
  order_by_cor <- patients.df %>% arrange(tumour_signature_r) %>% pull(patient)
  box.plot <- tn.df %>%
    semi_join(feature_stats.df %>% filter(centring_set), by = "feature_id") %>%
    left_join(patients.df %>% select(patient, has_wgs), by = "patient") %>%
    mutate(patient = factor(patient, levels = order_by_cor),
           wgs = if_else(has_wgs, "with WGS", "no WGS")) %>%
    ggplot(aes(x = patient, y = delta_pc, fill = wgs)) +
    geom_boxplot(outlier.shape = NA, linewidth = 0.2) +
    geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.3) +
    scale_fill_manual(values = c("with WGS" = col_focus, "no WGS" = col_other), name = "") +
    coord_cartesian(ylim = c(-2, 2)) +
    theme_simple() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
    labs(x = "", y = "T - N, patient-centered (log2)")
  cor.plot <- patients.df %>%
    mutate(patient = factor(patient, levels = order_by_cor),
           wgs = if_else(has_wgs, "with WGS", "no WGS")) %>%
    ggplot(aes(x = patient, y = tumour_signature_r, fill = wgs)) +
    geom_col(col = "black", linewidth = 0.2) +
    scale_fill_manual(values = c("with WGS" = col_focus, "no WGS" = col_other), name = "") +
    theme_simple() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1), legend.position = "none") +
    labs(x = "", y = "Cohort-signature resemblance (r)")
  plot_grid(
    ggdraw() + draw_label(paste0(assay_name, ": tumor - normal per patient (", n_pat, " patients, ",
                                 sum(centering_set), " features in the centering set)"),
                          size = 8, x = 0.02, hjust = 0),
    box.plot, cor.plot, ncol = 1, rel_heights = c(0.1, 1, 0.8))
}
save_plot_safely(make_qc_plot, file.path(plots_dir, paste0(assay_name, "_tn_by_patient.pdf")),
                 width = 180, height = 110)

message(assay_name, ": ", n_pat, " patients (", nrow(pairs.df), " pairs), ", nrow(delta), " features; ",
        sum(!feature_stats.df$expressed), " not expressed, ", sum(feature_stats.df$strong_tn),
        " strong cohort-wide T-N changes")
