# Tests for bin/paired_tn.R

# A small matrix with three special feature ids and four patients. Patient P1
# has two replicate pairs (P1_T1/P1_N1 and P1_T2/P1_N2).
make_matrix <- function() {
  ids <- c("10000-28", "00123", "12345678901234567890", "F4", "F5", "F6")
  samples <- c("P1_T1", "P1_N1", "P1_T2", "P1_N2", "P2_T", "P2_N", "P3_T", "P3_N", "P4_T", "P4_N")
  set.seed(1)
  values <- matrix(round(rnorm(length(ids) * length(samples), 10, 1), 3),
                   nrow = length(ids), dimnames = list(ids, samples))
  as_tibble(values, rownames = "feature_id")
}

make_sheet <- function() {
  tibble(patient = c("P1", "P2", "P3", "P4"),
         prot_tumour = c("P1_T1;P1_T2", "P2_T", "P3_T", "P4_T"),
         prot_normal = c("P1_N1;P1_N2", "P2_N", "P3_N", "P4_N"))
}

run_paired <- function(matrix.df, sheet.df, ...) {
  outdir <- tempfile("paired_")
  result <- run_script("paired_tn.R", c(
    "--assay", "protein",
    "--matrix", write_input(matrix.df, basename(tempfile("matrix_", fileext = ".tsv"))),
    "--samplesheet", write_input(sheet.df, basename(tempfile("sheet_", fileext = ".tsv"))),
    "--outdir", outdir, ...))
  list(result = result, outdir = outdir, tn = file.path(outdir, "protein_tn.tsv.gz"))
}

test_that("feature ids such as 10000-28, 00123 and a 20-digit number are not altered", {
  run <- run_paired(make_matrix(), make_sheet())
  expect_script_ok(run$result)
  tn.df <- read_text_tsv(run$tn)
  expect_setequal(unique(tn.df$feature_id), make_matrix()$feature_id)
  expect_true(all(c("10000-28", "00123", "12345678901234567890") %in% tn.df$feature_id))
})

test_that("unequal replicate lists stop the run and name the patient", {
  sheet.df <- make_sheet()
  sheet.df$prot_normal[1] <- "P1_N1"
  run <- run_paired(make_matrix(), sheet.df)
  expect_script_fails(run$result, "different lengths for patient(s): P1")
})

test_that("a patient listed twice stops the run", {
  sheet.df <- bind_rows(make_sheet(), make_sheet()[2, ])
  run <- run_paired(make_matrix(), sheet.df)
  expect_script_fails(run$result, "more than one row")
})

test_that("a duplicated pair and a sample used as both tumor and normal are rejected", {
  duplicated_pair.df <- make_sheet()
  duplicated_pair.df$prot_tumour[1] <- "P1_T1;P1_T1"
  duplicated_pair.df$prot_normal[1] <- "P1_N1;P1_N1"
  expect_script_fails(run_paired(make_matrix(), duplicated_pair.df)$result, "listed more than once")

  both_roles.df <- make_sheet()
  both_roles.df$prot_normal[2] <- "P3_T"
  expect_script_fails(run_paired(make_matrix(), both_roles.df)$result, "both tumor and normal")
})

test_that("a missing replicate is averaged over the pairs that remain", {
  matrix.df <- make_matrix()
  matrix.df$P1_T2[matrix.df$feature_id == "F4"] <- NA     # second pair of P1 missing for F4
  run <- run_paired(matrix.df, make_sheet())
  expect_script_ok(run$result)
  tn.df <- read_tsv(run$tn, col_types = cols(.default = "c", delta = "d", n_pairs_used = "i"))
  f4 <- tn.df %>% filter(feature_id == "F4", patient == "P1")
  f5 <- tn.df %>% filter(feature_id == "F5", patient == "P1")
  expect_equal(f4$n_pairs_used, 1L)
  expect_equal(f5$n_pairs_used, 2L)
  first_pair <- matrix.df$P1_T1[matrix.df$feature_id == "F4"] - matrix.df$P1_N1[matrix.df$feature_id == "F4"]
  expect_equal(f4$delta, first_pair)                       # mean over the one usable pair
  both_pairs <- with(filter(matrix.df, feature_id == "F5"), mean(c(P1_T1 - P1_N1, P1_T2 - P1_N2)))
  expect_equal(f5$delta, both_pairs)
})

test_that("a patient without a usable pair is reported, not silently dropped", {
  sheet.df <- make_sheet()
  sheet.df$prot_tumour[4] <- "P4_missing_from_matrix"
  run <- run_paired(make_matrix(), sheet.df)
  expect_script_ok(run$result)
  excluded.df <- read_text_tsv(file.path(run$outdir, "protein_excluded.tsv"))
  expect_setequal(excluded.df$patient, "P4")
  expect_true(any(str_detect(excluded.df$reason, "patient excluded")))
})

test_that("non-numeric matrix values and duplicated feature ids are rejected", {
  text_value.df <- make_matrix() %>% mutate(across(-feature_id, as.character))
  text_value.df$P2_T[1] <- "oops"
  expect_script_fails(run_paired(text_value.df, make_sheet())$result, "could not be parsed")

  duplicated_id.df <- bind_rows(make_matrix(), make_matrix()[1, ])
  expect_script_fails(run_paired(duplicated_id.df, make_sheet())$result, "duplicated feature ids")
})

test_that("an empty centering set gives NA centered values and a warning, not a crash", {
  run <- run_paired(make_matrix(), make_sheet(), "--min_ave", "1000")
  expect_script_ok(run$result)
  expect_match(run$result$output, "centering set is empty", fixed = TRUE)
  tn.df <- read_tsv(run$tn, col_types = cols(.default = "c"))
  expect_true(all(is.na(tn.df$delta_pc)))
  expect_false(any(is.na(tn.df$delta)))
})

test_that("a plotting failure warns but the tables are still written", {
  outdir <- tempfile("paired_")
  dir.create(outdir)
  file.create(file.path(outdir, "plots"))     # a file where the plots directory should be
  result <- run_script("paired_tn.R", c(
    "--assay", "protein",
    "--matrix", write_input(make_matrix(), basename(tempfile("matrix_", fileext = ".tsv"))),
    "--samplesheet", write_input(make_sheet(), basename(tempfile("sheet_", fileext = ".tsv"))),
    "--outdir", outdir))
  expect_script_ok(result)
  expect_match(result$output, "could not write", fixed = TRUE)
  expect_true(file.exists(file.path(outdir, "protein_tn.tsv.gz")))
  expect_true(file.exists(file.path(outdir, "protein_patients.tsv")))
})
