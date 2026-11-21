#!/usr/bin/env Rscript
#################################################
# Title: Protein graph export                   #
# Date: November 10, 2026                       #
# Notes: Step 5. Turns the outputs of steps 1,  #
#   3 and 4 into node and edge CSVs and a       #
#   protein x patient matrix:                   #
#     graph/nodes_patient.csv     id,label      #
#     graph/nodes_block.csv       id,label,...  #
#     graph/nodes_protein.csv     id,label,...  #
#     graph/edges_protein_on_block.csv          #
#        source (protein), target (block),      #
#        type = ON_BLOCK                        #
#     proteins/protein_features.tsv             #
#        protein_id, chrom, block_id, pos (TSS),#
#        gene_names, gene_ids, symbol_ambiguous,#
#        map_status                             #
#     proteins/protein_tn_matrix.tsv.gz         #
#        proteins x patients, T-N (log2)        #
#   Only proteins that map to exactly one gene  #
#   in exactly one block (cis_ok) get an        #
#   ON_BLOCK edge. With --drop_ambiguous TRUE   #
#   (default) proteins whose gene symbol        #
#   matched several equally ranked loci get no  #
#   edge either (map_status ambiguous_symbol).  #
#   The matrix keeps every protein, for late    #
#   fusion. The T-N input may be the step 4     #
#   table or any long table with the columns    #
#   feature_id, patient and the chosen value.   #
#################################################

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "utils.R"))

##############
# Parameters #
##############

opt <- parse_args_required(list(
  make_option("--blocks", type = "character", help = "blocks.tsv from block_partition.R"),
  make_option("--features", type = "character", help = "features.tsv from feature_map.R"),
  make_option("--tn", type = "character",
              help = "protein_tn.tsv.gz from paired_tn.R, or a long table: feature_id, patient, <value>"),
  make_option("--value", type = "character", default = "delta",
              help = "which T-N to export: delta, delta_pc or delta_dc [delta]"),
  make_option("--drop_ambiguous", type = "logical", default = TRUE,
              help = "no ON_BLOCK edge for proteins with a symbol-ambiguous gene [TRUE]"),
  make_option("--digits", type = "integer", default = 4L,
              help = "decimal places kept in the exported matrix [4]"),
  make_option("--outdir", type = "character", default = ".", help = "output directory")
), required = c("blocks", "features", "tn"))

value_col <- match.arg(opt$value, c("delta", "delta_pc", "delta_dc"))
if (is.na(opt$digits) || opt$digits < 0) stop("--digits must be 0 or larger")
graph_dir <- file.path(opt$outdir, "graph")
protein_dir <- file.path(opt$outdir, "proteins")
dir.create(graph_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(protein_dir, showWarnings = FALSE, recursive = TRUE)

#############
# Read data #
#############

# Stop if a table lacks columns that this script needs; returns the header.
require_columns <- function(file, needed) {
  header <- names(read_tsv(file, n_max = 0, col_types = cols(.default = "c"), show_col_types = FALSE))
  missing <- setdiff(needed, header)
  if (length(missing) > 0) {
    stop(basename(file), " lacks column(s): ", paste(missing, collapse = ", "), call. = FALSE)
  }
  invisible(header)
}

blocks_needed <- c("block_id", "chrom", "start", "end", "arm")
require_columns(opt$blocks, blocks_needed)
blocks.df <- read_tsv(opt$blocks, show_col_types = FALSE,
                      col_types = cols_only(block_id = "c", chrom = "c", start = "d", end = "d", arm = "c"))
stop_on_parse_problems(blocks.df, basename(opt$blocks))

# Explicit types: IDs and mapping fields are text even if a column is empty
# (an all-unmapped panel has no block_id or chrom at all), flags are logical
# and the TSS is a number.
features_needed <- c("assay", "feature_id", "gene_names", "gene_ids", "cis_ok", "symbol_ambiguous",
                     "chrom", "block_id", "tss", "map_status")
require_columns(opt$features, features_needed)
features.df <- read_tsv(opt$features, show_col_types = FALSE,
                        col_types = cols_only(assay = "c", feature_id = "c", gene_names = "c", gene_ids = "c",
                                              cis_ok = "l", symbol_ambiguous = "l", chrom = "c",
                                              block_id = "c", tss = "d", map_status = "c")) %>%
  filter(assay == "protein")
stop_on_parse_problems(features.df, basename(opt$features))
if (nrow(features.df) == 0) stop("no protein features in ", opt$features)
if (anyDuplicated(features.df$feature_id)) {
  stop("duplicated protein rows in ", basename(opt$features), ", e.g. ",
       paste(head(unique(features.df$feature_id[duplicated(features.df$feature_id)]), 3), collapse = ", "))
}

# The T-N table: only feature_id, patient and the chosen value are used. The
# value is read as a number; text in it is a parsing problem and stops the run.
tn_header <- require_columns(opt$tn, c("feature_id", "patient", value_col))
tn_types <- c(list(assay = "c", feature_id = "c", patient = "c"), setNames(list("d"), value_col))
tn_types <- tn_types[names(tn_types) %in% tn_header]   # the step 4 column "assay" is optional
tn.df <- read_tsv(opt$tn, show_col_types = FALSE, col_types = do.call(cols_only, tn_types))
stop_on_parse_problems(tn.df, basename(opt$tn))

# Check the T-N table: complete keys, one value per protein and patient, no
# infinite values, and (if the step 4 table is given) protein data only.
if ("assay" %in% names(tn.df) && any(tn.df$assay != "protein", na.rm = TRUE)) {
  stop(basename(opt$tn), " has rows for an assay other than 'protein'")
}
if (nrow(tn.df) == 0) stop(basename(opt$tn), " has no rows")
if (any(is.na(tn.df$feature_id) | tn.df$feature_id == "")) stop(basename(opt$tn), ": missing feature_id")
if (any(is.na(tn.df$patient) | tn.df$patient == "")) stop(basename(opt$tn), ": missing patient")
if (any(is.infinite(tn.df[[value_col]]))) stop(basename(opt$tn), ": infinite values in ", value_col)
if (anyDuplicated(tn.df[c("feature_id", "patient")])) {
  duplicated.df <- tn.df[duplicated(tn.df[c("feature_id", "patient")]), ]
  stop(basename(opt$tn), ": ", nrow(duplicated.df), " duplicated (feature_id, patient) row(s), e.g. ",
       duplicated.df$feature_id[1], " / ", duplicated.df$patient[1],
       ". Each protein needs one value per patient.")
}
if ("protein_id" %in% tn.df$patient) stop("a patient may not be called 'protein_id' (matrix column name)")

##########################
# Proteins and positions #
##########################

# One row per protein in the T-N table. Position = the TSS of its gene, when it
# maps to exactly one gene in exactly one block (cis_ok). Proteins whose gene
# symbol matched several equally ranked loci (symbol_ambiguous) get no block
# unless --drop_ambiguous FALSE. Every protein without a block keeps a reason
# in map_status.
protein_features.df <- tibble(protein_id = unique(tn.df$feature_id)) %>%
  left_join(features.df %>% select(protein_id = feature_id, gene_names, gene_ids, symbol_ambiguous,
                                   cis_ok, chrom, block_id, tss, map_status),
            by = "protein_id") %>%
  mutate(cis_ok = cis_ok %in% TRUE,
         symbol_ambiguous = symbol_ambiguous %in% TRUE,
         has_edge = cis_ok & !(opt$drop_ambiguous & symbol_ambiguous),
         map_status = case_when(
           is.na(map_status)         ~ "not_in_feature_list",
           cis_ok & !has_edge        ~ "ambiguous_symbol",
           TRUE                      ~ map_status),
         block_id = if_else(has_edge, block_id, NA_character_),
         chrom = if_else(has_edge, chrom, NA_character_),
         pos = if_else(has_edge, tss, NA_real_),
         haplograph_block_id = make_haplograph_block_id(block_id))

###############
# T-N matrix  #
###############

# Wide matrix, one row per protein and one column per patient (a tidy long table
# of thousands of proteins x many patients would be large and slow to read).
# Values are rounded to --digits decimal places.
tn_matrix.df <- tn.df %>%
  select(protein_id = feature_id, patient, value = all_of(value_col)) %>%
  mutate(value = round(value, opt$digits)) %>%
  pivot_wider(names_from = patient, values_from = value) %>%
  arrange(match(protein_id, protein_features.df$protein_id))
patients <- setdiff(names(tn_matrix.df), "protein_id")

##################
# Graph tables   #
##################

nodes_patient.df <- tibble(id = patients, label = "Patient")
nodes_block.df <- blocks.df %>%
  transmute(id = block_id, label = "Block", chrom, start, end, arm,
            haplograph_block_id = make_haplograph_block_id(block_id))
nodes_protein.df <- protein_features.df %>%
  transmute(id = protein_id, label = "Protein", gene_names)
edges_on_block.df <- protein_features.df %>%
  filter(has_edge) %>%
  transmute(source = protein_id, target = block_id, type = "ON_BLOCK")

# Every ON_BLOCK target must be a block node.
if (!all(edges_on_block.df$target %in% nodes_block.df$id)) {
  stop("some ON_BLOCK targets are not in the block table; features.tsv and blocks.tsv do not belong together")
}

###############
# Save output #
###############

write_csv(nodes_patient.df, file.path(graph_dir, "nodes_patient.csv"), na = "")
write_csv(nodes_block.df, file.path(graph_dir, "nodes_block.csv"), na = "")
write_csv(nodes_protein.df, file.path(graph_dir, "nodes_protein.csv"), na = "")
write_csv(edges_on_block.df, file.path(graph_dir, "edges_protein_on_block.csv"), na = "")
write_tsv(protein_features.df %>%
            select(protein_id, chrom, block_id, haplograph_block_id, pos, gene_names, gene_ids,
                   symbol_ambiguous, map_status),
          file.path(protein_dir, "protein_features.tsv"), na = "")
write_tsv(tn_matrix.df, file.path(protein_dir, "protein_tn_matrix.tsv.gz"), na = "")
message("proteins: ", nrow(protein_features.df), " (", nrow(edges_on_block.df),
        " with an ON_BLOCK edge); patients: ", length(patients), "; T-N column: ", value_col,
        "; rounded to ", opt$digits, " decimals")
