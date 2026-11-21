#!/usr/bin/env Rscript
#################################################
# Title: Feature map: blocks -> genes ->        #
#   proteins and RNA                            #
# Date: November 10, 2026                       #
# Notes: Step 3. The ProGenome-style "label     #
#   each block with its proteins" layer (a      #
#   separate implementation, not ProGenome's    #
#   scripts).                                   #
#   Every gene is placed in                     #
#   the block that contains its TSS; genes      #
#   whose body spans several blocks are         #
#   counted. Protein analytes are mapped by     #
#   gene symbol, RNA features by Ensembl gene   #
#   id. Stops when genes.tsv and blocks.tsv     #
#   carry different genome build labels, and    #
#   warns when most genes fall in no block      #
#   (usually a build or chromosome naming       #
#   mismatch). Writes (under --outdir):         #
#     genes_in_blocks.tsv  one row per gene     #
#     feature_map.tsv      feature x gene       #
#     features.tsv         one row per feature, #
#                          with its block, a    #
#                          cis_ok flag and a    #
#                          map_status reason    #
#     block_features.tsv   one row per block    #
#     feature_map_summary.tsv,                  #
#     plots/features_per_block_<chrom>.pdf      #
#################################################

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "utils.R"))

##############
# Parameters #
##############

opt <- parse_args_required(list(
  make_option("--genes", type = "character", help = "genes.tsv from gene_models.R"),
  make_option("--blocks", type = "character", help = "blocks.tsv from block_partition.R"),
  make_option("--protein_features", type = "character", default = NULL,
              help = "TSV: feature_id, gene_symbol ('|' between several), extra columns kept"),
  make_option("--rna_features", type = "character", default = NULL,
              help = "TSV: feature_id, gene_id (Ensembl, version optional), extra columns kept"),
  make_option("--drop_ambiguous", type = "logical", default = TRUE,
              help = "leave symbol-ambiguous proteins out of the block cis counts [TRUE]"),
  make_option("--plot_chrom", type = "character", default = "chr1",
              help = "chromosome for the features-per-block plot [chr1]"),
  make_option("--outdir", type = "character", default = ".", help = "output directory")
), required = c("genes", "blocks"))

plots_dir <- file.path(opt$outdir, "plots")
dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)
chrom_order <- paste0("chr", c(1:22, "X", "Y", "M"))

#############
# Read data #
#############

# Explicit column types: IDs and names are text, positions are numbers, so
# that an empty or unusual column is never guessed wrongly.
genes.df <- read_tsv(opt$genes, show_col_types = FALSE,
                     col_types = cols(.default = "c", start = "d", end = "d", tss = "d"))
stop_on_parse_problems(genes.df, basename(opt$genes))
blocks.df <- read_tsv(opt$blocks, show_col_types = FALSE,
                      col_types = cols(.default = col_guess(), block_id = "c", chrom = "c", arm = "c",
                                       start = "d", end = "d", genome_build = "c"))
stop_on_parse_problems(blocks.df, basename(opt$blocks))

# Genes and blocks must be on the same genome build (label set in steps 1 and 2).
for (needed in list(list(genes.df, "genes.tsv"), list(blocks.df, "blocks.tsv"))) {
  if (!"genome_build" %in% names(needed[[1]])) {
    stop(needed[[2]], " has no genome_build column; re-create it with steps 1 and 2 of this version")
  }
}
builds <- c(unique(genes.df$genome_build), unique(blocks.df$genome_build))
if (n_distinct(builds) != 1) {
  stop("genome build differs: genes are ", paste(unique(genes.df$genome_build), collapse = "/"),
       ", blocks are ", paste(unique(blocks.df$genome_build), collapse = "/"))
}

read_features <- function(file, required) {
  if (is.null(file)) return(NULL)
  f.df <- read_tsv(file, col_types = cols(.default = "c"), show_col_types = FALSE)
  missing <- setdiff(required, names(f.df))
  if (length(missing) > 0) stop(basename(file), " lacks column(s): ", paste(missing, collapse = ", "))
  if (anyDuplicated(f.df$feature_id)) stop(basename(file), ": duplicated feature_id")
  f.df
}
protein.df <- read_features(opt$protein_features, c("feature_id", "gene_symbol"))
rna.df <- read_features(opt$rna_features, c("feature_id", "gene_id"))
if (is.null(protein.df) && is.null(rna.df)) stop("give --protein_features and/or --rna_features")

#####################
# Genes into blocks #
#####################

genes_in_blocks.df <- genes.df %>%
  mutate(block_id = locate_in_blocks(chrom, tss, blocks.df),
         n_blocks_body = count_overlapping_blocks(chrom, start, end, blocks.df)) %>%
  left_join(blocks.df %>% select(block_id, arm), by = "block_id")

# How many genes have a TSS outside every block? Most genes unplaced usually
# means the genes and the blocks are on different builds or use different
# chromosome names.
n_unplaced <- sum(is.na(genes_in_blocks.df$block_id))
message("genes in no block: ", n_unplaced, " of ", nrow(genes_in_blocks.df))
if (nrow(genes_in_blocks.df) > 0 && n_unplaced / nrow(genes_in_blocks.df) > 0.5) {
  warning("MOST GENES (", n_unplaced, " of ", nrow(genes_in_blocks.df), ") FALL IN NO BLOCK. ",
          "Likely a genome build or chromosome naming mismatch between genes.tsv and blocks.tsv. ",
          "Shared chromosomes: ", sum(unique(genes.df$chrom) %in% unique(blocks.df$chrom)), ".",
          call. = FALSE, immediate. = TRUE)
}

# One locus per gene symbol: prefer the
# protein-coding locus, then a non-PAR_Y copy, then the first chromosome.
# Symbols that still have more than one candidate are flagged.
symbol_loci.df <- genes_in_blocks.df %>%
  mutate(pref_coding = gene_type != "protein_coding",
         pref_par = grepl("_PAR_Y$", gene_id),
         pref_chrom = match(chrom, chrom_order)) %>%
  arrange(gene_name, pref_coding, pref_par, pref_chrom, start) %>%
  group_by(gene_name) %>%
  mutate(n_loci = n(),
         # more than one locus ties on the preferences: the choice is arbitrary
         symbol_ambiguous = sum(pref_coding == first(pref_coding) & pref_par == first(pref_par)) > 1) %>%
  slice(1) %>%
  ungroup() %>%
  select(gene_name, gene_id, n_loci, symbol_ambiguous)

####################
# Map the features #
####################

gene_cols <- c("gene_id", "gene_name", "gene_type", "chrom", "tss", "block_id", "arm", "n_blocks_body")

# Long table: one row per feature x gene it targets.
map_long <- list()
if (!is.null(protein.df)) {
  map_long$protein <- protein.df %>%
    select(feature_id, gene_symbol) %>%
    mutate(symbol = strsplit(gene_symbol, "|", fixed = TRUE)) %>%
    unnest(symbol) %>%
    mutate(symbol = str_trim(symbol)) %>%
    left_join(symbol_loci.df, by = c("symbol" = "gene_name")) %>%
    left_join(genes_in_blocks.df %>% select(all_of(gene_cols)), by = "gene_id") %>%
    mutate(assay = "protein", input_gene = symbol)
}
if (!is.null(rna.df)) {
  map_long$rna <- rna.df %>%
    select(feature_id, input_gene = gene_id) %>%
    mutate(gene_id = strip_version(input_gene)) %>%
    left_join(genes_in_blocks.df %>% select(all_of(gene_cols)), by = "gene_id") %>%
    mutate(assay = "rna", n_loci = 1L, symbol_ambiguous = FALSE,
           gene_id = if_else(is.na(gene_name), NA_character_, gene_id))
}
feature_map.df <- bind_rows(map_long) %>%
  select(assay, feature_id, input_gene, all_of(gene_cols), n_loci, symbol_ambiguous)

# One row per feature. A feature is "cis_ok" (placed in one block) when it
# targets exactly one gene, that gene was found, and its TSS lies in a block.
# map_status gives the reason for every feature that is not cis_ok:
#   unmapped         no targeted gene was found
#   partly_unmapped  some, not all, targeted genes were found
#   no_block         the gene(s) found have a TSS in no block
#   multi_block      the targeted genes lie in different blocks
#   multi_gene       several targeted genes, all in one block (no single locus)
#   mapped           one gene, one block (cis_ok)
# symbol_ambiguous is not a status here: the graph export decides on it.
features.df <- feature_map.df %>%
  group_by(assay, feature_id) %>%
  summarise(
    n_genes_input = n(),
    n_genes_mapped = sum(!is.na(gene_id)),
    gene_names = paste(na.omit(gene_name), collapse = ";"),
    gene_ids = paste(na.omit(gene_id), collapse = ";"),
    gene_types = paste(na.omit(gene_type), collapse = ";"),
    n_blocks = n_distinct(na.omit(block_id)),
    block_id = if (n_distinct(na.omit(block_id)) == 1) first(na.omit(block_id)) else NA_character_,
    chrom = if (n_distinct(na.omit(chrom)) == 1) first(na.omit(chrom)) else NA_character_,
    tss = if (n() == 1) first(tss) else NA_real_,
    n_blocks_body = if (n() == 1) first(n_blocks_body) else NA_integer_,
    symbol_ambiguous = any(symbol_ambiguous %in% TRUE),
    .groups = "drop"
  ) %>%
  left_join(blocks.df %>% select(block_id, arm), by = "block_id") %>%
  mutate(map_status = case_when(
    n_genes_mapped == 0                ~ "unmapped",
    n_genes_mapped < n_genes_input     ~ "partly_unmapped",
    n_blocks == 0                      ~ "no_block",
    n_blocks > 1                       ~ "multi_block",
    n_genes_input > 1                  ~ "multi_gene",
    TRUE                               ~ "mapped"),
    cis_ok = map_status == "mapped") %>%
  relocate(arm, .after = block_id)

# Keep the extra annotation columns of the inputs (UniProt, target, ...). An
# extra column whose name is already used by a computed column (chrom,
# map_status, cis_ok, ...) is renamed with the prefix "input_" so that it can
# never overwrite or be confused with the computed one.
prefix_clashing_columns <- function(extra.df, reserved_names) {
  clash <- setdiff(intersect(names(extra.df), reserved_names), "feature_id")
  if (length(clash) == 0) return(extra.df)
  new_names <- paste0("input_", clash)
  if (any(new_names %in% c(names(extra.df), reserved_names))) {
    stop("input column(s) ", paste(clash, collapse = ", "), " clash with computed columns, and ",
         paste(new_names, collapse = ", "), " is already taken; rename them in the input file")
  }
  message("input columns renamed to avoid a clash with computed columns: ",
          paste0(clash, " -> input_", clash, collapse = ", "))
  rename_with(extra.df, ~ paste0("input_", .x), all_of(clash))
}
reserved_names <- c(names(features.df), "assay")
extras <- bind_rows(
  if (!is.null(protein.df)) {
    protein.df %>% select(-gene_symbol) %>%
      prefix_clashing_columns(reserved_names) %>% mutate(assay = "protein")
  },
  if (!is.null(rna.df)) {
    rna.df %>% select(-gene_id) %>%
      prefix_clashing_columns(reserved_names) %>% mutate(assay = "rna")
  })
features.df <- features.df %>% left_join(extras, by = c("assay", "feature_id"))

###################
# Block summaries #
###################

# Block counts leave out proteins whose symbol matched several equally ranked
# loci (symbol_ambiguous), unless --drop_ambiguous FALSE.
cis.df <- features.df %>% filter(cis_ok, !(opt$drop_ambiguous & symbol_ambiguous))
block_features.df <- blocks.df %>%
  select(block_index, block_id, chrom, start, end, arm) %>%
  left_join(genes_in_blocks.df %>%
              filter(!is.na(block_id)) %>%
              group_by(block_id) %>%
              summarise(n_genes = n(),
                        n_protein_coding = sum(gene_type == "protein_coding"),
                        protein_coding_genes = paste(gene_name[gene_type == "protein_coding"], collapse = ";"),
                        .groups = "drop"),
            by = "block_id") %>%
  left_join(cis.df %>% count(block_id, assay) %>%
              pivot_wider(names_from = assay, values_from = n, names_prefix = "n_cis_"),
            by = "block_id") %>%
  mutate(across(starts_with("n_"), ~ replace_na(.x, 0L)))
for (col in c("n_cis_protein", "n_cis_rna")) {
  if (!col %in% names(block_features.df)) block_features.df[[col]] <- 0L
}

summary.df <- bind_rows(
  features.df %>% count(assay, map_status, name = "n") %>% rename(group = map_status),
  features.df %>% filter(cis_ok) %>% count(assay, name = "n") %>% mutate(group = "cis_ok"),
  features.df %>% filter(cis_ok, symbol_ambiguous) %>% count(assay, name = "n") %>%
    mutate(group = "cis_ok_symbol_ambiguous"),
  block_features.df %>%
    summarise(protein = sum(n_cis_protein > 0), rna = sum(n_cis_rna > 0)) %>%
    pivot_longer(everything(), names_to = "assay", values_to = "n") %>%
    mutate(group = "blocks_with_cis_feature")
) %>% select(assay, group, n)

###############
# Save output #
###############

write_tsv(genes_in_blocks.df, file.path(opt$outdir, "genes_in_blocks.tsv"), na = "")
write_tsv(feature_map.df, file.path(opt$outdir, "feature_map.tsv"), na = "")
write_tsv(features.df, file.path(opt$outdir, "features.tsv"), na = "")
write_tsv(block_features.df, file.path(opt$outdir, "block_features.tsv"), na = "")
write_tsv(summary.df, file.path(opt$outdir, "feature_map_summary.tsv"), na = "")
print(summary.df, n = Inf)

###########################
# Plot: features per block #
###########################

# How many measured proteins / RNA genes sit in each block of one chromosome.
# This sets the resolution of any within-arm analysis. A plotting failure only
# gives a warning: the tables above are already saved.
make_coverage_plot <- function() {
  block_features.df %>%
    filter(chrom == opt$plot_chrom) %>%
    pivot_longer(c(n_cis_protein, n_cis_rna), names_to = "assay", values_to = "n") %>%
    mutate(assay = recode(assay, n_cis_protein = "Proteins", n_cis_rna = "RNA genes")) %>%
    ggplot() +
    geom_rect(aes(xmin = start / 1e6, xmax = end / 1e6, ymin = 0, ymax = n, fill = assay)) +
    facet_wrap(~assay, ncol = 1, scales = "free_y") +
    scale_fill_manual(values = c("Proteins" = col_focus, "RNA genes" = col_other)) +
    scale_x_continuous(expand = c(0, 0)) +
    theme_simple() +
    theme(legend.position = "none", panel.grid = element_blank()) +
    labs(x = paste(opt$plot_chrom, "position (Mb,", unique(blocks.df$genome_build), ")"),
         y = "Features per block (cis_ok)",
         title = paste("Measured features per haploblock,", opt$plot_chrom))
}
save_plot_safely(make_coverage_plot,
                 file.path(plots_dir, paste0("features_per_block_", opt$plot_chrom, ".pdf")),
                 width = 150, height = 70)
