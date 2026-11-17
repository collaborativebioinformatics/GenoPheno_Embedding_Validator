#!/usr/bin/env Rscript
#################################################
# Title: Gene models from a GENCODE GTF         #
# Date: November 10, 2026                       #
# Notes: Step 2. Reads the "gene" records of a  #
#   GTF (plain or .gz; other record types are   #
#   skipped) and writes one row per gene, with  #
#   0-based half-open coordinates and the TSS:  #
#     + strand: TSS = start (0-based)           #
#     - strand: TSS = end - 1                   #
#   Chromosome names are converted to "chr"     #
#   style by default (1 -> chr1, MT -> chrM).   #
#   Contigs other than chr1-22, X, Y and M are  #
#   dropped and reported. The genome build      #
#   label is stored in column genome_build.     #
#################################################

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "utils.R"))

##############
# Parameters #
##############

opt <- parse_args_required(list(
  make_option("--gtf_genes", type = "character", help = "GENCODE GTF (plain or .gz)"),
  make_option("--chrom_style", type = "character", default = "chr",
              help = "chr: rename chromosomes to chr style (1 -> chr1, MT -> chrM); asis: keep the names as given [chr]"),
  make_option("--assembly", type = "character", default = "GRCh38",
              help = "genome build label stored in the output (column genome_build) [GRCh38]"),
  make_option("--out", type = "character", default = "genes.tsv", help = "output TSV")
), required = "gtf_genes")

chrom_style <- match.arg(opt$chrom_style, c("chr", "asis"))
chrom_order <- paste0("chr", c(1:22, "X", "Y", "M"))

# GTF attribute value, e.g. gtf_attr(x, "gene_name").
gtf_attr <- function(attributes, key) {
  str_match(attributes, paste0(key, ' "([^"]+)"'))[, 2]
}

# Name the GTF columns that are read and keep the gene records only.
keep_gene_records <- function(chunk.df, pos) {
  chunk.df %>%
    set_names(c("chrom", "source", "feature", "start1", "end1", "strand", "attributes")) %>%
    filter(feature == "gene")
}

#############
# Read data #
#############

# The GTF is read in chunks of 100,000 lines and only the gene records are
# kept as each chunk arrives. Transcript, exon and CDS records (the large
# majority of the file, each with a long attribute string) are therefore never
# held in memory together. The whole file is still scanned once.
gtf.df <- read_tsv_chunked(opt$gtf_genes, callback = DataFrameCallback$new(keep_gene_records),
                           chunk_size = 100000, comment = "#", col_names = FALSE,
                           col_types = "cccii-c-c", progress = FALSE)
if (nrow(gtf.df) == 0) stop("no 'gene' records found in ", opt$gtf_genes)

##################
# Build the table #
##################

# Convert chromosome names, then keep the standard chromosomes and report the rest.
gtf.df <- gtf.df %>%
  mutate(chrom = if (chrom_style == "chr") normalize_chrom(chrom) else chrom)
report_dropped_contigs(gtf.df$chrom[!gtf.df$chrom %in% chrom_order], "gene models")
gtf.df <- gtf.df %>% filter(chrom %in% chrom_order)
if (nrow(gtf.df) == 0) {
  stop("no gene is left on chr1-22, X, Y or M. Check --chrom_style (now '", chrom_style, "').")
}
if (anyNA(gtf.df$start1) || anyNA(gtf.df$end1)) stop("gene records with a missing start or end")
if (!all(gtf.df$strand %in% c("+", "-"))) stop("gene records need strand + or -")

genes.df <- gtf.df %>%
  transmute(
    gene_id = strip_version(gtf_attr(attributes, "gene_id")),
    gene_id_version = gtf_attr(attributes, "gene_id"),
    gene_name = gtf_attr(attributes, "gene_name"),
    gene_type = gtf_attr(attributes, "gene_type"),
    chrom,
    start = start1 - 1L,          # GTF is 1-based, closed
    end = end1,
    strand,
    tss = if_else(strand == "-", end - 1L, start),
    genome_build = opt$assembly
  )

if (anyDuplicated(genes.df$gene_id)) stop("duplicated gene ids in the GTF")
if (anyNA(genes.df$gene_name)) stop("gene records without a gene_name attribute")

###############
# Save output #
###############

write_tsv(genes.df, opt$out, na = "")
message("genes: ", nrow(genes.df), " (", sum(genes.df$gene_type == "protein_coding", na.rm = TRUE),
        " protein-coding), ", opt$assembly)
