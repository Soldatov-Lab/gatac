"""
Genomic constants and chromosome sizes.
"""

# GRCh38 / hg38 chromosome sizes (Gencode v41)
HG38 = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559,
    "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
    "chr7": 159345973, "chr8": 145138636, "chr9": 138394717,
    "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16": 90338345, "chr17": 83257441, "chr18": 80373285,
    "chr19": 58617616, "chr20": 64444167, "chr21": 46709983,
    "chr22": 50818468, "chrX": 156040895, "chrY": 57227415
}

# GRCh37 / hg19 chromosome sizes
HG19 = {
    "chr1": 249250621, "chr2": 243199373, "chr3": 198022430,
    "chr4": 191154276, "chr5": 180915260, "chr6": 171115067,
    "chr7": 159138663, "chr8": 146364022, "chr9": 141213431,
    "chr10": 135534747, "chr11": 135006516, "chr12": 133851895,
    "chr13": 115169878, "chr14": 107349540, "chr15": 102531392,
    "chr16": 90354753, "chr17": 81195210, "chr18": 78077248,
    "chr19": 59128983, "chr20": 63025520, "chr21": 48129895,
    "chr22": 51304566, "chrX": 155270560, "chrY": 59373566
}

# GRCm39 / mm39 chromosome sizes (Gencode vM30)
MM39 = {
    "chr1": 195154279, "chr2": 181755017, "chr3": 159745316,
    "chr4": 156860686, "chr5": 151758149, "chr6": 149588044,
    "chr7": 144995196, "chr8": 130127694, "chr9": 124359700,
    "chr10": 130530862, "chr11": 121973369, "chr12": 120092757,
    "chr13": 120883175, "chr14": 125139656, "chr15": 104073951,
    "chr16": 98008968, "chr17": 95294699, "chr18": 90720763,
    "chr19": 61420004, "chrX": 169476592, "chrY": 91455967
}

# GRCm38 / mm10 chromosome sizes (Gencode vM25)
MM10 = {
    "chr1": 195471971, "chr2": 182113224, "chr3": 160039680,
    "chr4": 156508116, "chr5": 151834684, "chr6": 149736546,
    "chr7": 145441459, "chr8": 129401213, "chr9": 124595110,
    "chr10": 130694993, "chr11": 122082543, "chr12": 120129022,
    "chr13": 120421639, "chr14": 124902244, "chr15": 104043685,
    "chr16": 98207768, "chr17": 94987271, "chr18": 90702639,
    "chr19": 61431566, "chrX": 171031299, "chrY": 91744698
}

GENOMES = {
    "hg38": HG38,
    "GRCh38": HG38,
    "hg19": HG19,
    "GRCh37": HG19,
    "mm10": MM10,
    "GRCm38": MM10,
    "mm39": MM39,
    "GRCm39": MM39,
}

def get_chrom_sizes(genome_name: str) -> dict[str, int]:
    """
    Get chromosome sizes for a given genome name.
    
    Parameters
    ----------
    genome_name : str
        Name of the genome (e.g., 'hg38', 'mm10')
        
    Returns
    -------
    dict
        Dictionary of chromosome names and sizes
        
    Raises
    ------
    ValueError
        If genome name is not recognized
    """
    if genome_name in GENOMES:
        return GENOMES[genome_name]
    raise ValueError(f"Unknown genome: {genome_name}. Available genomes: {list(GENOMES.keys())}")
