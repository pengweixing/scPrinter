import pooch
import os
import stat
from pooch import Decompress, Untar, Unzip
_datasets = None

def giverightstothegroup(fname, action, pooch):
    '''
    Processes the downloaded file and returns a new file name.

    The function **must** take as arguments (in order):

    fname : str
        The full path of the file in the local data storage
    action : str
        Either: "download" (file doesn't exist and will be downloaded),
        "update" (file is outdated and will be downloaded), or "fetch"
        (file exists and is updated so no download is necessary).
    pooch : pooch.Pooch
        The instance of the Pooch class that is calling this function.

    The return value can be anything but is usually a full path to a file
    (or list of files). This is what will be returned by Pooch.fetch and
    pooch.retrieve in place of the original file path.
    '''
    if action == "download":
        os.chmod(fname, stat.S_IRWXG | stat.S_IRWXU)
    if 'tar' in fname:
        return Untar()(fname, action, pooch)
    elif 'gz' in fname:
        return Decompress(method="gzip")(fname, action, pooch)
    elif 'zip' in fname:
        return Unzip()(fname, action, pooch)
    return fname

def datasets():
    global _datasets
    if _datasets is None:
        # dir1 = os.path.dirname(pooch.__file__)
        # dir1 = "/".join(dir1.split("/")[:-1])
        # dir = os.path.join(dir1, 'scprinter_cache')
        _datasets = pooch.create(
            path=pooch.os_cache("scprinter"),
            base_url="",
            env="SCPRINTER_DATA",
            registry={
                # scp files
                "dispersion_model_py.h5": "md5:cbd6cefed73f36aaf121aa73f2d2b658",
                'nucleosome_model_py.pt' : 'md5:fc58e8698b1f869b67b2c1b7b4398b3b',
                'TFBS_model_py.pt' : 'md5:5bd79a9c4f3374241a6f4341eb39fe2c',
                'TFBS_model_model1_py.pt': 'md5:7893684aa234df3b58995b212d9a8363',
                # motif database
                "JASPAR2022_core_nonredundant.jaspar": "md5:af268b3e9589f52440007b43cba358f8",
                'CisBP_Human.jaspar':"md5:23b85a4cd8299416dd5d85516c0cdcbf",
                'CisBPJASPA.jaspar':"md5:7f965084f748d9e91f950a7981ffd7d5",

                # bias file
                # "hg38Tn5Bias.h5": "md5:5ff8b43c50eb23639e3d93b5b1e8a50a",
                "ce11Tn5Bias.tar.gz": "md5:10d8d17f94f695c06c0f66968f67b55b",
                "danRer11Tn5Bias.tar.gz": "md5:8d4fe94ccbde141f6edefc1f0ce36c10",
                "dm6Tn5Bias.tar.gz": "md5:7f256a41b7232bd5c3389b0e190d9788",
                "hg38Tn5Bias.tar.gz": "md5:89f205e6be682b15f87a2c2cc00e8cbd",
                "mm10Tn5Bias.tar.gz": "md5:901b928946b65e7bfba3a93e085f19f0",
                "panTro6Tn5Bias.tar.gz": "md5:ba208a4cdc2e1fc09d66cac44e85e001",
                "sacCer3Tn5Bias.tar.gz": "md5:ed811aabe1ffa4bdb1520d4b25ee9289",

                # Genome files
                "gencode_v41_GRCh37.gff3.gz": "sha256:df96d3f0845127127cc87c729747ae39bc1f4c98de6180b112e71dda13592673",
                "gencode_v41_GRCh37.fa.gz": "sha256:94330d402e53cf39a1fef6c132e2500121909c2dfdce95cc31d541404c0ed39e",
                "gencode_v41_GRCh38.gff3.gz": "sha256:b82a655bdb736ca0e463a8f5d00242bedf10fa88ce9d651a017f135c7c4e9285",
                "gencode_v41_GRCh38.fa.gz": "sha256:4fac949d7021cbe11117ddab8ec1960004df423d672446cadfbc8cca8007e228",
                "gencode_vM25_GRCm38.gff3.gz": "sha256:e8ed48bef6a44fdf0db7c10a551d4398aa341318d00fbd9efd69530593106846",
                "gencode_vM25_GRCm38.fa.gz": "sha256:617b10dc7ef90354c3b6af986e45d6d9621242b64ed3a94c9abeac3e45f18c17",
                "gencode_vM30_GRCm39.gff3.gz": "sha256:6f433e2676e26569a678ce78b37e94a64ddd50a09479e433ad6f75e37dc82e48",
                "gencode_vM30_GRCm39.fa.gz": "sha256:3b923c06a0d291fe646af6bf7beaed7492bf0f6dd5309d4f5904623cab41b0aa",

                # Tutorial files
                "BMMCTutorial.zip": "md5:d9027cf73b558d03276483384ddad88c"
            },
            urls={
                "dispersion_model_py.h5": "https://drive.google.com/uc?export=download&id=1O7zGvmJIArJjLooW0pzZyEbm2AaFE2bg",
                # "TFBS_model_py.h5": "https://drive.google.com/uc?export=download&id=1gogFjnhhiVn8oJRkFNa1x1uEN1bCYKKA",
                'nucleosome_model_py.pt': 'https://drive.google.com/uc?export=download&id=16TVhzfSAva4um_mB0hpoOlByVlSz3Yv9',
                'TFBS_model_py.pt': 'https://drive.google.com/uc?export=download&id=1gtJIbkNEAq93s4i-WNV89lxmM-0cRotW',
                'TFBS_model_model1_py.pt': 'https://drive.google.com/uc?export=download&id=1SaY4zv_uMXyDTLDZMsAhkWU-j1WvoCrn',
                # motif database
                "JASPAR2022_core_nonredundant.jaspar": "https://drive.google.com/uc?export=download&id=1YmRZ3sABLJvv9uj40BY97Rdqyodd852P",
                'CisBP_Human.jaspar': "https://drive.google.com/uc?export=download&id=1IVcg27kxzG5TtnjqFrheGxXa-0kfAOW7",
                'CisBPJASPA.jaspar': "https://drive.google.com/uc?export=download&id=1I62z-JZaQOnue7iimU0Q8Uf7ZjpEHLGn",


                # bias file
                # "hg38Tn5Bias.h5": "https://drive.google.com/uc?export=download&confirm=s5vl&id=1Ias_dP2docuXRGcoQrGHMNOlIwhgJZoJ",

                "ce11Tn5Bias.tar.gz": "https://zenodo.org/record/7121027/files/ce11Tn5Bias.tar.gz",
                "danRer11Tn5Bias.tar.gz": "https://zenodo.org/record/7121027/files/danRer11Tn5Bias.tar.gz",
                "dm6Tn5Bias.tar.gz": "https://zenodo.org/record/7121027/files/dm6Tn5Bias.tar.gz",
                "hg38Tn5Bias.tar.gz": "https://zenodo.org/record/7121027/files/hg38Tn5Bias.tar.gz",
                "mm10Tn5Bias.tar.gz": "https://zenodo.org/record/7121027/files/mm10Tn5Bias.tar.gz",
                "panTro6Tn5Bias.tar.gz": "https://zenodo.org/record/7121027/files/panTro6Tn5Bias.tar.gz",
                "sacCer3Tn5Bias.tar.gz": "https://zenodo.org/record/7121027/files/sacCer3Tn5Bias.tar.gz",

                "gencode_v41_GRCh37.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/GRCh37_mapping/gencode.v41lift37.basic.annotation.gff3.gz",
                "gencode_v41_GRCh37.fa.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/GRCh37_mapping/GRCh37.primary_assembly.genome.fa.gz",
                "gencode_v41_GRCh38.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.basic.annotation.gff3.gz",
                "gencode_v41_GRCh38.fa.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/GRCh38.primary_assembly.genome.fa.gz",
                "gencode_vM25_GRCm38.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.basic.annotation.gff3.gz",
                "gencode_vM25_GRCm38.fa.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/GRCm38.primary_assembly.genome.fa.gz",
                "gencode_vM30_GRCm39.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M30/gencode.vM30.basic.annotation.gff3.gz",
                "gencode_vM30_GRCm39.fa.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M30/GRCm39.primary_assembly.genome.fa.gz",

                "BMMCTutorial.zip": 'https://drive.google.com/uc?export=download&confirm=s5vl&id=1L-9yPkNKf-IcWgubfp2Yz2oYusEVEico'
            },
        )
    return _datasets



def JASPAR2022_core():
    return str(datasets().fetch("JASPAR2022_core_nonredundant.jaspar", processor=giverightstothegroup))

def CisBP_Human():
    return str(datasets().fetch("CisBP_Human.jaspar", processor=giverightstothegroup))

def CisBPJASPA():
    return str(datasets().fetch("CisBPJASPA.jaspar", processor=giverightstothegroup))

def TFBS_model():
    """
    A wrapper function to get Pretrained TFBS model
    You can also get it by `scprinter.datasets.pretrained_TFBS_model`

    Returns
    -------
    str: path to the TFBS model
    """
    return str(datasets().fetch("TFBS_model_py.pt", processor=giverightstothegroup))


def TFBS_model_classI():
    """
    A wrapper function to get Pretrained TFBS model (class I, meaning only TFs that left a strong footprints)
    You can also get it by `scprinter.datasets.pretrained_TFBS_model_classI`

    Returns
    -------
    str: path to the TFBS model (class I)
    """
    return str(datasets().fetch("TFBS_model_model1_py.pt", processor=giverightstothegroup))

def NucBS_model():
    """
    A wrapper function to get Pretrained NucBS model
    You can also get it by `scprinter.datasets.pretrained_NucBS_model`

    Returns
    -------
    str: path to the NucBS model
    """
    return str(datasets().fetch("nucleosome_model_py.pt", processor=giverightstothegroup))

def dispersion_model():
    """
    A wrapper function to get Pretrained dispersion model
    You can also get it by `scprinter.datasets.pretrained_dispersion_model`

    Returns
    -------
    str: path to the dispersion model
    """
    return str(datasets().fetch("dispersion_model_py.h5", processor=giverightstothegroup))

def BMMCTutorial():
    """
    A wrapper function to get BMMC Tutorial data.

    Returns
    -------
    str: path to the BMMC Tutorial data
    """
    files = datasets().fetch("BMMCTutorial.zip", processor=giverightstothegroup)
    dict1 = {}
    for f in files:
        if 'bed' in f:
            dict1['region'] = f
        elif 'Fragments' in f:
            dict1['fragments'] = f
        elif 'groupInfo' in f:
            dict1['groupInfo'] = f
        elif 'barcodeGrouping' in f:
            dict1['barcodeGrouping'] = f
    return dict1


pretrained_TFBS_model = datasets().fetch("TFBS_model_py.pt", processor=giverightstothegroup)
pretrained_TFBS_model_classI = datasets().fetch("TFBS_model_model1_py.pt", processor=giverightstothegroup)
pretrained_NucBS_model = datasets().fetch("nucleosome_model_py.pt", processor=giverightstothegroup)
pretrained_dispersion_model = datasets().fetch("dispersion_model_py.h5", processor=giverightstothegroup)