from __future__ import annotations
import anndata
from .import genome
from .utils import *
import h5py
import pyBigWig
from tqdm.auto import tqdm, trange
import gffutils
import gc
import atexit
from pathlib import Path


class PyPrinter():
    """
    Core Class of scprinter

    Parameters
    ----------
    adata: anndata.AnnData or snapatac.adataset
        snapatac adata or adataset, which stores insertion_profile
    adata_path: str
        path to adata
    insertion_profile: dict
        direct assignment of insertion_profile
    genome: scp.genome
        a scp.genome object

    """
    def __init__(self,
                 adata: snap.AnnData | anndata.AnnData | None = None,
                 adata_path: str | Path | None = None,
                 insertion_profile: dict | None = None, # direct assignment of insertion_profile
                 genome: genome.Genome | None = None, # a scp.genome object
                 ) -> None:

        self.bindingscoreadata = {}
        self.footprintsadata = {}
        self.gene_region_width=1000
        self.genome=genome
        if adata is not None:
            # Lazy loading of insertion_profile
            if 'binding score' not in adata.uns:
                adata.uns['binding score'] = {}
            if 'footprints' not in adata.uns:
                adata.uns['footprints'] = {}

            self.insertion_file = adata
            self.obs = self.insertion_file.obs
            self.obsm = self.insertion_file.obsm
            self.uns = self.insertion_file.uns
            self.unique_string = self.insertion_file.uns['unique_string']
            # Start to read in the binding score data and footprint data
            for result_key, save in zip(['binding score', 'footprints'],
                                         [self.bindingscoreadata, self.footprintsadata]):
                remove_key = []
                for save_key in adata.uns[result_key]:
                    p = adata.uns[result_key][save_key]
                    success = False
                    if p != "None" and len(p) > 0 and os.path.exists(p):
                        try:
                            # Only allow read when load.
                            print("loading", save_key, p)
                            save[save_key] = snap.read(p, backed='r')
                            success = True
                        except:
                            pass
                    if not success:
                        remove_key.append(save_key)


                # Update uns
                a = adata.uns[result_key]
                for key in remove_key:
                    del a[key]
                adata.uns[result_key] = a

        # Get genome related info
        if genome is not None:
            bias_path = str(genome.fetch_bias())
            bias_bw_path = bias_path.replace(".h5", ".bw")
            adata.uns['bias_path'] = bias_path
            adata.uns['bias_bw'] = bias_bw_path
            # If it's the very first time you load this genome

            # Create a bias_bw
            if not os.path.exists(adata.uns['bias_bw']):
                print ("creating bias bigwig (runs for new bias h5 file)")
                with h5py.File(adata.uns['bias_path'], 'r') as dct:
                    precomputed_bias = {chrom: np.array(dct[chrom]) for chrom in dct.keys()}
                    bw = pyBigWig.open(adata.uns['bias_bw'], 'w')
                    header = []
                    for chrom in precomputed_bias:
                        sig = precomputed_bias[chrom]
                        length = sig.shape[-1]
                        header.append((chrom, length))
                    bw.addHeader(header, maxZooms=0)
                    for chrom in tqdm(precomputed_bias):
                        sig = precomputed_bias[chrom]
                        bw.addEntries(str(chrom),
                                      np.arange(len(sig)),
                                      values=sig.astype('float'), span=1, )
                    bw.close()

            gff = genome.fetch_gff()
            adata.uns['gff_db'] = str(gff)+".db"
            # First time create a gff_db to query
            if not os.path.exists(adata.uns['gff_db']):
                print ("Creating GFF database (Runs for new genome)")
                # Specifying the id_spec was necessary for gff files from NCBI.
                self.gff_db = gffutils.create_db(gff, adata.uns['gff_db'],
                                                 id_spec={'gene': 'gene_name', 'transcript': "transcript_id"},
                                            merge_strategy="create_unique")
            else:
                print ("Initializing GFF-db")
                self.gff_db = gffutils.FeatureDB(adata.uns['gff_db'])

        self.insertion_profile = insertion_profile

        if adata_path is not None:
            self.file_path = adata_path

        # load dispersion models
        self.load_disp_model()
        # initialize binding score model empty dict
        self.bindingScoreModel = {}
        return


    def remove_bindingscore(self, key: str):
        """
        Remove a binding score adata

        Parameters
        ----------
        key: str
            key of the binding score adata to be removed

        """
        a = self.insertion_file.uns['binding score']
        if a[key] != 'None':
            os.remove(a[key])
        del a[key]
        self.insertion_file.uns['binding score'] = a
        del self.bindingscoreadata[key]


    def remove_footprints(self, key: str):
        """
        Remove a footprints adata

        Parameters
        ----------
        key: str
            key of the footprints adata to be removed

        """
        a = self.insertion_file.uns['footprints']
        if a[key] != 'None':
            os.remove(a[key])
        del a[key]
        self.insertion_file.uns['footprints'] = a
        del self.bindingscoreadata[key]

    def fetch_insertion_profile(self, set_global=False):
        """
        Fetch the insertion profile from the adata file
        If the insertion profile is not in csc format, it will be converted to csc format, split by chromosome, and saved
        for future use.

        """
        # global insertion_profile
        if self.insertion_profile is None:
            print ("Insertion profile from csr to csc")
            indx = list(np.cumsum(self.insertion_file.uns['reference_sequences']['reference_seq_length']).astype('int'))
            start = [0] + indx
            end = indx
            self.insertion_profile = split_insertion_profile(
                self.insertion_file.obsm['insertion'],
                self.insertion_file.uns['reference_sequences']['reference_seq_name'],
                start,
                end, to_csc=True)
            gc.collect()
        if set_global:
            unique_string = self.unique_string
            print (unique_string)
            globals()[unique_string + "insertion_profile"] = self.insertion_profile

        return self.insertion_profile

    def close(self):
        self.insertion_file.close()
        for data in self.bindingscoreadata.values():
            try:
                data.close()
            except:
                pass
        for data in self.footprintsadata.values():
            try:
                data.close()
            except:
                pass

    def __repr__(self):
        print("head project")
        print (self.insertion_file)
        if len(self.bindingscoreadata) > 0:
            print ("detected %d bindingscoreadata" % len(self.bindingscoreadata))
            for key in self.bindingscoreadata:
                print("name", key)
                response = str(self.bindingscoreadata[key])
                response = response.split("\n")
                if len(response) > 1:
                    a = response[-1].strip().split(": ")[1].split(", ")
                    new_final = "    obsm: %d regions results in total: e.g. %s"  %(len(a), a[0])
                    response[-1] = new_final
                    print ("\n".join(response))
                else:
                    print (response)

        if len(self.footprintsadata) > 0:
            print ("detected %d footprintsadata" % len(self.footprintsadata))
            for key in self.footprintsadata:
                print ("name", key)
                response = str(self.footprintsadata[key])
                response = response.split("\n")
                if len(response) > 1:
                    a = response[-1].strip().split(": ")[1].split(", ")
                    new_final = "    obsm: %d regions results in total: e.g. %s" % (len(a), a[0])
                    response[-1] = new_final
                    print("\n".join(response))
                else:
                    print (response)

        return ""



    def load_disp_model(self,
                        path: str | Path | None = None,
                        set_global=False):
        """
        Load the dispersion model from the path
        When path is None, the default pretrained model will be loaded

        Parameters
        ----------
        path: str | Path | None
            path to the dispersion model

        """
        from .datasets import pretrained_dispersion_model
        if path is None:
            path = pretrained_dispersion_model
        self.dispersionModel = loadDispModel(path)
        if set_global:
            globals()[self.unique_string + "_dispModels"] = self.dispersionModel

    def set_global_bindingscore_model(self,
                                      key: str):
        globals()[self.unique_string + "_bindingscoremodel"] = self.bindingScoreModel[key]

    def load_bindingscore_model(self,
                                key: str,
                                path: str | Path | None = None,
                                set_global=False):
        """
        Load the binding score model from the path and save it to the adata file with the key

        Parameters
        ----------
        key: str
            key to save the binding score model
        path: str | Path | None
            path to the binding score model

        """
        self.bindingScoreModel[key] = loadBindingScoreModel_pt(path)
        if set_global:
            globals()[self.unique_string + "_bindingscoremodel"] = self.bindingScoreModel[key]


def load_printer(path: str | Path,
                 genome: genome.Genome):
    """
    Load a printer from adata file

    Parameters
    ----------
    path: str | Path
        path to the scprinter main h5ad file
    genome: Genome
        genome object. Must be the same as the one used to process the data
    """
    data = snap.read(path)
    assert data.uns['genome'] == f'{genome=}'.split('=')[0], "Process data with %s, but now loading with %s" %(data.uns['genome'],
                                                                                                               f'{genome=}'.split('=')[0])
    printer = PyPrinter(
        adata = data,
        adata_path = path,
        insertion_profile=None,
        genome=genome
    )
    # register close automatically.
    atexit.register(printer.close)
    return printer
