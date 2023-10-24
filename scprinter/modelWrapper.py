from .getCounts import *
from .getTFBS import *
from .getFootprint import *
from .getBias import *

class Footprinter():
    def __init__(self,
                 projectName,
                 refGenome,
                 projectMainDir,
                 regions,
                 barcodeGrouping,
                 groups,
                 model_dir,
                 ):
        self.projectName = projectName
        self.refGenome = refGenome
        self.projectMainDir = projectMainDir
        self.projectDataDir = os.path.join(self.projectMainDir, "data", self.projectName)
        if not os.path.exists(self.projectDataDir):
            os.makedirs(self.projectDataDir)

        self.regions = regions
        self.barcodeGrouping = barcodeGrouping
        self.groups = groups

        self.CountTensorPath = None
        self.FootPrintPath = None
        self.BindingScorePath = {'TF': None}

        self.dispersionModels = loadDispModel(os.path.join(model_dir, "shared", 'dispModel', 'models.h5'))
        self.TFBSModel = loadBindingScoreModel(os.path.join(model_dir, "TFBSPrediction", 'TFBS_model_py.h5'))
        self.BindingScoreModel = {'TF': self.TFBSModel}
        self.precomputed_bias_path = os.path.join(model_dir, "shared", '%sTn5Bias.h5' % self.refGenome)
        self.Tn5Bias=getPrecomputedBias(self.precomputed_bias_path,
                           self.regions,
                           savePath=None)
        self.avail_bindingmodels = ['TF']

    def computeCountTensor(
                        self,
                       pathToFrags,  # Path or list of paths to fragments file
                       maxFragLength=None,  # Fragment length upper limit
                       saveName='chunkedCountTensor.h5',
                       nrows=np.Inf,  # Max number of rows when reading from fragments file
                       chunkSize=2000,  # Chunk size for parallel processing of regions (I want to remove this arg)
                       fragchunkSize=1000000,
                       nCores=16,  # Number of cores to use
                       returnCombined=False,
                       # Whether to return the combined result for all chunks. Set it to False when data is too big,
                       plus_shift=4,
                       minus_shift=-5
                       ):
        self.CountTensorPath = os.path.join(self.projectDataDir, saveName)
        r =  computeCountTensor(
            pathToFrags,
            self.regions,
            self.barcodeGrouping,
            self.projectDataDir,
            saveName=saveName,
            maxFragLength=maxFragLength,  # Fragment length upper limit
            nrows=nrows,  # Max number of rows when reading from fragments file
            chunkSize=chunkSize,  # Chunk size for parallel processing of regions (I want to remove this arg)
            fragchunkSize=fragchunkSize,
            nCores=nCores,  # Number of cores to use
            returnCombined=returnCombined,
            # Whether to return the combined result for all chunks. Set it to False when data is too big,
            plus_shift=plus_shift,
            minus_shift=minus_shift
        )
        if returnCombined:
            return r

    def getFootprints(
                self,
                modes, # int or list of int. This is used for retrieving the correct dispersion model.
                footprintRadius=None, # Radius of the footprint region
                flankRadius=None, # Radius of the flanking region (not including the footprint region)
                nCores = 16, # Number of cores to use
                saveName = "chunkedFootprintResults.h5",
                verbose=True,
                returnCombined=None,
                 ):
        assert self.CountTensorPath is not None, "Run computeCountTensor first!"
        self.FootPrintPath = os.path.join(self.projectDataDir, saveName)
        CountTensorPath = self.CountTensorPath

        r = getFootprints(
                    CountTensorPath,
                    self.regions,
                    self.groups,
                    self.dispersionModels,
                    modes, # int or list of int. This is used for retrieving the correct dispersion model.
                    Tn5Bias=self.Tn5Bias,
                    footprintRadius=footprintRadius, # Radius of the footprint region
                    flankRadius=flankRadius, # Radius of the flanking region (not including the footprint region)
                    nCores = nCores, # Number of cores to use
                    saveDir = self.projectDataDir,
                    saveName = saveName,
                    verbose=verbose,
                    returnCombined=returnCombined,
                    append_mode=False, # When True, it means, the new results will be appended ore groups (mode 2)
            # / more regions (mode 1)), cannot have more reads, because that just changes the read count.
            # / more scales (mode 4), mode 3 is reserved for reads, but not supported here
                     )
        if returnCombined:
            return r


    def getBindingScore(self,
                        model = 'TF',
                        motifs=None,
                        contextRadius=100,  # well it's never ued in the R version... so... yeah
                        nCores=16,
                        saveName="chunkedBindingResults.h5",
                        returnCombined=None,
                        ):
        assert self.CountTensorPath is not None, "Run computeCountTensor first!"
        CountTensorPath = self.CountTensorPath
        self.BindingScorePath[model] = os.path.join(self.projectDataDir, saveName)
        r = getBindingScore(
            CountTensorPath,
            self.regions,
            self.groups,
            self.dispersionModels,
            self.BindingScoreModel[model],
            self.Tn5Bias,
            motifs=motifs,
            contextRadius=contextRadius,  # well it's never ued in the R version... so... yeah
            nCores=nCores,
            saveDir=self.projectDataDir,
            saveName=saveName,
            returnCombined=returnCombined,
            append_mode=False,  # When True, it means, the new results will be appended ore groups (mode 2)
            # / more regions (mode 1)), cannot have more reads, because that just changes the read count.
        )
        if returnCombined:
            return r
        