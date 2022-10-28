import mne
from mne.io import concatenate_raws

import numpy as np
import pandas as pd
import pywt
from statsmodels.tsa.filters.hp_filter import hpfilter
from numpy import fft
from scipy.fft import dct, idct
from scipy.signal import find_peaks
from scipy.cluster import hierarchy as spc
from scipy.stats import norm
from emd import sift
from random import shuffle

import copy
import pickle
import itertools
import os
from os.path import join
from warnings import warn



# General Modules
class WU_MEG_DP_lib:
    def __init__(self, printVer=1, printMessage=1):
        self.__version__ = '3.7.3'
        self.message = "Script by Dmitry Patashov"

        if printMessage:
            print(self.message)
        if printVer:
            print("WU Script Version: " + self.__version__ + "\n")

    def Data_Concatenation(self, DataFileNames, inDataDir):
        if DataFileNames.__len__() > 1:
            raw_file = []
            for name in DataFileNames:
                raw_file.append(mne.io.read_raw_fif(join(inDataDir, name), preload=True))
            raw = concatenate_raws(raw_file)
        else:
            raw = mne.io.read_raw_fif(join(inDataDir, DataFileNames[0]), preload=True)
        return raw

    def Remove_Outlier_Quartiles(self, SignalIn):
        if SignalIn is None:
            return None, np.array([])
        vecIn = SignalIn.copy()

        q1 = np.percentile(vecIn, 25)
        q3 = np.percentile(vecIn, 75)
        iqr = q3 - q1
        fence_low = q1 - 1.5 * iqr
        fence_high = q3 + 1.5 * iqr

        vecOut = vecIn[np.logical_and(vecIn > fence_low, vecIn < fence_high)]
        locs = np.where(np.logical_and(vecIn > fence_low, vecIn < fence_high) == False)[0]

        return vecOut, list(locs)

    def FilterRawData_BPF(self, raw, cutoff_l=0.1, cutoff_h=40):
        filt = raw.copy()
        filt = filt.filter(l_freq=cutoff_l, h_freq=cutoff_h)

        return filt

    def Save_Object(self, obj, fileName, folderPath):
        print("Saving data object to file.")

        if not os.path.exists(folderPath):
            os.mkdir(folderPath)

        filePath = join(folderPath, fileName)
        with open(filePath, 'wb') as outp:
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

        print("Done")

    def Load_Object(self, fileName, folderPath):
        filePath = join(folderPath, fileName)
        with open(filePath, 'rb') as inp:
            obj = pickle.load(inp)

            return obj


class DropEpochsManager:
    def __init__(self, epochs_condition):
        self.epochs_condition = epochs_condition
        if epochs_condition is None:
            self.totalNumberOfEpochs = 0
        else:
            self.totalNumberOfEpochs = len(epochs_condition.drop_log)
        self.numberOfFaultyEpochs = 0
        self.numberOfValidEpochs = self.totalNumberOfEpochs

        self.EpochsToDrop = []
        self.reasons = []

        self.droppedEpochs = {}
        self.unreliableEpochs = []
        self.reliableEpochs = list(range(self.totalNumberOfEpochs))

        self.PGA = {'train': None, 'test': None}

    def copy(self):
        inst = DropEpochsManager(self.epochs_condition.copy())
        inst.totalNumberOfEpochs = self.totalNumberOfEpochs
        inst.numberOfFaultyEpochs = self.numberOfFaultyEpochs
        inst.numberOfValidEpochs = self.numberOfValidEpochs
        inst.EpochsToDrop = self.EpochsToDrop
        inst.reasons = self.reasons
        inst.droppedEpochs = self.droppedEpochs
        inst.unreliableEpochs = self.unreliableEpochs
        inst.reliableEpochs = self.reliableEpochs
        inst.PGA = self.PGA

        return inst

    def BadEpochs(self, faulty_epochs, reason, method):
        if self.totalNumberOfEpochs == 0:
            return None

        if method == 'index':
            selection = list(self.epochs_condition.selection)
            epochsToDrop = []
            while faulty_epochs:
                epochsToDrop.append(selection[faulty_epochs.pop()])

        elif method == 'number':
            epochsToDrop = faulty_epochs

        else:
            raise Exception('Incorrect method. Acceptable methods are: "index" or "number".')

        self.EpochsToDrop.extend(epochsToDrop)
        self.reasons.extend([reason] * len(epochsToDrop))

        self.numberOfFaultyEpochs = len(self.unreliableEpochs) + len(set(self.EpochsToDrop))
        self.numberOfValidEpochs = self.totalNumberOfEpochs - self.numberOfFaultyEpochs

    def DropBadEpochs(self):
        if self.totalNumberOfEpochs == 0:
            return None

        while self.EpochsToDrop:
            ep = self.EpochsToDrop.pop()
            reason = self.reasons.pop()
            selection = list(self.epochs_condition.selection)
            drop_log = self.epochs_condition.drop_log

            if ep in self.unreliableEpochs:
                if reason not in self.droppedEpochs[ep]:
                    self.droppedEpochs[ep].append(reason)

                if ep in selection:
                    raise Exception('WU and MNE dropped epochs log missmatch!')

            elif ep not in selection:
                if drop_log[ep]:
                    self.droppedEpochs[ep] = [drop_log[ep]]
                else:
                    self.droppedEpochs[ep] = ['Unknown']

                self.unreliableEpochs.append(ep)
                self.reliableEpochs.remove(ep)

            else:
                self.droppedEpochs[ep] = [reason]
                self.unreliableEpochs.append(ep)
                self.reliableEpochs.remove(ep)

                self.epochs_condition.drop(selection.index(ep), reason=reason)


class Dump_File:
    def __init__(self, experiment):
        self.experiment = experiment
        self.contains = []
        self.data = {}

    def push(self, var, name):
        if name in self.contains:
            print('Variable already exists in Dump! \nOverwriting Data!')
            self.data[name] = var
        else:
            self.contains.append(name)
            self.data[name] = var

    def pop(self):
        if self.contains:
            name = self.contains.pop()
            var = self.data.pop(name)
            return var, name
        else:
            print('Dump is empty!')
            return None, None

    def get(self, name):
        if name in self.contains:
            self.contains.remove(name)
            var = self.data.pop(name)
            return var
        else:
            print('No such variable in the Dump!')
            return None


class IntersectedCombinationsGenerator:
    def __init__(self, numberOfElements=5, maxIntersection=2):
        self.numberOfElements = numberOfElements
        self.maxIntersection = maxIntersection

    def intersectionSize(self, C, V):
        d = np.zeros(C.shape[0])
        for k in range(C.shape[0]):
            xy = np.intersect1d(C[k], V)
            d[k] = xy.size

        return d

    def GenerateSet(self, picks):

        Cb = list(itertools.combinations(picks, self.numberOfElements))
        C = Cb.copy()

        sel = np.random.randint(len(C))
        V = C.pop(sel)

        group = []
        group.append(V)

        C = np.array(C)
        d = self.intersectionSize(C, V)

        C = C[d <= self.maxIntersection]

        while C.size != 0:
            V = C[0]
            C = np.delete(C, 0, axis=0)

            group.append(tuple(V))

            d = self.intersectionSize(C, V)
            C = C[d <= self.maxIntersection]

        return set(group)

    def uniquePicks(self, group):
        return list(set([item for t in group for item in t]))

    def BestSetOutOfX(self, picks, X):

        sortPicks = picks.copy()
        sortPicks.sort()

        optionGroup = []
        groupSize = []
        while len(optionGroup) < X:
            group = self.GenerateSet(sortPicks)

            usedPicks = self.uniquePicks(group)
            usedPicks.sort()
            if usedPicks == sortPicks:
                optionGroup.append(group)
                groupSize.append(len(group))

        bestGroup = list(optionGroup.pop(np.argmax(groupSize)))

        return bestGroup, optionGroup

    def CalculateSelectionOrder(self, group):
        scoreVec = []
        for k in range(len(group)):
            V = group[k]
            d = self.intersectionSize(np.array(group), V)
            scoreVec.append(np.sum(d))

        scoreVec = np.array(scoreVec)
        selectionOrder = np.argsort(scoreVec)

        return selectionOrder

    def refCombinationSets(self, combSelections, sizeListOpt, refSizeTe, refSizeTr, TeTrSize):

        for sampleSize in sizeListOpt:
            teS = int(np.floor(sampleSize / 2))

            picksList = list(range(sampleSize))
            shuffle(picksList)

            picksListTe = picksList[:teS]
            picksListTe.sort()
            picksListTr = picksList[teS:]
            picksListTr.sort()

            testGroup = []
            continueFlag = True
            while len(testGroup) < refSizeTe or continueFlag:
                testGroup = list(self.GenerateSet(picksListTe))

                usedPicks = self.uniquePicks(testGroup)
                usedPicks.sort()
                if usedPicks == picksListTe:
                    continueFlag = False
                else:
                    continueFlag = True

            selectionOrder = self.CalculateSelectionOrder(testGroup).astype(int)
            testGroup = [testGroup[selectionOrder[k]] for k in range(refSizeTe)]

            trainGroup = []
            if sampleSize - teS != teS or TeTrSize == 'max':
                continueFlag = True
                while len(trainGroup) < refSizeTr or continueFlag:
                    trainGroup = list(self.GenerateSet(picksListTr))

                    usedPicks = self.uniquePicks(trainGroup)
                    usedPicks.sort()
                    if usedPicks == picksListTr:
                        continueFlag = False
                    else:
                        continueFlag = True

                selectionOrder = self.CalculateSelectionOrder(trainGroup).astype(int)
                trainGroup = [trainGroup[selectionOrder[k]] for k in range(refSizeTr)]

            elif sampleSize - teS == teS and TeTrSize == 'same':
                zipIt = zip(picksListTe, picksListTr)
                translatePicks = dict(zipIt)
                for vec in testGroup:
                    trainGroup.append(tuple([translatePicks[val] for val in vec]))

            selectionGr = {'test': testGroup, 'train': trainGroup}
            combSelections[sampleSize] = selectionGr

        return combSelections

    def CreateCombinationsSet(self, sizeList, numberOfAttempts, TeTrSize='max'):

        sizeListOpt = list(set(sizeList))

        minSize = np.min(np.array(sizeListOpt))
        teS = int(np.floor(minSize / 2))

        picksList = list(range(minSize))
        shuffle(picksList)

        picksListTe = picksList[:teS]
        picksListTe.sort()
        picksListTr = picksList[teS:]
        picksListTr.sort()

        testGroup, _ = self.BestSetOutOfX(picksListTe, numberOfAttempts)

        refSizeTe = int(len(testGroup))

        trainGroup = []
        if minSize - teS != teS:
            continueFlag = True
            while len(trainGroup) < refSizeTe or continueFlag:
                trainGroup = list(self.GenerateSet(picksListTr))

                usedPicks = self.uniquePicks(trainGroup)
                usedPicks.sort()
                if usedPicks == picksListTr:
                    continueFlag = False
                else:
                    continueFlag = True

        else:
            zipIt = zip(picksListTe, picksListTr)
            translatePicks = dict(zipIt)
            for vec in testGroup:
                trainGroup.append(tuple([translatePicks[val] for val in vec]))

        if TeTrSize == 'same':
            if len(trainGroup) > refSizeTe:
                selectionOrder = self.CalculateSelectionOrder(trainGroup).astype(int)
                trainGroup = [trainGroup[selectionOrder[k]] for k in range(refSizeTe)]

            elif len(trainGroup) < refSizeTe:
                raise Exception('FATAL ERROR! Report to the developer.')

            refSizeTr = refSizeTe

        elif TeTrSize == 'max':
            refSizeTr = len(trainGroup)

        else:
            raise Exception('Unrecognized size option.')

        selectionGr = {'test': testGroup, 'train': trainGroup}
        combSelections = {minSize: selectionGr}

        sizeListOpt.remove(minSize)
        if sizeListOpt:
            combSelections = self.refCombinationSets(combSelections, sizeListOpt, refSizeTe, refSizeTr, TeTrSize)

        return combSelections



# Sumitomo Device Modules
class Sumitomo_Machine(WU_MEG_DP_lib):
    def __init__(self, Settings):
        super().__init__()

        self.DeviceNormalization = Settings["DeviceNormalization"]
        self.rectifyEvents = Settings["rectifyEvents"]
        self.numberOfChannels = Settings["numberOfChannels"]
        self.eventChannels = Settings["eventChannels"]
        self.eventAmplitudeTh = Settings["eventAmplitudeTh"]
        self.coilType = Settings["coilType"]
        self.rangeFactor = Settings["rangeFactor"]
        self.badChannels = Settings["badChannels"]

        self.amp = ["8.00292E-11", "7.85925E-11", "8.03307E-11", "-8.00268E-11", "-7.08582E-11",
                    "8.11455E-11", "-7.36791E-11", "8.2933E-11", "7.95981E-11", "7.9355E-11",
                    "8.2405E-11", "7.87805E-11", "-7.3652E-11", "-7.51707E-11", "-7.4057E-11",
                    "-7.49308E-11", "7.9748E-11", "8.12413E-11", "-6.11611E-07", "-7.42093E-11",
                    "7.94606E-11", "3.12562E-06", "-7.25894E-11", "-7.22576E-11", "8.07283E-11",
                    "7.99994E-11", "-7.41697E-11", "-7.40234E-11", "8.00739E-11", "8.05213E-11",
                    "-7.55817E-11", "7.97724E-11", "-7.16392E-11", "-7.56278E-11", "-7.45675E-11",
                    "8.14637E-11", "-7.33673E-11", "8.05111E-11", "8.02151E-11", "-7.35424E-11",
                    "8.11478E-11", "8.22796E-11", "-6.98919E-11", "8.04984E-11", "7.54834E-11",
                    "7.96823E-11", "7.93856E-11", "8.28757E-11", "8.16307E-11", "-7.30717E-11",
                    "7.77317E-11", "-7.55367E-11", "7.85881E-11", "8.11303E-11", "8.03006E-11",
                    "8.18859E-11", "-7.20612E-11", "8.20413E-11", "8.04149E-11", "-7.48125E-11",
                    "8.18346E-11", "-7.24964E-11", "7.92682E-11", "-7.17316E-11"]

    def dataNormalization(self, raw):
        print("Normalization ON: \n\t Performing data normalization.")

        # Correction coefficients section
        for k in range(self.numberOfChannels):
            raw.info["chs"][k]["cal"] = float(self.amp[k])

        # Normalization
        data0 = raw.get_data()
        data1 = data0.copy()
        for i in range(self.numberOfChannels):
            cal_factor = raw.info["chs"][i]["cal"]
            data1[i] = (data0[i] / cal_factor) * self.rangeFactor
        raw = mne.io.RawArray(data1, raw.info)

        return raw

    def deviceInit(self, raw):
        for ch in raw.info["chs"]:
            ch["coil_type"] = self.coilType

        return raw

    def eventChannelsRectification(self, raw):
        data2 = raw.get_data()

        subData = data2[self.eventChannels]
        for k in range(subData.shape[0]):
            time_index = np.where(subData[k] < self.eventAmplitudeTh)
            subData[k][time_index] = 0.0

            if self.rectifyEvents == 'yes':
                eSig = subData[k]
                time_index = np.array(np.where(eSig >= self.eventAmplitudeTh))

                if time_index.shape[1] > 0:
                    dfi1 = np.diff(time_index)
                    dfi2 = np.insert(dfi1, 0, 0, axis=1)

                    sub_index1 = np.array(np.where(dfi1[0] > 1))
                    sub_index1 = np.append(sub_index1, [[len(time_index[0]) - 1]], axis=1)

                    downPoints = time_index[0][sub_index1[0]]

                    sub_index2 = np.array(np.where(dfi2[0] > 1))
                    sub_index2 = np.insert(sub_index2, 0, 0, axis=1)

                    upPoints = time_index[0][sub_index2[0]]

                    segmentRanges = zip(upPoints, downPoints)
                    for (s, e) in segmentRanges:
                        eSig[s:e + 1] = np.round(np.median(eSig[s:e + 1]) * 2) / 2

                    subData[k] = eSig
        data2[self.eventChannels] = subData

        raw = mne.io.RawArray(data2, raw.info)
        return raw

    def FilterRawData_DT(self, raw, DT='auto', DT_param=None):
        filt = raw.copy()

        if DT == 'HP':
            if DT_param is None:
                lamb = 10 ** 9
            else:
                lamb = DT_param
            print("\nPerforming Hodrick-Prescott filter for detrending.\nlambda = " + str(lamb) + "\n")
            data = filt.get_data()

            data0 = data[0:64]
            data1 = data0.copy()
            for k in range(data0.shape[0]):
                cycle, trend = hpfilter(data0[k, :], lamb=lamb)
                data1[k, :] = cycle

            data[0:64] = data1
            filt = mne.io.RawArray(data, raw.info)

        elif DT == 'DCT' or DT == 'auto':
            if DT_param is None or DT == 'auto':
                cutoff_dt = 1
            else:
                cutoff_dt = DT_param
            print("\nPerforming DCT-based filter for detrending.\ncutoff time = " + str(cutoff_dt) + "s\n")
            data = filt.get_data()

            sf = filt.info['sfreq']
            sig_len = data.shape[1]
            cutoff_ind = np.int64(2 * sig_len / (sf * cutoff_dt))
            data0 = data[0:64]

            data1 = data0.copy()
            for k in range(data0.shape[0]):
                dct_coeffs = dct(data0[k, :], 1)
                dct_coeffs[0:cutoff_ind] = 0
                data1[k, :] = idct(dct_coeffs, 1)

            data[0:64] = data1
            filt = mne.io.RawArray(data, raw.info)

        else:
            warn("Detrending type was not defined properly, skipping procedure.")

        return filt


class Event_Log_Sumitomo:
    def __init__(self, headers):
        self.header = headers
        self.labels = []
        self.body = []

    def __inputEntry__(self, body):
        if len(body) != len(self.header):
            warn("The number of provided fields is incorrect! Performing default rectification!")

        if len(body) > len(self.header):
            body = body[:len(self.header)]
        elif len(body) < len(self.header):
            nBody = [""] * (len(self.header) - len(body))
            body.extend(nBody)

        return body

    def __inputInfo__(self, entries):
        if len(entries) != len(self.labels):
            warn("The number of provided elements is incorrect! Performing default rectification!")

        if len(entries) > len(self.labels):
            entries = entries[:len(self.labels)]
        elif len(entries) < len(self.labels):
            nEntries = [""] * (len(self.labels) - len(entries))
            entries.extend(nEntries)

        return entries

    def addEntry(self, label, body):
        body = self.__inputEntry__(body)

        self.labels.append(label)
        self.body.append(body)

    def updateEntry(self, key, label, body):
        body = self.__inputEntry__(body)

        if isinstance(key, str):
            key = self.labels.index(key)

        self.body[key] = body
        self.labels[key] = label

    def addInfo(self, header, entries):
        entries = self.__inputInfo__(entries)

        self.header.append(header)
        for k in range(len(entries)):
            self.body[k].append(entries[k])

    def updateInfo(self, key, header, entries):
        entries = self.__inputInfo__(entries)

        if isinstance(key, str):
            key = self.header.index(key)

        self.header[key] = header
        for k in range(len(entries)):
            self.body[k][key] = entries[k]

    def removeInfo(self, key):
        if isinstance(key, str):
            key = self.header.index(key)

        del self.header[key]
        for row in self.body:
            del row[key]

    def printLog(self):
        df = pd.DataFrame(self.body, self.labels, self.header)
        print(df)


class FrequencyAnalysisKit_Sumitomo:
    def __init__(self, SpectralBands="auto"):
        self.lib = WU_MEG_DP_lib()

        self.epochs_manager = None
        self.emdData = None
        self.Settings = None
        self.eventLog = None

        if SpectralBands == "auto":
            self.SpectralBands = {}
            # self.SpectralBands["Delta"] = (0.5, 4)
            self.SpectralBands["Delta"] = (None, 4)
            self.SpectralBands["Theta"] = (4, 8)
            self.SpectralBands["Alpha"] = (9, 13)
            self.SpectralBands["Beta"] = (14, 30)
            # self.SpectralBands["Gamma"] = (35, 45)
            self.SpectralBands["Gamma"] = (35, None)
        else:
            self.SpectralBands = SpectralBands
        self.BandTypes = list(self.SpectralBands.keys())

        self.SpectralBands["min"] = None
        self.SpectralBands["max"] = None

        self.frequencyBand = {}
        self.waveletBand = {}
        self.emdBand = {}

        self.waveletBandMatch = {}

        self.waveletBandRange = {}
        self.emdBandRange = {}

    def LoadDataset(self, Settings):
        dataDump = self.lib.Load_Object(Settings["DataFileNames"][0], Settings["inDataDir"])

        self.epochs_manager = dataDump.get("epochs")
        self.eventLog = self.epochs_manager['log']
        if Settings["loadEMD"] == 'yes':
            self.emdData = dataDump.get("emd")
        self.Settings = dataDump.get("settings")

        self.SpectralBands["max"] = self.Settings["new_freq"] / 2
        self.SpectralBands["min"] = 0
        # self.SpectralBands["min"] = 1 / (2 * (self.Settings["tmax"] - self.Settings["tmin"]))
        # lowFreq = [0, self.Settings["cutoff_l"], self.Settings["DT_param"]]
        # self.SpectralBands["min"] = np.max(np.array([x for x in lowFreq if x is not None]))

        self.lib = Sumitomo_Decoding_Long(self.Settings)

    def BandData2Evoked(self, dataDict, band):
        dataD = dataDict[band]

        evoked_conditions = {}
        for subSet in self.Settings["epochTypes"]:
            dataTensor = dataD[subSet]

            desInfo = self.epochs_manager[subSet].epochs_condition.info.copy()

            selectChannels = [ch for ch in range(self.lib.numberOfChannels) if ch not in self.lib.badChannels]
            selectedNames = [desInfo.ch_names[ch] for ch in selectChannels]
            desInfo.pick_channels(selectedNames)

            epochsArray = mne.EpochsArray(dataTensor, desInfo,
                                tmin=self.Settings["tmin"], baseline=self.Settings["baseline"])

            evoked_conditions[subSet] = epochsArray.average()

        return evoked_conditions

    def ExtractFrequencyBands(self):

        selectChannels = [ch for ch in range(self.lib.numberOfChannels) if ch not in self.lib.badChannels]
        for band in self.BandTypes:
            evTypes = {}
            for type in self.Settings["epochTypes"]:
                ep_condition = self.epochs_manager[type].epochs_condition.copy()
                ep_condition.filter(l_freq=self.SpectralBands[band][0], h_freq=self.SpectralBands[band][1])

                evTypes[type] = ep_condition.get_data()[:,selectChannels,:]
            self.frequencyBand[band] = evTypes

    def detectMainEnergyBorders(self, psdSig, energyTh):

        sig = np.abs(psdSig)
        searchVec = sig.copy()
        inds = []

        inds.append(np.argmax(searchVec))
        searchVec[inds[0]] = 0

        ener = 0
        while ener < energyTh:
            inds.append(np.argmax(searchVec))
            indRange = [np.min(np.array(inds)), np.max(np.array(inds))]
            searchVec[indRange[0]:indRange[1] + 1] = 0

            ener = np.sum(sig[indRange[0]:indRange[1] + 1]) / np.sum(sig)

        return indRange

    # def overlapCoeff(self, refVec, targetVec):
    #
    #     vecR = list(refVec)
    #     if vecR[0] == None:
    #         vecR[0] = self.SpectralBands["min"]
    #     if vecR[1] == None:
    #         vecR[1] = self.SpectralBands["max"]
    #
    #     # vecT = list(targetVec)
    #     # if vecT[0] == 0:
    #     #     vecT[0] = self.SpectralBands["min"]
    #
    #     # ovrlap = np.min(np.array([vecR[0], vecT[0]])) * np.min(np.array([vecR[1], vecT[1]]))
    #     # Score = ovrlap / (vecT[0] * vecT[1])
    #
    #     overlap = np.min(np.array([vecR[1], targetVec[1]])) - np.max(np.array([vecR[0], targetVec[0]]))
    #     Score = overlap / (targetVec[1] - targetVec[0])
    #
    #     return Score
    #
    # def detectMainFrequencyRange(self, signal, energyTh):
    #
    #     ac_func = np.correlate(signal, signal, mode='full')
    #     psd_func = fft.fftshift(fft.fft(ac_func))
    #     psd_func = psd_func[int(np.round(psd_func.size / 2)):]
    #     indRange = self.detectMainEnergyBorders(psd_func, energyTh)
    #     freqRange = np.linspace(0, self.SpectralBands["max"], psd_func.size)
    #
    #     return (freqRange[indRange[0]], freqRange[indRange[1]])

    def energyWithinBand(self, signal, band):

        ac_func = np.correlate(signal, signal, mode='full')
        psd_func = fft.fftshift(fft.fft(ac_func))
        psd_func = np.abs(psd_func[int(np.round(psd_func.size / 2)):])

        bandRange = list(self.SpectralBands[band])
        if bandRange[0] == None:
            bandRange[0] = self.SpectralBands["min"]
        if bandRange[1] == None:
            bandRange[1] = self.SpectralBands["max"]

        freqRange = np.linspace(0, self.SpectralBands["max"], psd_func.size)
        indRange = (np.argmin(np.abs(bandRange[0] - freqRange)), np.argmin(np.abs(bandRange[1] - freqRange)))

        score = np.sum(psd_func[indRange[0]:indRange[1]]) / np.sum(psd_func)

        return score

    def energyBandAnalysis(self, signal, band, energyTh):

        ac_func = np.correlate(signal, signal, mode='full')
        psd_func = fft.fftshift(fft.fft(ac_func))
        psd_func = np.abs(psd_func[int(np.round(psd_func.size / 2)):])

        bandRange = list(self.SpectralBands[band])
        if bandRange[0] == None:
            bandRange[0] = self.SpectralBands["min"]
        if bandRange[1] == None:
            bandRange[1] = self.SpectralBands["max"]

        freqSpace = np.linspace(0, self.SpectralBands["max"], psd_func.size)
        indRange = (np.argmin(np.abs(bandRange[0] - freqSpace)), np.argmin(np.abs(bandRange[1] - freqSpace)))

        BandMatchScore = np.sum(psd_func[indRange[0]:indRange[1]]) / np.sum(psd_func)

        indRange = self.detectMainEnergyBorders(psd_func, energyTh)
        freqRange = [freqSpace[indRange[0]], freqSpace[indRange[1]]]

        return BandMatchScore, freqRange

    def ExtractWaveletBand(self, level=6, mwType='db4', energyTh=0.90):

        selectChannels = [ch for ch in range(self.lib.numberOfChannels) if ch not in self.lib.badChannels]
        evTypes_tmp = {}
        for type in self.Settings["epochTypes"]:
            ep_condition = self.epochs_manager[type].epochs_condition.copy()
            data = ep_condition.get_data()[:, selectChannels, :]

            # samp = data[1,0,:]
            # coefficients_level = pywt.wavedec(samp, mwType, 'smooth', level=level)
            # return samp, coefficients_level

            initData = {}
            bandMatch = {}
            bandRange = {}
            for band in self.BandTypes:
                initData[band] = np.zeros(data.shape)
                bandMatch[band] = []
                bandRange[band] = []

            for ch in range(data.shape[1]):
                for ep in range(data.shape[0]):

                    coefficients_levels = pywt.wavedec(data[ep,ch,:], mwType, 'smooth', level=level)
                    for bandInd in range(len(self.BandTypes)):
                        band = self.BandTypes[bandInd]
                        coefficients = coefficients_levels.copy()
                        for k in range(len(coefficients_levels)):
                            if k != bandInd:
                                coefficients[k] = coefficients[k] * 0
                        reconstructedSignal = pywt.waverec(coefficients, mwType, 'smooth')
                        initData[band][ep, ch, :] = reconstructedSignal

                        # score = self.energyWithinBand(reconstructedSignal, band)
                        BandMatchScore, freqRange = self.energyBandAnalysis(reconstructedSignal, band, energyTh)
                        bandMatch[band].append(BandMatchScore)
                        bandRange[band].append(freqRange)

            evTypes_tmp[type] = initData

        for band in self.BandTypes:
            evTypes = {}
            for type in self.Settings["epochTypes"]:
                evTypes[type] = evTypes_tmp[type][band]
            self.waveletBand[band] = evTypes
            self.waveletBandMatch[band] = (np.mean(bandMatch[band]), np.std(bandMatch[band]))

            bandST = np.array(bandRange[band])[:, 0]
            bandED = np.array(bandRange[band])[:, 1]
            meanST = np.mean(bandST)
            meanED = np.mean(bandED)
            bandSTD = np.std(np.concatenate((bandST - meanST, bandED - meanED)))
            self.waveletBandRange[band] = [(meanST, meanED), bandSTD]

    def ExtractEMDBand(self, recalculateEMD='no', level=None):

        if recalculateEMD == 'no' and self.emdData == None:
            recalculateEMD = 'yes'
            level = self.Settings["cycles"]
        elif recalculateEMD == 'no':
            emdData = self.emdData

        if recalculateEMD == 'yes':
            if level == None:
                level = self.Settings["cycles"]

            emdSetting = {}
            emdSetting["cycles"] = level
            emdSetting["epochTypes"] = self.Settings["epochTypes"]
            # emd data structure: [epochType](epoch, channel, level, sample)
            emdData = self.lib.DatasetEMD(self.epochs_manager, emdSetting)

        samp = emdData["010"][1,0,:,:]
        return samp




class Sumitomo_Decoding_Long(Sumitomo_Machine):
    def __init__(self, Settings, loadSettings='no'):

        if loadSettings == 'yes':
            dataDump = self.Load_Object(Settings["DataFileNames"][0], Settings["inDataDir"])
            Settings = dataDump.get("settings")
        self.Settings = Settings

        super().__init__(Settings)

        self.idStartInd = Settings["idStartInd"]
        self.idEndInd = Settings["idEndInd"]
        self.dateStartInd = Settings["dateStartInd"]
        self.dateEndInd = Settings["dateEndInd"]
        self.fixEventSequence = Settings["fixEventSequence"]
        self.exposureTime = Settings["exposureTime"]
        self.exposure_err = Settings["exposure_err"]
        self.delayTime = Settings["delayTime"]
        self.errorMargin = Settings["errorMargin"]

    def fixSequence(self, data):
        stim2 = data[65]

        expErr = self.exposure_err * self.exposureTime

        df = np.diff(stim2)
        dfd = np.concatenate((df, np.array([0])))
        dfu = np.concatenate((np.array([0]), df))

        dfu[dfu < 0] = 0
        dfu[dfu > 0] = 1
        dfd[dfd > 0] = 0
        dfd[dfd < 0] = -1

        sdf = dfu + dfd

        upInd = list(np.where(sdf > 0))[0]
        dnInd = list(np.where(sdf < 0))[0]

        sections = list(zip(upInd, dnInd))
        secLen = dnInd - upInd

        sect300 = np.array(sections)[np.logical_and(secLen > self.exposureTime - expErr, secLen < self.exposureTime + expErr)]
        cent = np.round((sect300[:, 1] + sect300[:, 0]) / 2)
        initLocs = cent

        seqList = []
        for stVal in initLocs:
            seqList.append(range(np.int64(stVal), sect300[-1, 1], self.delayTime))

        binMap = np.zeros(stim2.shape)
        for st, ed in sect300:
            binMap[st:ed] = 1

        multVecList = []
        for k in range(seqList.__len__()):
            testVec = np.zeros(binMap.shape)
            testVec[seqList[k]] = 1
            multVecList.append(testVec * binMap)

        dfCountList = []
        for Vec in multVecList:
            dfBin = np.diff(Vec)
            dfCount = dfBin[dfBin > 0].size
            dfCountList.append(dfCount)

        ind60 = list(np.where(np.array(dfCountList) == 60))[0]
        ind40 = list(np.where(np.array(dfCountList) == 40))[0]
        ind20 = list(np.where(np.array(dfCountList) == 20))[0]

        indAll = []
        indAll.extend(ind60)
        indAll.extend(ind40)
        indAll.extend(ind20)

        fullSeq = np.zeros(stim2.shape)
        for ind in indAll:
            seqVec = multVecList[ind]
            fullSeq = fullSeq + seqVec
        fullSeq[fullSeq > 0] = 1

        nStim4 = np.zeros(stim2.shape)
        nStim2 = np.zeros(stim2.shape)
        for st, ed in sections:
            if np.sum(fullSeq[st:ed]):
                nStim2[st:ed] = stim2[st:ed]
            else:
                nStim4[st:ed] = stim2[st:ed]

        data[75] = nStim4
        data[65] = nStim2

        return data

    def recreateEvents(self, raw):

        data = raw.get_data()
        stim = data[64:67]

        stim[stim > 0] = 1

        stim_111 = stim[0] * stim[1] * stim[2]

        stim_011 = stim[0] * stim[1] - stim_111
        stim_101 = stim[0] * stim[2] - stim_111
        stim_110 = stim[1] * stim[2] - stim_111

        stim_100 = stim[2] - stim_111 - stim_101 - stim_110
        stim_010 = stim[1] - stim_111 - stim_110 - stim_011
        stim_001 = np.zeros(stim_111.shape)

        stim_all = stim_111 + stim_011 + stim_101 + stim_110 + stim_100 + stim_010

        rep = 1
        stim_template_a = np.zeros(self.delayTime + self.exposureTime)
        stim_template_a[0:self.exposureTime] = 1
        stim_template_a[self.delayTime + 1:] = 1
        stim_template_single = np.zeros(self.delayTime)
        stim_template_single[self.delayTime - self.exposureTime + 1:] = 1
        for k in range(rep - 1):
            stim_template_a = np.concatenate((stim_template_a, stim_template_single))

        stim_template = stim_template_a

        conv = np.convolve(stim_all, stim_template, mode='same') / np.sum(stim_template)

        height = 1 - self.exposure_err
        peaks, _ = find_peaks(conv, height=height)
        half_len_b = np.int(stim_template.size / 2 - self.exposureTime / 2)

        events_list = np.sort(np.concatenate((peaks - half_len_b, peaks + half_len_b)))

        events_locs = []
        flag = 1
        for k in range(events_list.size - 1):
            if events_list[k + 1] - events_list[k] < self.exposureTime:
                events_locs.append(events_list[k])
                flag = 0
            else:
                if flag:
                    events_locs.append(events_list[k])
                else:
                    flag = 1
        events_locs_arr = np.array(events_locs)

        events_dir = np.zeros(stim_all.shape)
        events_dir[events_locs_arr] = 1

        event = np.ones(self.exposureTime)
        stim_111_r = np.convolve(events_dir * stim_111, event, mode='same')
        stim_011_r = np.convolve(events_dir * stim_011, event, mode='same')
        stim_101_r = np.convolve(events_dir * stim_101, event, mode='same')
        stim_110_r = np.convolve(events_dir * stim_110, event, mode='same')
        stim_100_r = np.convolve(events_dir * stim_100, event, mode='same')

        delayErr = self.delayTime * self.errorMargin

        stimEv = peaks

        ds = np.diff(stimEv)
        ds_gr = np.abs(ds - self.delayTime)
        num = ds_gr[ds_gr < delayErr].size

        indList = []
        for k in range(peaks.size - 1):
            subPeaks = peaks[list(range(k)) + list(range(k + 1, peaks.size))]
            ds_s = np.diff(subPeaks)
            ds_gr_s = np.abs(ds_s - self.delayTime)
            subNum = ds_gr_s[ds_gr_s < delayErr].size
            if subNum >= num:
                indList.append(k)

        if indList:
            events_locs_arr2 = np.delete(events_locs_arr.copy(), np.array(indList))
        else:
            events_locs_arr2 = events_locs_arr.copy()
        events_dir2 = np.zeros(stim_all.shape)
        events_dir2[events_locs_arr2] = 1

        while np.sum(events_dir2 * stim_010) > 50 and rep < 10:
            warn("Voice keys fall in sync with event keys. Attempting desynchronization of higher order.")
            rep = rep + 1

            stim_template_a = np.zeros(self.delayTime + self.exposureTime)
            stim_template_a[0:self.exposureTime] = 1
            stim_template_a[self.delayTime+1:] = 1
            stim_template_single = np.zeros(self.delayTime)
            stim_template_single[self.delayTime - self.exposureTime+1:] = 1
            for k in range(rep - 1):
                stim_template_a = np.concatenate((stim_template_a, stim_template_single))

            stim_template = stim_template_a

            conv = np.convolve(stim_all, stim_template, mode='same') / np.sum(stim_template)

            height = 1 - self.exposure_err
            peaks, _ = find_peaks(conv, height=height)
            half_len_b = np.int(stim_template.size / 2 - self.exposureTime / 2)

            events_list = np.sort(np.concatenate((peaks - half_len_b, peaks + half_len_b)))

            events_locs = []
            flag = 1
            for k in range(events_list.size - 1):
                if events_list[k + 1] - events_list[k] < self.exposureTime:
                    events_locs.append(events_list[k])
                    flag = 0
                else:
                    if flag:
                        events_locs.append(events_list[k])
                    else:
                        flag = 1
            events_locs_arr = np.array(events_locs)

            events_dir2 = np.zeros(stim_all.shape)
            events_dir2[events_locs_arr] = 1

            stimEv = peaks

            ds = np.diff(stimEv)
            ds_gr = np.abs(ds - self.delayTime)
            num = ds_gr[ds_gr < delayErr].size

            indList = []
            for k in range(peaks.size - 1):
                subPeaks = peaks[list(range(k)) + list(range(k + 1, peaks.size))]
                ds_s = np.diff(subPeaks)
                ds_gr_s = np.abs(ds_s - self.delayTime)
                subNum = ds_gr_s[ds_gr_s < delayErr].size
                if subNum >= num:
                    indList.append(k)

            if indList:
                events_locs_arr2 = np.delete(events_locs_arr.copy(), np.array(indList))
            else:
                events_locs_arr2 = events_locs_arr.copy()
            events_dir2 = np.zeros(stim_all.shape)
            events_dir2[events_locs_arr2] = 1

        if rep >= 10:
            warn("All desynchronization attempts failed!")
        elif rep > 1:
            mes = "Desynchronization was successful at level: " + str(rep)
            warn(mes)

        stim_010_r = np.convolve(events_dir2 * stim_010, event, mode='same')
        stim_001_r = stim_001

        stim_mat = np.concatenate((stim_001_r.reshape(1, -1), stim_010_r.reshape(1, -1), stim_011_r.reshape(1, -1),
                                   stim_100_r.reshape(1, -1), stim_101_r.reshape(1, -1), stim_110_r.reshape(1, -1),
                                   stim_111_r.reshape(1, -1)), axis=0)
        info_stim = mne.create_info(
            ['STIM_001', 'STIM_010', 'STIM_011', 'STIM_100', 'STIM_101', 'STIM_110', 'STIM_111'], raw.info['sfreq'],
            ['stim', 'stim', 'stim', 'stim', 'stim', 'stim', 'stim'])
        stim_raw = mne.io.RawArray(stim_mat, info_stim)
        raw.add_channels([stim_raw], force_update_info=True)

        voice = np.zeros(stim_001_r.shape).reshape(1, -1)
        info_voice = mne.create_info(['Voice'], raw.info['sfreq'], ['stim'])
        stim_v = mne.io.RawArray(voice, info_voice)
        raw.add_channels([stim_v], force_update_info=True)

        return raw

    def PrepareRawData(self, inDataDir, DataFileNames, outDataDir):

        # Input Check
        fileNameGroup = []
        if not os.path.exists(inDataDir):
            raise Exception('Read directory does not exist.')
        elif DataFileNames != "all":
            subNum = DataFileNames[0][self.idStartInd:self.idEndInd]
            expDate = DataFileNames[0][self.dateStartInd:self.dateEndInd]
            for name in DataFileNames:
                if not os.path.exists(join(inDataDir, name)):
                    raise Exception('At least one read file does not exist.')
                else:
                    if not subNum == name[self.idStartInd:self.idEndInd]:
                        warn("Concatenated data is from different subjects!")
                    elif not expDate == name[self.dateStartInd:self.dateEndInd]:
                        warn("Concatenated data has different date stamps!")
            fileNameGroup.append(DataFileNames)
        else:
            fileList = []
            for file in os.listdir(inDataDir):
                if file.endswith(".fif"):
                    fileList.append(file)

            while fileList != []:
                fileName = fileList[0]
                singleGroup = []
                for k in range(fileList.__len__()):
                    if fileName[self.idStartInd:self.dateEndInd] == fileList[k][self.idStartInd:self.dateEndInd]:
                        singleGroup.append(fileList[k])

                fileNameGroup.append(singleGroup)
                for fName in fileNameGroup[-1]:
                    fileList.remove(fName)

        if not os.path.exists(outDataDir):
            os.mkdir(outDataDir)

        for DataFileNames in fileNameGroup:

            outFileName = DataFileNames[0][:-10] + "00_raw.fif"

            # Data Concatenation
            raw = self.Data_Concatenation(DataFileNames, inDataDir)

            # Data Normalization
            print("Normalization ON: \n\t Performing data normalization.")
            if self.DeviceNormalization == 'yes':
                raw = self.dataNormalization(raw)

            # Initialization
            raw = self.deviceInit(raw)
            print("Coil Type: ", raw.info["chs"][0]["coil_type"])

            # Event channels rectification
            raw = self.eventChannelsRectification(raw)

            # Recreate event channels
            raw = self.recreateEvents(raw)

            # Event Sequence Correction
            if self.fixEventSequence == 'yes':
                data3 = self.fixSequence(raw.get_data())
                raw = mne.io.RawArray(data3, raw.info)

            raw.save(join(outDataDir, outFileName), overwrite=True)

    def examineVoiseResponce(self, events_conditions, events_stim4, eventCycleLength=8000, timerStart=6000):

        events_stim4 = events_stim4[:, 0]

        events = events_conditions.copy()
        events = events[:, 0].reshape(-1, 1)

        eventsRange = np.concatenate((events, events + eventCycleLength), axis=1)
        analysisList = eventsRange.tolist()

        voiceResponse = []
        for k in range(eventsRange.shape[0]):
            st, ed = analysisList[k][0:2]
            idx = np.where((events_stim4 >= st) & (events_stim4 <= ed))
            responces = events_stim4[idx].tolist()
            if responces:
                analysisList[k].append(responces)
            else:
                analysisList[k].append([-1])

            distList = []
            for voiceLoc in analysisList[k][2]:
                if not voiceLoc == -1:
                    distList.append(voiceLoc - (analysisList[k][0] + timerStart))
                else:
                    distList.append(-0.1)

            analysisList[k].append(distList)
            voiceResponse.append(distList[0])
        voiceResponseInfo = np.array(voiceResponse)

        return analysisList, voiceResponseInfo

    def eventConditions(self, filt, events_stim1, events_stim2, events_stim3, min_voice_duration,
                        tmin, tmax, preload, baseline, grad):
        printFlag = 1
        printNum = None

        reject = dict(grad=grad)  # T / m (gradiometers)

        voiceResponsesInfo = {}
        events_stim_voice = mne.find_events(filt, stim_channel='Voice', min_duration=min_voice_duration)

        # Set 1: 0 0 1
        events_condition001 = np.asarray(
            [x for x in events_stim1 if (x[0] not in events_stim2[:, 0]) and (x[0] not in events_stim3[:, 0])])
        print("\n\nConditions: STIM3    STIM2   STIM1   Size")
        print('Set 1 - Empty (None)     :      0        0       1      ', events_condition001.shape[0])
        if events_condition001.__len__() > 0:
            epochs_condition001 = mne.Epochs(raw=filt, events=events_condition001, tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            if printFlag:
                prtMessage = epochs_condition001.info['sfreq']
                printFlag = 0
                printNum = '001'

            _, voiceResponsesInfo['001'] = self.examineVoiseResponce(events_condition001, events_stim_voice)
        else:
            epochs_condition001 = None
            voiceResponsesInfo['001'] = None

        # Set 2: 0 1 0
        events_condition010 = np.asarray(
            [x for x in events_stim2 if (x[0] not in events_stim1[:, 0]) and (x[0] not in events_stim3[:, 0])])
        print("\n\nConditions: STIM3    STIM2   STIM1   Size")
        print('Set 2 - Image1 (Sushi):      0        1       0      ', events_condition010.shape[0])
        if events_condition010.__len__() > 0:
            epochs_condition010 = mne.Epochs(raw=filt, events=events_condition010, tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            if printFlag:
                prtMessage = epochs_condition010.info['sfreq']
                printFlag = 0
                printNum = '010'

            _, voiceResponsesInfo['010'] = self.examineVoiseResponce(events_condition010, events_stim_voice)
        else:
            epochs_condition010 = None
            voiceResponsesInfo['010'] = None

        # Set 3: 0 1 1
        events_condition011 = np.asarray(
            [x for x in events_stim1 if (x[0] in events_stim2[:, 0]) and (x[0] not in events_stim3[:, 0])])
        print("\n\nConditions: STIM3    STIM2   STIM1   Size")
        print('Set 3 - Image2 (Gyoza):      0        1       1      ', events_condition011.shape[0])
        if events_condition011.__len__() > 0:
            epochs_condition011 = mne.Epochs(raw=filt, events=events_condition011, tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            if printFlag:
                prtMessage = epochs_condition011.info['sfreq']
                printFlag = 0
                printNum = '011'

            _, voiceResponsesInfo['011'] = self.examineVoiseResponce(events_condition011, events_stim_voice)
        else:
            epochs_condition011 = None
            voiceResponsesInfo['011'] = None

        # Set 4: 1 0 0
        events_condition100 = np.asarray(
            [x for x in events_stim3 if (x[0] not in events_stim1[:, 0]) and (x[0] not in events_stim2[:, 0])])
        print("\n\nConditions: STIM3    STIM2   STIM1   Size")
        print('Set 4 - Image3 (Cookie):      1        0       0      ', events_condition100.shape[0])
        if events_condition100.__len__() > 0:
            epochs_condition100 = mne.Epochs(raw=filt, events=events_condition100, tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            if printFlag:
                prtMessage = epochs_condition100.info['sfreq']
                printFlag = 0
                printNum = '100'

            _, voiceResponsesInfo['100'] = self.examineVoiseResponce(events_condition100, events_stim_voice)
        else:
            epochs_condition100 = None
            voiceResponsesInfo['100'] = None

        # Set 5: 1 0 1
        events_condition101 = np.asarray(
            [x for x in events_stim1 if (x[0] not in events_stim2[:, 0]) and (x[0] in events_stim3[:, 0])])
        print("\n\nConditions: STIM3    STIM2   STIM1   Size")
        print('Set 5 - Image5 (Knife):      1        0       1      ', events_condition101.shape[0])
        if events_condition101.__len__() > 0:
            epochs_condition101 = mne.Epochs(raw=filt, events=events_condition101, tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            if printFlag:
                prtMessage = epochs_condition101.info['sfreq']
                printFlag = 0
                printNum = '101'

            _, voiceResponsesInfo['101'] = self.examineVoiseResponce(events_condition101, events_stim_voice)
        else:
            epochs_condition101 = None
            voiceResponsesInfo['101'] = None

        # Set 6: 1 1 0
        events_condition110 = np.asarray(
            [x for x in events_stim2 if (x[0] not in events_stim1[:, 0]) and (x[0] in events_stim3[:, 0])])
        print("\n\nConditions: STIM3    STIM2   STIM1   Size")
        print('Set 6 - Image4 (Kushi):      1        1       0      ', events_condition110.shape[0])
        if events_condition110.__len__() > 0:
            epochs_condition110 = mne.Epochs(raw=filt, events=events_condition110, tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            if printFlag:
                prtMessage = epochs_condition110.info['sfreq']
                printFlag = 0
                printNum = '110'

            _, voiceResponsesInfo['110'] = self.examineVoiseResponce(events_condition110, events_stim_voice)
        else:
            epochs_condition110 = None
            voiceResponsesInfo['110'] = None

        # Set 7: 1 1 1
        events_condition111 = np.asarray(
            [x for x in events_stim1 if (x[0] in events_stim2[:, 0]) and (x[0] in events_stim3[:, 0])])
        print("\n\nConditions: STIM3    STIM2   STIM1   Size")
        print('Set 7 - Image6 (Pencil):      1        1       1      ', events_condition111.shape[0])
        if events_condition111.__len__() > 0:
            epochs_condition111 = mne.Epochs(raw=filt, events=events_condition111, tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            if printFlag:
                prtMessage = epochs_condition111.info['sfreq']
                printFlag = 0
                printNum = '111'

            _, voiceResponsesInfo['111'] = self.examineVoiseResponce(events_condition111, events_stim_voice)
        else:
            epochs_condition111 = None
            voiceResponsesInfo['111'] = None

        events_conditions = {'001': events_condition001, '010': events_condition010, '011': events_condition011,
                             '100': events_condition100, '101': events_condition101, '110': events_condition110,
                             '111': events_condition111, 'log': None}

        epochs_conditions = {'001': epochs_condition001, '010': epochs_condition010, '011': epochs_condition011,
                             '100': epochs_condition100, '101': epochs_condition101, '110': epochs_condition110,
                             '111': epochs_condition111, 'log': None}

        return events_conditions, epochs_conditions, voiceResponsesInfo, printNum, prtMessage

    def BadEpochsGroup(self, epochs_manager, faulty_epochs, eventLogs, reason, method):
        epochs_manager['001'].BadEpochs(list(faulty_epochs['001']), reason=reason, method=method)

        epochs_manager['010'].BadEpochs(list(faulty_epochs['010']), reason=reason, method=method)

        epochs_manager['011'].BadEpochs(list(faulty_epochs['011']), reason=reason, method=method)

        epochs_manager['100'].BadEpochs(list(faulty_epochs['100']), reason=reason, method=method)

        epochs_manager['101'].BadEpochs(list(faulty_epochs['101']), reason=reason, method=method)

        epochs_manager['110'].BadEpochs(list(faulty_epochs['110']), reason=reason, method=method)

        epochs_manager['111'].BadEpochs(list(faulty_epochs['111']), reason=reason, method=method)


        eventLogs.updateInfo("Using", "Using", [str(epochs_manager['001'].numberOfValidEpochs),
                                                str(epochs_manager['010'].numberOfValidEpochs),
                                                str(epochs_manager['011'].numberOfValidEpochs),
                                                str(epochs_manager['100'].numberOfValidEpochs),
                                                str(epochs_manager['101'].numberOfValidEpochs),
                                                str(epochs_manager['110'].numberOfValidEpochs),
                                                str(epochs_manager['111'].numberOfValidEpochs)])

    def detectFaultyResponseEpochs(self, events_conditions, voiceResponsesInfo, epochs_manager):
        events_condition001 = events_conditions['001']
        events_condition010 = events_conditions['010']
        events_condition011 = events_conditions['011']
        events_condition100 = events_conditions['100']
        events_condition101 = events_conditions['101']
        events_condition110 = events_conditions['110']
        events_condition111 = events_conditions['111']

        event_log = Event_Log_Sumitomo(["STIM3","STIM2", "STIM1", "MNE-All", "Using"])
        faulty_epochs = {}
        reliable_epochs = {}
        if voiceResponsesInfo['001'] is not None:
            faulty_epochs['001'] = np.where(voiceResponsesInfo['001'] <= 0)[0]
            reliable_epochs['001'] = np.where(voiceResponsesInfo['001'] > 0)[0]
            event_log.addEntry("Set 1 - Empty (None)", ["0", "0", "1",
                                                        str(events_condition001.shape[0]),
                                                        str(reliable_epochs['001'].shape[0])])
        else:
            faulty_epochs['001'] = np.array([])
            event_log.addEntry("Set 1 - Empty (None)", ["0", "0", "1",
                                                        str(events_condition001.shape[0]), "0"])

        if voiceResponsesInfo['010'] is not None:
            faulty_epochs['010'] = np.where(voiceResponsesInfo['010'] <= 0)[0]
            reliable_epochs['010'] = np.where(voiceResponsesInfo['010'] > 0)[0]
            event_log.addEntry("Set 2 - Image1 (Sushi)", ["0", "1", "0",
                                                        str(events_condition010.shape[0]),
                                                        str(reliable_epochs['010'].shape[0])])
        else:
            faulty_epochs['010'] = np.array([])
            event_log.addEntry("Set 2 - Image1 (Sushi)", ["0", "1", "0",
                                                        str(events_condition010.shape[0]), "0"])

        if voiceResponsesInfo['011'] is not None:
            faulty_epochs['011'] = np.where(voiceResponsesInfo['011'] <= 0)[0]
            reliable_epochs['011'] = np.where(voiceResponsesInfo['011'] > 0)[0]
            event_log.addEntry("Set 3 - Image2 (Gyoza)", ["0", "1", "1",
                                                          str(events_condition011.shape[0]),
                                                          str(reliable_epochs['011'].shape[0])])

        else:
            faulty_epochs['011'] = np.array([])
            event_log.addEntry("Set 3 - Image2 (Gyoza)", ["0", "1", "1",
                                                          str(events_condition011.shape[0]), "0"])

        if voiceResponsesInfo['100'] is not None:
            faulty_epochs['100'] = np.where(voiceResponsesInfo['100'] <= 0)[0]
            reliable_epochs['100'] = np.where(voiceResponsesInfo['100'] > 0)[0]
            event_log.addEntry("Set 4 - Image3 (Cookie)", ["1", "0", "0",
                                                          str(events_condition100.shape[0]),
                                                          str(reliable_epochs['100'].shape[0])])
        else:
            faulty_epochs['100'] = np.array([])
            event_log.addEntry("Set 4 - Image3 (Cookie)", ["1", "0", "0",
                                                           str(events_condition100.shape[0]), "0"])

        if voiceResponsesInfo['101'] is not None:
            faulty_epochs['101'] = np.where(voiceResponsesInfo['101'] <= 0)[0]
            reliable_epochs['101'] = np.where(voiceResponsesInfo['101'] > 0)[0]
            event_log.addEntry("Set 5 - Image5 (Knife)", ["1", "0", "1",
                                                           str(events_condition101.shape[0]),
                                                           str(reliable_epochs['101'].shape[0])])
        else:
            faulty_epochs['101'] = np.array([])
            event_log.addEntry("Set 5 - Image5 (Knife)", ["1", "0", "1",
                                                          str(events_condition101.shape[0]), "0"])

        if voiceResponsesInfo['110'] is not None:
            faulty_epochs['110'] = np.where(voiceResponsesInfo['110'] <= 0)[0]
            reliable_epochs['110'] = np.where(voiceResponsesInfo['110'] > 0)[0]
            event_log.addEntry("Set 6 - Image4 (Kushi)", ["1", "1", "0",
                                                          str(events_condition110.shape[0]),
                                                          str(reliable_epochs['110'].shape[0])])
        else:
            faulty_epochs['110'] = np.array([])
            event_log.addEntry("Set 6 - Image4 (Kushi)", ["1", "1", "0",
                                                          str(events_condition110.shape[0]), "0"])

        if voiceResponsesInfo['111'] is not None:
            faulty_epochs['111'] = np.where(voiceResponsesInfo['111'] <= 0)[0]
            reliable_epochs['111'] = np.where(voiceResponsesInfo['111'] > 0)[0]
            event_log.addEntry("Set 7 - Image6 (Pencil)", ["1", "1", "1",
                                                          str(events_condition111.shape[0]),
                                                          str(reliable_epochs['111'].shape[0])])
        else:
            faulty_epochs['111'] = np.array([])
            event_log.addEntry("Set 7 - Image6 (Pencil)", ["1", "1", "1",
                                                          str(events_condition111.shape[0]), "0"])

        self.BadEpochsGroup(epochs_manager, faulty_epochs, event_log, reason='Response Issue', method='number')

        return event_log

    def DropMarkedEpochs(self, epochs_manager):
        print('001')
        epochs_manager['001'].DropBadEpochs()
        print('010')
        epochs_manager['010'].DropBadEpochs()
        print('011')
        epochs_manager['011'].DropBadEpochs()
        print('100')
        epochs_manager['100'].DropBadEpochs()
        print('101')
        epochs_manager['101'].DropBadEpochs()
        print('110')
        epochs_manager['110'].DropBadEpochs()
        print('111')
        epochs_manager['111'].DropBadEpochs()

    def DropVoceResponses(self, voiceResponsesInfo, epochs_manager):
        voiceResponseTimes = copy.deepcopy(voiceResponsesInfo)

        if voiceResponseTimes['001'] is not None:
            voiceResponseTimes['001'] = voiceResponseTimes['001'][epochs_manager['001'].reliableEpochs]

        if voiceResponseTimes['010'] is not None:
            voiceResponseTimes['010'] = voiceResponseTimes['010'][epochs_manager['010'].reliableEpochs]

        if voiceResponseTimes['011'] is not None:
            voiceResponseTimes['011'] = voiceResponseTimes['011'][epochs_manager['011'].reliableEpochs]

        if voiceResponseTimes['100'] is not None:
            voiceResponseTimes['100'] = voiceResponseTimes['100'][epochs_manager['100'].reliableEpochs]

        if voiceResponseTimes['101'] is not None:
            voiceResponseTimes['101'] = voiceResponseTimes['101'][epochs_manager['101'].reliableEpochs]

        if voiceResponseTimes['110'] is not None:
            voiceResponseTimes['110'] = voiceResponseTimes['110'][epochs_manager['110'].reliableEpochs]

        if voiceResponseTimes['111'] is not None:
            voiceResponseTimes['111'] = voiceResponseTimes['111'][epochs_manager['111'].reliableEpochs]

        return voiceResponseTimes

    def DetectVoceOutliers(self, voiceResponseTimes):
        locs = {}

        _, locs['001'] = self.Remove_Outlier_Quartiles(voiceResponseTimes['001'])
        _, locs['010'] = self.Remove_Outlier_Quartiles(voiceResponseTimes['010'])
        _, locs['011'] = self.Remove_Outlier_Quartiles(voiceResponseTimes['011'])
        _, locs['100'] = self.Remove_Outlier_Quartiles(voiceResponseTimes['100'])
        _, locs['101'] = self.Remove_Outlier_Quartiles(voiceResponseTimes['101'])
        _, locs['110'] = self.Remove_Outlier_Quartiles(voiceResponseTimes['110'])
        _, locs['111'] = self.Remove_Outlier_Quartiles(voiceResponseTimes['111'])

        return locs

    def ResampleEpochsGroup(self, epochs_manager, new_freq):
        if epochs_manager['001'].epochs_condition is not None:
            epochs_manager['001'].epochs_condition.resample(new_freq)

        if epochs_manager['010'].epochs_condition is not None:
            epochs_manager['010'].epochs_condition.resample(new_freq)

        if epochs_manager['011'].epochs_condition is not None:
            epochs_manager['011'].epochs_condition.resample(new_freq)

        if epochs_manager['100'].epochs_condition is not None:
            epochs_manager['100'].epochs_condition.resample(new_freq)

        if epochs_manager['101'].epochs_condition is not None:
            epochs_manager['101'].epochs_condition.resample(new_freq)

        if epochs_manager['110'].epochs_condition is not None:
            epochs_manager['110'].epochs_condition.resample(new_freq)

        if epochs_manager['111'].epochs_condition is not None:
            epochs_manager['111'].epochs_condition.resample(new_freq)

    def reconfigureEvents(self, filt, delayTime, errorMargin, min_event_duration, count):

        events_stim1 = mne.find_events(filt, stim_channel='STIM1', min_duration=min_event_duration)
        events_stim2 = mne.find_events(filt, stim_channel='STIM2', min_duration=min_event_duration)
        events_stim3 = mne.find_events(filt, stim_channel='STIM3', min_duration=min_event_duration)

        delayT = delayTime
        delayErr = errorMargin * delayT

        for k in range(events_stim1.shape[0]):
            x = events_stim1[k]
            dif12 = np.abs(x[0] - events_stim2[:, 0])
            dif13 = np.abs(x[0] - events_stim3[:, 0])
            if np.min(dif12) < delayErr and np.min(dif13) < delayErr:
                ind2 = np.argmin(dif12)
                ind3 = np.argmin(dif13)
                opt = np.array([x[0], events_stim2[ind2, 0], events_stim3[ind3, 0]])
                alignVal = np.min(opt)

                if events_stim1[k, 0] != alignVal or\
                        events_stim2[ind2, 0] != alignVal or \
                        events_stim3[ind3, 0] != alignVal:
                    count['111'] = count['111'] + 1

                events_stim1[k, 0] = alignVal
                events_stim2[ind2, 0] = alignVal
                events_stim3[ind3, 0] = alignVal

            elif np.min(dif12) < delayErr:
                ind = np.argmin(dif12)
                opt = np.array([x[0], events_stim2[ind, 0]])
                alignVal = np.min(opt)

                if events_stim1[k, 0] != alignVal or \
                        events_stim2[ind, 0] != alignVal:
                    count['011'] = count['011'] + 1

                events_stim1[k, 0] = alignVal
                events_stim2[ind, 0] = alignVal

            elif np.min(dif13) < delayErr:
                ind = np.argmin(dif13)
                opt = np.array([x[0], events_stim3[ind, 0]])
                alignVal = np.min(opt)

                if events_stim1[k, 0] != alignVal or \
                        events_stim3[ind, 0] != alignVal:
                    count['101'] = count['101'] + 1

                events_stim1[k, 0] = alignVal
                events_stim3[ind, 0] = alignVal

        for k in range(events_stim2.shape[0]):
            x = events_stim2[k]
            dif23 = np.abs(x[0] - events_stim3[:, 0])
            if np.min(dif23) < delayErr:
                ind = np.argmin(dif23)
                opt = np.array([x[0], events_stim3[ind, 0]])
                alignVal = np.min(opt)

                if events_stim2[k, 0] != alignVal or events_stim3[ind, 0] != alignVal:
                    count['110'] = count['110'] + 1

                events_stim2[k, 0] = alignVal
                events_stim3[ind, 0] = alignVal

        stimEv = np.concatenate((events_stim1[:, 0], events_stim2[:, 0], events_stim3[:, 0]))
        stimEv = np.unique(stimEv)
        stimEv.sort()

        ds = np.diff(stimEv)

        ds_gr = np.abs(ds - delayT)
        ds_gr[ds_gr < delayErr] = 0

        pos = stimEv[np.concatenate((ds_gr, np.array([0]))) > 0]

        rmPos = []
        for k in range(ds_gr.shape[0] - 1):
            if ds_gr[k] and ds_gr[k + 1]:
                if ds_gr[k] - delayT < delayErr:
                    m = k + 1
                    splitVal = np.abs(np.sum(ds_gr[k:m]) - delayT)
                    while splitVal < delayErr:
                        m = m + 1
                        splitVal = np.abs(np.sum(ds_gr[k:m]) - delayT)

                    rmPos.append(range(k, m))

        if rmPos:
            remPosVec = np.concatenate(rmPos)

            remEvents = stimEv[remPosVec + 1]
        else:
            remEvents = np.array([])

        events_stim1_rm = np.asarray([x for x in events_stim1 if x[0] in remEvents])
        events_stim2_rm = np.asarray([x for x in events_stim2 if x[0] in remEvents])
        events_stim3_rm = np.asarray([x for x in events_stim3 if x[0] in remEvents])

        events_stim1 = np.asarray([x for x in events_stim1 if x[0] not in remEvents])
        events_stim2 = np.asarray([x for x in events_stim2 if x[0] not in remEvents])
        events_stim3 = np.asarray([x for x in events_stim3 if x[0] not in remEvents])

        return events_stim1, events_stim2, events_stim3, events_stim1_rm, events_stim2_rm, events_stim3_rm, count

    def eventConditions_recreated(self, filt, min_event_duration, min_voice_duration,
                                  tmin, tmax, preload, baseline, grad):
        printFlag = 1
        printNum = None

        reject = dict(grad=grad)  # T / m (gradiometers)

        voiceResponsesInfo = {}
        events_stim_voice = mne.find_events(filt, stim_channel='Voice', min_duration=min_voice_duration)

        # Set 1: 0 0 1
        events_condition001 = mne.find_events(filt, stim_channel='STIM_001', min_duration=min_event_duration)
        print("\n\nConditions: STIM3    STIM2   STIM1   Size")
        print('Set 1 - Empty (None)     :      0        0       1      ', events_condition001.shape[0])
        if events_condition001.__len__() > 0:
            epochs_condition001 = mne.Epochs(raw=filt, events=events_condition001, tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            if printFlag:
                prtMessage = epochs_condition001.info['sfreq']
                printFlag = 0
                printNum = '001'

            _, voiceResponsesInfo['001'] = self.examineVoiseResponce(events_condition001, events_stim_voice)
        else:
            epochs_condition001 = None
            voiceResponsesInfo['001'] = None

        # Set 2: 0 1 0
        events_condition010 = mne.find_events(filt, stim_channel='STIM_010', min_duration=min_event_duration)
        print("\n\nConditions: STIM3    STIM2   STIM1   Size")
        print('Set 2 - Image1 (Sushi):      0        1       0      ', events_condition010.shape[0])
        if events_condition010.__len__() > 0:
            epochs_condition010 = mne.Epochs(raw=filt, events=events_condition010, tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            if printFlag:
                prtMessage = epochs_condition010.info['sfreq']
                printFlag = 0
                printNum = '010'

            _, voiceResponsesInfo['010'] = self.examineVoiseResponce(events_condition010, events_stim_voice)
        else:
            epochs_condition010 = None
            voiceResponsesInfo['010'] = None

        # Set 3: 0 1 1
        events_condition011 = mne.find_events(filt, stim_channel='STIM_011', min_duration=min_event_duration)
        print("\n\nConditions: STIM3    STIM2   STIM1   Size")
        print('Set 3 - Image2 (Gyoza):      0        1       1      ', events_condition011.shape[0])
        if events_condition011.__len__() > 0:
            epochs_condition011 = mne.Epochs(raw=filt, events=events_condition011, tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            if printFlag:
                prtMessage = epochs_condition011.info['sfreq']
                printFlag = 0
                printNum = '011'

            _, voiceResponsesInfo['011'] = self.examineVoiseResponce(events_condition011, events_stim_voice)
        else:
            epochs_condition011 = None
            voiceResponsesInfo['011'] = None

        # Set 4: 1 0 0
        events_condition100 = mne.find_events(filt, stim_channel='STIM_100', min_duration=min_event_duration)
        print("\n\nConditions: STIM3    STIM2   STIM1   Size")
        print('Set 4 - Image3 (Cookie):      1        0       0      ', events_condition100.shape[0])
        if events_condition100.__len__() > 0:
            epochs_condition100 = mne.Epochs(raw=filt, events=events_condition100, tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            if printFlag:
                prtMessage = epochs_condition100.info['sfreq']
                printFlag = 0
                printNum = '100'

            _, voiceResponsesInfo['100'] = self.examineVoiseResponce(events_condition100, events_stim_voice)
        else:
            epochs_condition100 = None
            voiceResponsesInfo['100'] = None

        # Set 5: 1 0 1
        events_condition101 = mne.find_events(filt, stim_channel='STIM_101', min_duration=min_event_duration)
        print("\n\nConditions: STIM3    STIM2   STIM1   Size")
        print('Set 5 - Image5 (Knife):      1        0       1      ', events_condition101.shape[0])
        if events_condition101.__len__() > 0:
            epochs_condition101 = mne.Epochs(raw=filt, events=events_condition101, tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            if printFlag:
                prtMessage = epochs_condition101.info['sfreq']
                printFlag = 0
                printNum = '101'

            _, voiceResponsesInfo['101'] = self.examineVoiseResponce(events_condition101, events_stim_voice)
        else:
            epochs_condition101 = None
            voiceResponsesInfo['101'] = None

        # Set 6: 1 1 0
        events_condition110 = mne.find_events(filt, stim_channel='STIM_110', min_duration=min_event_duration)
        print("\n\nConditions: STIM3    STIM2   STIM1   Size")
        print('Set 6 - Image4 (Kushi):      1        1       0      ', events_condition110.shape[0])
        if events_condition110.__len__() > 0:
            epochs_condition110 = mne.Epochs(raw=filt, events=events_condition110, tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            if printFlag:
                prtMessage = epochs_condition110.info['sfreq']
                printFlag = 0
                printNum = '110'

            _, voiceResponsesInfo['110'] = self.examineVoiseResponce(events_condition110, events_stim_voice)
        else:
            epochs_condition110 = None
            voiceResponsesInfo['110'] = None

        # Set 7: 1 1 1
        events_condition111 = mne.find_events(filt, stim_channel='STIM_111', min_duration=min_event_duration)
        print("\n\nConditions: STIM3    STIM2   STIM1   Size")
        print('Set 7 - Image6 (Pencil):      1        1       1      ', events_condition111.shape[0])
        if events_condition111.__len__() > 0:
            epochs_condition111 = mne.Epochs(raw=filt, events=events_condition111, tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            if printFlag:
                prtMessage = epochs_condition111.info['sfreq']
                printFlag = 0
                printNum = '111'

            _, voiceResponsesInfo['111'] = self.examineVoiseResponce(events_condition111, events_stim_voice)
        else:
            epochs_condition111 = None
            voiceResponsesInfo['111'] = None

        events_conditions = {'001': events_condition001, '010': events_condition010, '011': events_condition011,
                             '100': events_condition100, '101': events_condition101, '110': events_condition110,
                             '111': events_condition111, 'log': None}

        epochs_conditions = {'001': epochs_condition001, '010': epochs_condition010, '011': epochs_condition011,
                             '100': epochs_condition100, '101': epochs_condition101, '110': epochs_condition110,
                             '111': epochs_condition111, 'log': None}

        return events_conditions, epochs_conditions, voiceResponsesInfo, printNum, prtMessage

    def EpockingProcedures(self, raw, resemple='yes', new_freq=500, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0),
                           min_event_duration=0.1, min_voice_duration=0.01, reconfigureEvents='yes', errorMargin=0.01,
                           grad=7000e-13, preload=True, dropFaultyResponseEpochs='no', voiceOutliers='no',
                           useRecreatedEvents='yes'):

        filt = raw.copy()
        count = {'001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 0}

        if reconfigureEvents == 'yes':
            events_stim1, events_stim2, events_stim3, events_stim1_rm, events_stim2_rm, events_stim3_rm, count = \
                self.reconfigureEvents(filt, self.delayTime, errorMargin, min_event_duration, count)

            if events_stim1_rm.size != 0 or events_stim2_rm.size != 0 or events_stim3_rm.size != 0:
                if events_stim1_rm.size == 0:
                    events_stim1_rm = np.array([[]])
                if events_stim2_rm.size == 0:
                    events_stim2_rm = np.array([[]])
                if events_stim3_rm.size == 0:
                    events_stim3_rm = np.array([[]])

                count['001'] = count['001'] + events_stim1_rm.shape[0]
                count['010'] = count['010'] + events_stim2_rm.shape[0]
                count['100'] = count['100'] + events_stim3_rm.shape[0]
        else:
            events_stim1 = mne.find_events(filt, stim_channel='STIM1', min_duration=min_event_duration)
            events_stim2 = mne.find_events(filt, stim_channel='STIM2', min_duration=min_event_duration)
            events_stim3 = mne.find_events(filt, stim_channel='STIM3', min_duration=min_event_duration)


        # Conditions: STIM1 STIM2 STIM3
        events_conditions_b, epochs_conditions, voiceResponsesInfo, printNum, prtMessage = \
            self.eventConditions(filt, events_stim1, events_stim2, events_stim3, min_voice_duration,
                                 tmin, tmax, preload, baseline, grad)

        if useRecreatedEvents == 'yes':
            events_conditions_r, epochs_conditions, voiceResponsesInfo, printNum, prtMessage = \
                self.eventConditions_recreated(filt, min_event_duration, min_voice_duration,
                                               tmin, tmax, preload, baseline, grad)
            events_conditions_v = events_conditions_r
        else:
            events_conditions_r, _, _, _, _ = self.eventConditions_recreated(filt,
                                                                             min_event_duration, min_voice_duration,
                                                                             tmin, tmax, preload, baseline, grad)
            events_conditions_v = events_conditions_b

        count['001'] = count['001'] + events_conditions_b['001'].shape[0]
        count['010'] = count['010'] + events_conditions_b['010'].shape[0]
        count['011'] = count['011'] + events_conditions_b['011'].shape[0]
        count['100'] = count['100'] + events_conditions_b['100'].shape[0]
        count['101'] = count['101'] + events_conditions_b['101'].shape[0]
        count['110'] = count['110'] + events_conditions_b['110'].shape[0]
        count['111'] = count['111'] + events_conditions_b['111'].shape[0]

        epochs_manager = {}
        epochs_manager['001'] = DropEpochsManager(epochs_conditions['001'])
        epochs_manager['010'] = DropEpochsManager(epochs_conditions['010'])
        epochs_manager['011'] = DropEpochsManager(epochs_conditions['011'])
        epochs_manager['100'] = DropEpochsManager(epochs_conditions['100'])
        epochs_manager['101'] = DropEpochsManager(epochs_conditions['101'])
        epochs_manager['110'] = DropEpochsManager(epochs_conditions['110'])
        epochs_manager['111'] = DropEpochsManager(epochs_conditions['111'])

        if dropFaultyResponseEpochs == 'yes':
            eventLog = self.detectFaultyResponseEpochs(events_conditions_b, voiceResponsesInfo, epochs_manager)
            if voiceOutliers == 'yes':
                locs = self.DetectVoceOutliers(voiceResponsesInfo)
                self.BadEpochsGroup(epochs_manager, locs, eventLog, 'Voice Outlier', 'index')
                voiceResponseTimes = self.DropVoceResponses(voiceResponsesInfo, epochs_manager)
                voiceResponsesInfo['times'] = voiceResponseTimes
        else:
            eventLog = Event_Log_Sumitomo(["STIM3", "STIM2", "STIM1", "MNE-All", "MNE-Valid", "WU-All", "Using"])
            eventLog.addEntry("Set 1 - Empty (None)",
                              ["0", "0", "1",
                               str(count['001']),
                               str(events_conditions_b['001'].shape[0]),
                               str(events_conditions_r['001'].shape[0]),
                               str(events_conditions_v['001'].shape[0])])
            eventLog.addEntry("Set 2 - Image1 (Sushi)",
                              ["0", "1", "0",
                               str(count['010']),
                               str(events_conditions_b['010'].shape[0]),
                               str(events_conditions_r['010'].shape[0]),
                               str(events_conditions_v['010'].shape[0])])
            eventLog.addEntry("Set 3 - Image2 (Gyoza)",
                              ["0", "1", "1",
                               str(count['011']),
                               str(events_conditions_b['011'].shape[0]),
                               str(events_conditions_r['011'].shape[0]),
                               str(events_conditions_v['011'].shape[0])])
            eventLog.addEntry("Set 4 - Image3 (Cookie)",
                              ["1", "0", "0",
                               str(count['100']),
                               str(events_conditions_b['100'].shape[0]),
                               str(events_conditions_r['100'].shape[0]),
                               str(events_conditions_v['100'].shape[0])])
            eventLog.addEntry("Set 5 - Image5 (Knife)",
                              ["1", "0", "1",
                               str(count['101']),
                               str(events_conditions_b['101'].shape[0]),
                               str(events_conditions_r['101'].shape[0]),
                               str(events_conditions_v['101'].shape[0])])
            eventLog.addEntry("Set 6 - Image4 (Kushi)",
                              ["1", "1", "0",
                               str(count['110']),
                               str(events_conditions_b['110'].shape[0]),
                               str(events_conditions_r['110'].shape[0]),
                               str(events_conditions_v['110'].shape[0])])
            eventLog.addEntry("Set 7 - Image6 (Pencil)",
                              ["1", "1", "1",
                               str(count['111']),
                               str(events_conditions_b['111'].shape[0]),
                               str(events_conditions_r['111'].shape[0]),
                               str(events_conditions_v['111'].shape[0])])

        faulty_epochs = {}
        if epochs_conditions['001'] != None:
            faulty_epochs['001'] = np.array(range(len(epochs_conditions['001'])))
        else:
            faulty_epochs['001'] = np.array([])
        faulty_epochs['010'] = np.array([])
        faulty_epochs['011'] = np.array([])
        faulty_epochs['100'] = np.array([])
        faulty_epochs['101'] = np.array([])
        faulty_epochs['110'] = np.array([])
        faulty_epochs['111'] = np.array([])

        self.BadEpochsGroup(epochs_manager, faulty_epochs, eventLog, reason='Not data', method='index')
        self.DropMarkedEpochs(epochs_manager)

        if resemple == 'yes':
            self.ResampleEpochsGroup(epochs_manager, new_freq)

        if useRecreatedEvents == 'yes':
            events_conditions = events_conditions_r
        else:
            events_conditions = events_conditions_b
        events_conditions['log'] = eventLog
        epochs_manager['log'] = eventLog

        if resemple == 'yes':
            print('Original sampling rate:  ', prtMessage, 'Hz')
            print('New sampling rate:       ', epochs_manager[printNum].epochs_condition.info['sfreq'], 'Hz')

        return events_conditions, epochs_manager, voiceResponsesInfo, eventLog

    def PrepareEnvokedConditions(self, epochs_manager, eventLog, dedication='train'):
    # def PrepareEnvokedConditions(self, epochs_manager, eventLog, type='base', dedication='train'):
        # if type == 'base' or type == 'flip':
        if dedication == 'all':
            epochs_condition001 = epochs_manager['001'].epochs_condition
            epochs_condition010 = epochs_manager['010'].epochs_condition
            epochs_condition011 = epochs_manager['011'].epochs_condition
            epochs_condition100 = epochs_manager['100'].epochs_condition
            epochs_condition101 = epochs_manager['101'].epochs_condition
            epochs_condition110 = epochs_manager['110'].epochs_condition
            epochs_condition111 = epochs_manager['111'].epochs_condition
        # elif type == 'pga' and (dedication == 'train' or dedication == 'test'):
        elif dedication == 'train' or dedication == 'test':
            epochs_condition001 = epochs_manager['001'].PGA[dedication]
            epochs_condition010 = epochs_manager['010'].PGA[dedication]
            epochs_condition011 = epochs_manager['011'].PGA[dedication]
            epochs_condition100 = epochs_manager['100'].PGA[dedication]
            epochs_condition101 = epochs_manager['101'].PGA[dedication]
            epochs_condition110 = epochs_manager['110'].PGA[dedication]
            epochs_condition111 = epochs_manager['111'].PGA[dedication]


        if epochs_condition001 is not None:
            evoked_condition001 = epochs_condition001.average()
        else:
            evoked_condition001 = None

        if epochs_condition010 is not None:
            evoked_condition010 = epochs_condition010.average()
        else:
            evoked_condition010 = None

        if epochs_condition011 is not None:
            evoked_condition011 = epochs_condition011.average()
        else:
            evoked_condition011 = None

        if epochs_condition100 is not None:
            evoked_condition100 = epochs_condition100.average()
        else:
            evoked_condition100 = None

        if epochs_condition101 is not None:
            evoked_condition101 = epochs_condition101.average()
        else:
            evoked_condition101 = None

        if epochs_condition110 is not None:
            evoked_condition110 = epochs_condition110.average()
        else:
            evoked_condition110 = None

        if epochs_condition111 is not None:
            evoked_condition111 = epochs_condition111.average()
        else:
            evoked_condition111 = None

        evoked_conditions = {'001': evoked_condition001, '010': evoked_condition010, '011': evoked_condition011,
                             '100': evoked_condition100, '101': evoked_condition101, '110': evoked_condition110,
                             '111': evoked_condition111, 'log': eventLog}

        return evoked_conditions

    def DetectOutlierEpochs(self, data, stdBand, scoreTh):

        epochScoreList = []
        for ep in range(self.numberOfChannels):
            # Excessive fluctuation detection
            ch = data[:, ep, :]

            meanVec = np.mean(ch, axis=1)
            ch0 = (ch.transpose() - meanVec).transpose()
            stdVec = np.std(ch0, axis=0)

            # 1 - 68.2, 2 - 95.4, 3 - 99.7
            th1 = stdBand * stdVec

            chScore = np.zeros(ch0.shape, dtype=bool)
            for k in range(ch0.shape[1]):
                chScore[:, k] = np.abs(ch0[:, k]) < th1[k]

            epochScoreVal1 = np.round(100 * np.sum(np.int64(chScore), axis=1) / chScore.shape[1])

            # No fluctuation detection
            ch1 = ch0.copy()
            ch1[epochScoreVal1 < scoreTh, :] = 0
            ch1[np.logical_not(chScore)] = 0

            minVec = np.min(ch1, axis=1)
            ch2_min = (ch1.transpose() - minVec).transpose()
            stdVec_min = np.std(ch2_min, axis=0)

            ch3 = ch2_min.copy()
            cnt = np.percentile(ch3, scoreTh)
            ch0_min = ch2_min - cnt

            maxVec = np.max(ch1, axis=1)
            ch2_max = (ch1.transpose() - maxVec).transpose()
            stdVec_max = np.std(ch2_max, axis=0)

            ch3 = ch2_max.copy()
            cnt = np.percentile(ch3, 100 - scoreTh)
            ch0_max = ch2_max - cnt

            # 1 - 68.2, 2 - 95.4, 3 - 99.7
            th2_min = stdBand * stdVec_min
            th2_max = stdBand * stdVec_max

            chScore2_min = np.zeros(ch0_min.shape, dtype=bool)
            chScore2_max = np.zeros(ch0_max.shape, dtype=bool)
            for k in range(ch0_min.shape[1]):
                chScore2_min[:, k] = np.abs(ch0_min[:, k]) < th2_min[k]
                chScore2_max[:, k] = np.abs(ch0_max[:, k]) < th2_max[k]
            chScore2_total = np.logical_or(chScore2_min, chScore2_max)

            epochScoreVal2 = np.round(100 * np.sum(np.int64(chScore2_total), axis=1) / chScore2_total.shape[1])

            # Combined scoring
            epochScoreVal = np.min(
                np.concatenate((epochScoreVal1.reshape(1, -1), epochScoreVal2.reshape(1, -1)), axis=0), axis=0)
            epochScoreList.append(epochScoreVal)

        return epochScoreList

    def DropOutlierEpochs(self, epochs_manager, Settings):

        # Settings = epochs_manager['settings']
        faulty_epochs = {}
        for dtype in Settings['epochTypes']:
            data = epochs_manager[dtype].epochs_condition.get_data()[:, :self.numberOfChannels, :]

            epochScoreList = self.DetectOutlierEpochs(data, Settings["stdBand"], Settings["scoreTh"])

            for bc in sorted(Settings['badChannels'], reverse=True):
                epochScoreList.pop(bc)
            totalEpochScore = np.mean(np.array(epochScoreList), axis=0)

            locs = np.where(totalEpochScore < Settings["scoreTh"])[0]
            faulty_epochs[dtype] = locs
        faulty_epochs['001'] = np.array([])

        self.BadEpochsGroup(epochs_manager, faulty_epochs, epochs_manager['log'], 'Outlier', 'index')
        self.DropMarkedEpochs(epochs_manager)

    def PrepareStage1Data(self, Settings):
        inDataDir = Settings.pop("inDataDir")
        DataFileNames = Settings.pop("DataFileNames")
        outDataDir = Settings.pop("outDataDir")
        Settings["badChannels"].extend(self.badChannels)
        Settings["badChannels"] = list(set(Settings["badChannels"]))
        Settings = {"subject": ''} | Settings

        # Input Check
        fileList = []
        if not os.path.exists(inDataDir):
            raise Exception('Read directory does not exist.')

        elif DataFileNames != "all":
            for name in DataFileNames:
                if not os.path.exists(join(inDataDir, name)):
                    raise Exception('File does not exist in the specified folder.')
            fileList.extend(DataFileNames)

        else:
            for file in os.listdir(inDataDir):
                if file.endswith(".fif"):
                    fileList.append(file)

        if not os.path.exists(outDataDir):
            os.mkdir(outDataDir)

        # Print current settings
        print("\n", pd.DataFrame.from_dict(Settings, orient='index', columns=["Settings"]), "\n")

        for name in fileList:
            raw = mne.io.read_raw_fif(join(inDataDir, name), preload=Settings["preload"])

            # Kill bad channels
            data = raw.get_data()
            data[Settings["badChannels"]] = data[Settings["badChannels"]] * 0
            filt = mne.io.RawArray(data, raw.info)

            # Perform BPF and Detrending on the raw data.
            if Settings["RawDataFiltration"] == 'yes':
                filt = self.FilterRawData_BPF(filt, cutoff_l=Settings["cutoff_l"], cutoff_h=Settings["cutoff_h"])
                filt = self.FilterRawData_DT(filt, DT=Settings["DT"], DT_param=Settings["DT_param"])

            # Preparation of events and epochs for further analysis.
            events_conditions, epochs_manager, voiceResponsesInfo, eventLog = \
                self.EpockingProcedures(filt, resemple=Settings["resemple"], new_freq=Settings["new_freq"],
                                        tmin=Settings["tmin"], tmax=Settings["tmax"], baseline=Settings["baseline"],
                                        min_event_duration=Settings["min_event_duration"],
                                        min_voice_duration=Settings["min_voice_duration"],
                                        reconfigureEvents=Settings["reconfigureEvents"],
                                        errorMargin=Settings["errorMargin"], grad=Settings["grad"],
                                        preload=Settings["preload"],
                                        dropFaultyResponseEpochs=Settings["dropFaultyResponseEpochs"],
                                        voiceOutliers=Settings["voiceOutliers"],
                                        useRecreatedEvents=Settings["useRecreatedEvents"])

            if Settings["dropOutliers"] == 'yes':
                self.DropOutlierEpochs(epochs_manager, Settings)

            # Print current logfile
            print('\nData file: ' + name)
            eventLog.printLog()
            print("\n")

            # Create data Dump object
            dataDump = Dump_File(Settings["Sumitomo_Long_Short"])
            # Fill Dump object
            Settings["subject"] = name[Settings["idStartInd"]:Settings["idEndInd"]]
            dataDump.push(Settings, "settings")
            if Settings["fullDump"] == 'yes':
                dataDump.push(voiceResponsesInfo, "voice")
                dataDump.push(events_conditions, "events")
            dataDump.push(epochs_manager, "epochs")

            outFileName = name[:Settings["idEndInd"]] + "_Stage1.pkl"
            self.Save_Object(dataDump, outFileName, outDataDir)

    def decomposeEpochData(self, data, cycles=5):
        myImfs = []
        for m in range(data.shape[0]):

            imfList = []
            nch0 = data[m, :].reshape(-1, 1)
            for k in range(cycles):
                imf = sift.get_next_imf(nch0[:, 0])[0]
                imfList.append(imf[:, 0])
                nch0 = nch0 - imf

            imfs = np.array(imfList)

            myImfs.append(imfs)

        return np.array(myImfs)

    def DatasetEMD(self, epochs_manager, Settings):

        cycles = Settings["cycles"]
        emdData = {}
        for subSetName in Settings["epochTypes"]:
            epochs = epochs_manager[subSetName].epochs_condition.get_data()[:, :self.numberOfChannels, :]

            d = epochs.shape
            emd_epochs = np.zeros((d[0], d[1], cycles, d[2]))
            for k in range(epochs.shape[0]):
                imfs1 = self.decomposeEpochData(epochs[k], cycles=cycles)
                emd_epochs[k] = imfs1

            emdData[subSetName] = emd_epochs

        return emdData

    def PrepareStage2EMDdata(self, Settings):
        inDataDir = Settings.pop("inDataDir")
        DataFileNames = Settings.pop("DataFileNames")
        outDataDir = Settings.pop("outDataDir")

        # Input Check
        fileList = []
        if not os.path.exists(inDataDir):
            raise Exception('Read directory does not exist.')

        elif DataFileNames != "all":
            for name in DataFileNames:
                if not os.path.exists(join(inDataDir, name)):
                    raise Exception('File does not exist in the specified folder.')
            fileList.extend(DataFileNames)

        else:
            for file in os.listdir(inDataDir):
                if file.endswith(".pkl"):
                    fileList.append(file)

        if not os.path.exists(outDataDir):
            os.mkdir(outDataDir)

        for name in fileList:

            # Load dump file
            dataDump = self.Load_Object(name, inDataDir)

            # Update settings
            preSettings = dataDump.get("settings")
            Settings = preSettings | Settings
            Settings["subject"] = name[Settings["idStartInd"]:Settings["idEndInd"]]

            # Print current settings
            print(pd.DataFrame.from_dict(Settings, orient='index', columns=["Settings"]))

            # Calculate EMD
            epochs_manager = dataDump.get("epochs")
            emdData = self.DatasetEMD(epochs_manager, Settings)

            # Prepare Dump file
            dataDump.push(epochs_manager, "epochs")
            dataDump.push(Settings, "settings")
            dataDump.push(emdData, "emd")

            # Save file
            outFileName = name[:Settings["idEndInd"]] + "_Stage2_EMD.pkl"
            self.Save_Object(dataDump, outFileName, outDataDir)

    def Load_EMD_Dataset(self, Settings):

        dataDump = self.Load_Object(Settings["DataFileNames"][0], Settings["inDataDir"])

        epochs_manager = dataDump.get("epochs")
        if Settings["getFullDump"] == 'yes':
            voiceResponsesInfo = dataDump.get("voice")
            events_conditions = dataDump.get("events")

        # Print current settings
        preSettings = dataDump.get("settings")
        preSettings["filtrationType"] = Settings["getFilt"]
        print(pd.DataFrame.from_dict(preSettings, orient='index', columns=["Settings"]), "\n")
        epochs_manager['settings'] = preSettings

        if Settings["loadEMD"] == 'yes':
            emdData = dataDump.get("emd")
        else:
            print('Loading base data.')
            if Settings["getFullDump"] == 'yes':
                return epochs_manager, events_conditions, voiceResponsesInfo
            else:
                return epochs_manager

        if Settings["getFilt"] == 'base':
            print('Loading base data.')
            if Settings["getFullDump"] == 'yes':
                return epochs_manager, events_conditions, voiceResponsesInfo
            else:
                return epochs_manager

        elif Settings["getFilt"][:-1] == 'IMF' and str.isnumeric(Settings["getFilt"][-1]):
            print('Loading IMF data.')

            imfNum = int(Settings["getFilt"][-1])
            if imfNum < 0:
                print('Loading base data.')
                if Settings["getFullDump"] == 'yes':
                    return epochs_manager, events_conditions, voiceResponsesInfo
                else:
                    return epochs_manager

            for subSet in preSettings["epochTypes"]:
                emdTensor = emdData[subSet]
                if emdTensor.shape[2] <= imfNum:
                    msg = 'IMF of such order is not in the set. Highest avaliable order is: ' \
                          + str(emdTensor.shape[2])
                    raise Exception(msg)

                emdSubTens = emdTensor[:,:,imfNum-1,:]
                emd_epochs = np.concatenate((emdSubTens,
                                     epochs_manager[subSet].epochs_condition.get_data()[:,self.numberOfChannels:,:]),
                                     axis=1)

                epochs_manager[subSet].epochs_condition = \
                    mne.EpochsArray(emd_epochs, epochs_manager[subSet].epochs_condition.info,
                                    tmin=preSettings["tmin"], baseline=preSettings["baseline"])

            if Settings["getFullDump"] == 'yes':
                return epochs_manager, events_conditions, voiceResponsesInfo
            else:
                return epochs_manager

        elif Settings["getFilt"] == 'emd_filter1':
            print('Loading data filtered using emd. Reduction of high frequency noise.')

            for subSet in preSettings["epochTypes"]:
                emdTensor = emdData[subSet]
                if emdTensor.shape[2] < 1:
                    msg = 'Required IMFs are not in the set. Please calculate IMFs at least up to 1st order.'
                    raise Exception(msg)

                emdSubTens = emdTensor[:,:,0,:]

                filt_epochs = epochs_manager[subSet].epochs_condition.get_data()[:,:self.numberOfChannels,:] - emdSubTens
                emd_epochs = np.concatenate((filt_epochs,
                                     epochs_manager[subSet].epochs_condition.get_data()[:,self.numberOfChannels:, :]),
                                     axis=1)

                epochs_manager[subSet].epochs_condition = \
                    mne.EpochsArray(emd_epochs, epochs_manager[subSet].epochs_condition.info,
                                    tmin=preSettings["tmin"], baseline=preSettings["baseline"])

            if Settings["getFullDump"] == 'yes':
                return epochs_manager, events_conditions, voiceResponsesInfo
            else:
                return epochs_manager

        elif Settings["getFilt"] == 'emd_filter2':
            print('Loading data filtered using emd. Reconstruction of low frequency IMFs.')

            for subSet in preSettings["epochTypes"]:
                emdTensor = emdData[subSet]
                if emdTensor.shape[2] <= 4:
                    msg = 'Required IMFs are not in the set. Please calculate IMFs at least up to 1st order.'
                    raise Exception(msg)

                emdSubTens3 = emdTensor[:, :, 2, :]
                emdSubTens4 = emdTensor[:, :, 3, :]

                emd_epochs = np.concatenate((emdSubTens3 + emdSubTens4,
                                     epochs_manager[subSet].epochs_condition.get_data()[:,self.numberOfChannels:, :]),
                                     axis=1)

                epochs_manager[subSet].epochs_condition = \
                    mne.EpochsArray(emd_epochs, epochs_manager[subSet].epochs_condition.info,
                                    tmin=preSettings["tmin"], baseline=preSettings["baseline"])

            if Settings["getFullDump"] == 'yes':
                return epochs_manager, events_conditions, voiceResponsesInfo
            else:
                return epochs_manager

        else:
            raise Exception('Unrecognized getFilt setting!')

    def dataSplitML(self, sizeList):
        uniqueSize = list(set(sizeList))

        combSelections = {}
        for dsize in uniqueSize:
            TrSize = int(np.round(dsize * self.Settings["TrSize"] / 100))
            indList = np.array(range(dsize))
            shuffle(indList)

            combSelections[dsize] = {'train': indList[:TrSize], 'test': indList[TrSize:]}

        return combSelections

    def PrepareMLsets_PGA(self, segment_manager, numberOfAttempts,
                          numberOfElements=5, maxIntersection=2, TeTrSize='max'):

        epochs_manager = segment_manager.epochs_manager
        Settings = segment_manager.settings
        sizeList = []
        for epType in Settings['epochTypes']:
            sizeList.append(epochs_manager[epType].epochs_condition.selection.size)

        if type(TeTrSize) == str:
            cg = IntersectedCombinationsGenerator(numberOfElements, maxIntersection)
            combSelections = cg.CreateCombinationsSet(sizeList, numberOfAttempts, TeTrSize)
        else:
            combSelections = self.dataSplitML(sizeList)
        segment_manager.combinationsSet = combSelections

        trSizeLog = [0]
        teSizeLog = [0]
        combSet = combSelections[epochs_manager[Settings['epochTypes'][0]].epochs_condition.selection.size]
        for epType in Settings['epochTypes']:
            trSizeLog.append(len(combSet['train']))
            teSizeLog.append(len(combSet['test']))

        eventLog = epochs_manager['log']
        eventLog.addInfo('Train', trSizeLog)
        eventLog.addInfo('Test', teSizeLog)
        segment_manager.epochs_manager['log'] = eventLog

        return segment_manager

    def PrepareDataForML(self, Settings):

        inDataDir = Settings["inDataDir"]
        DataFileNames = Settings["DataFileNames"]
        outDataDir = Settings["outDataDir"]

        # Input Check
        fileList = []
        if not os.path.exists(inDataDir):
            raise Exception('Read directory does not exist.')

        elif DataFileNames != "all":
            for name in DataFileNames:
                if not os.path.exists(join(inDataDir, name)):
                    raise Exception('File does not exist in the specified folder.')
            fileList.extend(DataFileNames)

        else:
            for file in os.listdir(inDataDir):
                if file.endswith(".pkl"):
                    fileList.append(file)

        if not os.path.exists(outDataDir):
            os.mkdir(outDataDir)

        preset = Settings.copy()
        for name in fileList:
            Settings = preset.copy()
            Settings["DataFileNames"] = [name]
            epochs_manager = self.Load_EMD_Dataset(Settings)

            segment_manager = SegmentManager_Sumitomo_Long(epochs_manager, TDT=Settings["TDT"])
            segment_manager = self.PrepareMLsets_PGA(segment_manager, Settings["numberOfAttempts"],
                                                    Settings["numberOfElements"], Settings["maxIntersection"],
                                                    Settings["TrSize"])

            # Print current logfile
            eventLog = epochs_manager['log']
            eventLog.printLog()

            Settings = epochs_manager['settings']

            dataDump = Dump_File(Settings["Sumitomo_Long_Short"])
            dataDump.push(segment_manager, "segments")
            dataDump.push(Settings, "settings")

            outFileName = name[:Settings["idEndInd"]] + "_PGA.pkl"
            self.Save_Object(dataDump, outFileName, outDataDir)

    def LoadSegmentData(self, fileName, folderPath):

        dataDump = self.Load_Object(fileName, folderPath)
        print(pd.DataFrame.from_dict(dataDump.get("settings"), orient='index', columns=["Settings"]), "\n")
        return dataDump.get("segments")

    def Segment2Epoch_manager(self, segment_manager, timeRange, source, baseline):
        epochs_manager = segment_manager.epochs_manager
        Settings = epochs_manager['settings']

        timeSettings = np.array(timeRange) / 1000
        baseline = tuple(np.array(baseline) / 1000)

        dataSegment = segment_manager.GetSegment(timeRange)
        for epType in Settings['epochTypes']:
            tempConditions = epochs_manager[epType].epochs_condition.copy()
            ch_names = tempConditions.info.ch_names
            rm_names = ch_names[self.numberOfChannels:]
            tempConditions.drop_channels(rm_names)
            info = tempConditions.info

            trainSet = dataSegment[epType]['train']
            testSet = dataSegment[epType]['test']
            coreSet = dataSegment[epType][source]

            epochs_manager[epType].PGA['train'] = mne.EpochsArray(trainSet, info,
                                tmin=timeSettings[0], baseline=baseline)
            epochs_manager[epType].PGA['test'] = mne.EpochsArray(testSet, info,
                                tmin=timeSettings[0], baseline=baseline)
            epochs_manager[epType].epochs_condition = mne.EpochsArray(coreSet, info,
                                 tmin=timeSettings[0], baseline=baseline)

        return epochs_manager


class SegmentManager_Sumitomo_Long:
    def __init__(self, epochs_manager, TDT):
        super().__init__()

        self.epochs_manager = epochs_manager.copy()
        self.settings = epochs_manager['settings']
        self.numberOfChannels = self.settings["numberOfChannels"]
        self.TDT = TDT

        self.dataSegmentList = []
        self.numberOfSegments = 0

        self.combinationsSet = []

    def alignClustersPolarity_known(self, dataSegment):

        timeVec = dataSegment['time']
        TimeStamps = dataSegment['stamps']
        baseline = dataSegment['baseline']

        baseED = np.argmin(np.abs(timeVec - baseline[1] / 1000))
        for dtype in self.settings['epochTypes']:

            baseData = self.epochs_manager[dtype].epochs_condition.get_data().copy()
            dataIn = baseData[:, :self.numberOfChannels, baseED:TimeStamps[1]].copy()
            dataBase = dataSegment[dtype]['base']

            dataOut = dataBase.copy()
            for ch in range(self.numberOfChannels):
                if ch not in self.settings['badChannels']:
                    subData = dataIn[:, ch, :]

                    corrMat = np.corrcoef(subData) - np.eye(subData.shape[0])

                    pdist = spc.distance.pdist(corrMat)
                    linkage = spc.linkage(pdist, method='complete')
                    idx = spc.fcluster(linkage, 2, 'maxclust')

                    dataFlip = dataOut[:, ch, :].copy()
                    dataFlip[idx == 2] = -dataFlip[idx == 2]

                    if self.TDT == "known1":
                        avgData = np.mean(dataFlip, axis=0)
                        if np.sum(avgData[avgData > 0]) < np.sum(avgData[avgData < 0]):
                            dataFlip = -dataFlip

                    dataOut[:, ch, :] = dataFlip
            dataSegment[dtype]['flip'] = dataOut

    def gauss_depolarization_pga(self, dataSegment):

        timeVec = dataSegment['time']
        TimeStamps = dataSegment['stamps']
        baseline = dataSegment['baseline']

        baseED = np.argmin(np.abs(timeVec - baseline[1] / 1000))
        for dtype in self.settings['epochTypes']:

            baseData = self.epochs_manager[dtype].epochs_condition.get_data().copy()
            dataIn = baseData[:, :self.numberOfChannels, baseED:TimeStamps[1]].copy()

            dataOut = self.gaussDepolarization(dataIn.copy())
            dataSegment[dtype]['flip'] = dataOut

    def reference_depolarization_pga(self, dataSegment):

        timeVec = dataSegment['time']
        TimeStamps = dataSegment['stamps']
        baseline = dataSegment['baseline']

        baseED = np.argmin(np.abs(timeVec - baseline[1] / 1000))
        refData = np.zeros((self.numberOfChannels, TimeStamps[1] - baseED))
        for dtype in self.settings['epochTypes']:

            baseData = self.epochs_manager[dtype].epochs_condition.get_data().copy()
            dataIn = baseData[:, :self.numberOfChannels, baseED:TimeStamps[1]].copy()

            refData = refData + np.sum(dataIn, axis=0)

        for dtype in self.settings['epochTypes']:

            baseData = self.epochs_manager[dtype].epochs_condition.get_data().copy()
            dataIn = baseData[:, :self.numberOfChannels, baseED:TimeStamps[1]].copy()

            dataOut = self.referenceDepolarization(dataIn, refData)
            dataSegment[dtype]['flip'] = dataOut

    def clusters_depolarization_pga(self, dataSegment):

        timeVec = dataSegment['time']
        TimeStamps = dataSegment['stamps']
        baseline = dataSegment['baseline']

        baseED = np.argmin(np.abs(timeVec - baseline[1] / 1000))
        for dtype in self.settings['epochTypes']:

            baseData = self.epochs_manager[dtype].epochs_condition.get_data().copy()
            dataIn = baseData[:, :self.numberOfChannels, baseED:TimeStamps[1]].copy()

            combSet = self.combinationsSet[dataIn.shape[0]]
            trainSet = []
            for pgaSet in combSet['train']:
                trainSet.extend(list(pgaSet))
            trainSet = list(set(trainSet))

            dataBase = dataSegment[dtype]['base']
            trainDataset = dataBase[trainSet, :, :].copy()

            dataOut = trainDataset.copy()
            for ch in range(self.numberOfChannels):
                if ch not in self.settings['badChannels']:
                    subData = dataIn[trainSet, ch, :]

                    corrMat = np.corrcoef(subData) - np.eye(subData.shape[0])

                    pdist = spc.distance.pdist(corrMat)
                    linkage = spc.linkage(pdist, method='complete')
                    idx = spc.fcluster(linkage, 2, 'maxclust')

                    dataFlip = dataOut[:, ch, :].copy()
                    subMat = corrMat[idx == 1, :]
                    subMat = subMat[:, idx == 2]
                    if np.mean(subMat) < 0:
                        dataFlip[idx == 2] = -dataFlip[idx == 2]

                    avgData = np.mean(dataFlip, axis=0)
                    if np.sum(avgData[avgData > 0]) < np.sum(avgData[avgData < 0]):
                        dataFlip = -dataFlip

                    dataOut[:, ch, :] = dataFlip
            dataSegment[dtype]['flip'] = dataOut

            trainData = []
            for pgaSet in combSet['train']:
                pgaSetInd = []
                for val in pgaSet:
                    pgaSetInd.append(trainSet.index(val))
                group = dataOut[pgaSetInd, :, :]
                trainData.append(np.mean(group, axis=0))
            dataSegment[dtype]['train'] = np.array(trainData)

            testData = []
            for pgaSet in combSet['test']:

                groupOut = dataBase[pgaSet, :, :].copy()
                for ch in range(self.numberOfChannels):
                    if ch not in self.settings['badChannels']:
                        subData = dataIn[pgaSet, ch, :]

                        corrMat = np.corrcoef(subData) - np.eye(subData.shape[0])

                        pdist = spc.distance.pdist(corrMat)
                        linkage = spc.linkage(pdist, method='complete')
                        idx = spc.fcluster(linkage, 2, 'maxclust')

                        dataFlip = groupOut[:, ch, :].copy()
                        subMat = corrMat[idx == 1, :]
                        subMat = subMat[:, idx == 2]
                        if np.mean(subMat) < 0:
                            dataFlip[idx == 2] = -dataFlip[idx == 2]

                        avgData = np.mean(dataFlip, axis=0)
                        if np.sum(avgData[avgData > 0]) < np.sum(avgData[avgData < 0]):
                            dataFlip = -dataFlip

                        groupOut[:, ch, :] = dataFlip
                testData.append(np.mean(groupOut, axis=0))
            dataSegment[dtype]['test'] = np.array(testData)

    def clusters_depolarization_half_pga(self, dataSegment):

        timeVec = dataSegment['time']
        TimeStamps = dataSegment['stamps']
        baseline = dataSegment['baseline']

        baseED = np.argmin(np.abs(timeVec - baseline[1] / 1000))
        for dtype in self.settings['epochTypes']:

            baseData = self.epochs_manager[dtype].epochs_condition.get_data().copy()
            dataIn = baseData[:, :self.numberOfChannels, baseED:TimeStamps[1]].copy()

            combSet = self.combinationsSet[dataIn.shape[0]]

            trainSet = []
            for pgaSet in combSet['train']:
                trainSet.extend(list(pgaSet))
            trainSet = list(set(trainSet))

            dataBase = dataSegment[dtype]['base']
            trainDataset = dataBase[trainSet, :, :].copy()

            dataOut = trainDataset.copy()
            for ch in range(self.numberOfChannels):
                if ch not in self.settings['badChannels']:
                    subData = dataIn[trainSet, ch, :]

                    corrMat = np.corrcoef(subData) - np.eye(subData.shape[0])

                    pdist = spc.distance.pdist(corrMat)
                    linkage = spc.linkage(pdist, method='complete')
                    idx = spc.fcluster(linkage, 2, 'maxclust')

                    dataFlip = dataOut[:, ch, :].copy()
                    subMat = corrMat[idx == 1, :]
                    subMat = subMat[:, idx == 2]
                    if np.mean(subMat) < 0:
                        dataFlip[idx == 2] = -dataFlip[idx == 2]

                    dataOut[:, ch, :] = dataFlip
            dataSegment[dtype]['flip'] = dataOut

            trainData = []
            for pgaSet in combSet['train']:
                pgaSetInd = []
                for val in pgaSet:
                    pgaSetInd.append(trainSet.index(val))
                group = dataOut[pgaSetInd, :, :]
                trainData.append(np.mean(group, axis=0))
            dataSegment[dtype]['train'] = np.array(trainData)

            testSet = []
            for pgaSet in combSet['test']:
                testSet.extend(list(pgaSet))
            testSet = list(set(testSet))

            testData = dataBase[testSet, :, :].copy()
            dataSegment[dtype]['test'] = np.array(testData)

    def alignClustersPolarity_unknown(self, dataSegment):

        timeVec = dataSegment['time']
        TimeStamps = dataSegment['stamps']
        baseline = dataSegment['baseline']

        baseED = np.argmin(np.abs(timeVec - baseline[1] / 1000))
        flag = 1
        idxSort = {}
        for dtype in self.settings['epochTypes']:
            idxSort[dtype] = {}

            baseData = self.epochs_manager[dtype].epochs_condition.get_data().copy()
            if flag:
                dataIn = baseData[:, :self.numberOfChannels, baseED:TimeStamps[1]].copy()
                dataShape = {dtype: dataIn.shape[0]}
                flag = 0
            else:
                dataPart = baseData[:, :self.numberOfChannels, baseED:TimeStamps[1]].copy()
                dataIn = np.concatenate((dataIn, dataPart), axis=0)
                dataShape[dtype] = dataPart.shape[0]


        for ch in range(self.numberOfChannels):
            if ch not in self.settings['badChannels']:
                subData = dataIn[:, ch, :]

                corrMat = np.corrcoef(subData) - np.eye(subData.shape[0])

                pdist = spc.distance.pdist(corrMat)
                linkage = spc.linkage(pdist, method='complete')
                idx = spc.fcluster(linkage, 2, 'maxclust')

                for k in range(len(self.settings['epochTypes'])):
                    dtype = self.settings['epochTypes'][k]
                    if k:
                        prdtype = self.settings['epochTypes'][k-1]
                        idxSort[dtype][ch] = idx[dataShape[prdtype]:dataShape[prdtype] + dataShape[dtype]]
                    else:
                        idxSort[dtype][ch] = idx[:dataShape[dtype]]

        for dtype in self.settings['epochTypes']:

            dataOut = dataSegment[dtype]['base'].copy()
            for ch in range(self.numberOfChannels):
                if ch not in self.settings['badChannels']:

                    dataFlip = dataOut[:, ch, :].copy()
                    idx = idxSort[dtype][ch]

                    dataFlip[idx == 2] = -dataFlip[idx == 2]

                    dataOut[:, ch, :] = dataFlip
            dataSegment[dtype]['flip'] = dataOut

    def alignClustersPolarity_clustered(self, dataSegment, numberOfTypes=12):

        timeVec = dataSegment['time']
        TimeStamps = dataSegment['stamps']
        baseline = dataSegment['baseline']

        baseED = np.argmin(np.abs(timeVec - baseline[1] / 1000))
        flag = 1
        idxSort = {}
        for dtype in self.settings['epochTypes']:
            idxSort[dtype] = {}

            baseData = self.epochs_manager[dtype].epochs_condition.get_data().copy()
            if flag:
                dataIn = baseData[:, :self.numberOfChannels, baseED:TimeStamps[1]].copy()
                dataShape = {dtype: dataIn.shape[0]}
                flag = 0
            else:
                dataPart = baseData[:, :self.numberOfChannels, baseED:TimeStamps[1]].copy()
                dataIn = np.concatenate((dataIn, dataPart), axis=0)
                dataShape[dtype] = dataPart.shape[0]


        for ch in range(self.numberOfChannels):
            if ch not in self.settings['badChannels']:
                subData = dataIn[:, ch, :]

                corrMat = np.corrcoef(subData) - np.eye(subData.shape[0])

                pdist = spc.distance.pdist(corrMat)
                linkage = spc.linkage(pdist, method='complete')
                idx = spc.fcluster(linkage, numberOfTypes, 'maxclust')

                for k in range(len(self.settings['epochTypes'])):
                    dtype = self.settings['epochTypes'][k]
                    if k:
                        prdtype = self.settings['epochTypes'][k-1]
                        idxSort[dtype][ch] = idx[dataShape[prdtype]:dataShape[prdtype] + dataShape[dtype]]
                    else:
                        idxSort[dtype][ch] = idx[:dataShape[dtype]]


        for dtype in self.settings['epochTypes']:

            dataOut = dataSegment[dtype]['base'].copy()
            for ch in range(self.numberOfChannels):
                if ch not in self.settings['badChannels']:

                    dataFlip = dataOut[:, ch, :].copy()
                    idx = idxSort[dtype][ch]

                    buckets = np.zeros(numberOfTypes)
                    for k in range(1, numberOfTypes + 1):
                        buckets[k - 1] = len(idx[idx == k])

                    if self.TDT == "clustered":

                        red = 0
                        meanSigs = np.mean(dataFlip[idx == 1], axis=0).reshape([1,-1])
                        trans = {1: 1}
                        for k in range(2, numberOfTypes + 1):
                            if buckets[k - 1] > 0:
                                meanSigs = np.concatenate((meanSigs, np.mean(dataFlip[idx == k], axis=0).reshape([1,-1])), axis=0)
                            else:
                                red = red + 1
                            trans[k] = k - red

                        corrMatRec = np.corrcoef(meanSigs) - np.eye(meanSigs.shape[0])
                        corrWeights = np.mean(corrMatRec, axis=0)

                        for k in range(corrWeights.size):
                            if corrWeights[k] < 0:
                                dataFlip[idx == trans[k+1]] = -dataFlip[idx == trans[k+1]]

                    elif self.TDT == "clustered1" or self.TDT == "half_pga1":

                        refInd = np.argmax(buckets) + 1

                        refSig = np.mean(dataFlip[idx == refInd], axis=0)
                        for k in range(1, numberOfTypes + 1):
                            if k != refInd and buckets[k-1] > 0:
                                testSig = np.mean(dataFlip[idx == k], axis=0)
                                corrVal = np.corrcoef(refSig, testSig)
                                if corrVal[0,1] < 0:
                                    dataFlip[idx == k] = -dataFlip[idx == k]

                    elif self.TDT == "clustered2":

                        red = 0
                        meanSigs = np.mean(dataFlip[idx == 1], axis=0).reshape([1,-1])
                        trans = {1: 1}
                        for k in range(2, numberOfTypes + 1):
                            if buckets[k - 1] > 0:
                                meanSigs = np.concatenate((meanSigs, np.mean(dataFlip[idx == k], axis=0).reshape([1,-1])), axis=0)
                            else:
                                red = red + 1
                            trans[k] = k - red

                        for k in range(meanSigs.shape[0]):
                            corrMatRec = np.corrcoef(meanSigs) - np.eye(meanSigs.shape[0])
                            loc = np.argmax(np.abs(corrMatRec[k,:]))
                            if corrMatRec[k,loc] < 0:
                                meanSigs = -meanSigs[k,:]
                                dataFlip[idx == trans[k + 1]] = -dataFlip[idx == trans[k + 1]]

                    dataOut[:, ch, :] = dataFlip
            dataSegment[dtype]['flip'] = dataOut

    def addSegmentData(self, timeRange, baseline):

        timeVec = self.epochs_manager[self.settings['epochTypes'][0]].epochs_condition.times
        TimeStamps = (np.argmin(np.abs(timeVec - timeRange[0]/1000)), np.argmin(np.abs(timeVec - timeRange[1]/1000)))

        dataSegment = {'time': timeVec, 'stamps': TimeStamps, 'range': timeRange, 'baseline': baseline, 'MLready': 'no'}
        for dtype in self.settings['epochTypes']:

            baseData = self.epochs_manager[dtype].epochs_condition.get_data().copy()
            dataBase = baseData[:, :self.numberOfChannels, TimeStamps[0]:TimeStamps[1]].copy()

            databse = {'base': dataBase}
            dataSegment[dtype] = databse

        if self.TDT == "known" or self.TDT == "known1":
            self.alignClustersPolarity_known(dataSegment)
        elif self.TDT == "pga_gauss":
            self.gauss_depolarization_pga(dataSegment)
        elif self.TDT == "pga_ref":
            self.reference_depolarization_pga(dataSegment)
        elif self.TDT == "pga":
            self.clusters_depolarization_pga(dataSegment)
        elif self.TDT == "half_pga":
            self.clusters_depolarization_half_pga(dataSegment)
        elif self.TDT == "half_pga1":
            self.alignClustersPolarity_clustered(dataSegment)
        elif self.TDT == "unknown":
            self.alignClustersPolarity_unknown(dataSegment)
        elif self.TDT == "clustered" or self.TDT == "clustered1" or self.TDT == "clustered2":
            self.alignClustersPolarity_clustered(dataSegment)

        return dataSegment

    def segmentExists(self, timeRange):
        for dataSegment in self.dataSegmentList:
            if dataSegment['range'] == timeRange:
                return True
        return False

    def AddSegments(self, timeRangeList, baselineList):

        for k in range(len(timeRangeList)):
            timeRange = timeRangeList[k]
            if not self.segmentExists(timeRange):
                baseline = baselineList[k]
                self.dataSegmentList.append(self.addSegmentData(timeRange, baseline))
        self.numberOfSegments = len(self.dataSegmentList)

        self.prepareDataForML()

    def RemoveSegment(self, timeRange):
        for k in range(self.numberOfSegments):
            if self.dataSegmentList[k]['range'] == timeRange:
                del self.dataSegmentList[k]
        self.numberOfSegments = len(self.dataSegmentList)

    def pop(self, ind=-1):
        dataSegment = self.dataSegmentList.pop(ind)
        self.numberOfSegments = len(self.dataSegmentList)
        return dataSegment

    def GetSegment(self, timeRange):
        for k in range(self.numberOfSegments):
            if self.dataSegmentList[k]['range'] == timeRange:
                return self.dataSegmentList[k]

    def ResetTDT(self, newTDT):

        if self.TDT == 'All(PGA)':
            self.TDT = newTDT
        else:
            raise Exception("Unable to reset TDT due to incorrect pre-calculation settings.")

    def GetTDT(self):
        return self.TDT

    def createMLset_known(self, dataSegment):

        for epType in self.settings['epochTypes']:
            database = dataSegment[epType]
            data = database['flip']
            combSet = self.combinationsSet[data.shape[0]]

            testData = []
            for pgaSet in combSet['test']:
                group = data[pgaSet, :, :]
                testData.append(np.mean(group, axis=0))
            database['test'] = np.array(testData)

            trainData = []
            for pgaSet in combSet['train']:
                group = data[pgaSet, :, :]
                trainData.append(np.mean(group, axis=0))
            database['train'] = np.array(trainData)

            dataSegment[epType] = database

        return dataSegment

    def createMLset_classic(self, dataSegment):

        for epType in self.settings['epochTypes']:
            database = dataSegment[epType]
            data = database['base']

            combSet = self.combinationsSet[data.shape[0]]

            database['train'] = data[combSet['train'], :, :]
            database['test'] = data[combSet['test'], :, :]

            dataSegment[epType] = database

        return dataSegment

    def gaussDepolarization(self, data):

        x = np.arange(0, data.shape[2], 1)
        ref_pdf = norm.pdf(x, x.size / 2, x.size / 5)

        outData = np.zeros(data.shape)
        for ch in range(data.shape[1]):
            stitchData = np.concatenate((ref_pdf.reshape(1,-1), data[:,ch,:]), axis=0)
            corrMat = np.corrcoef(stitchData)
            for ep in range(data.shape[0]):
                if corrMat[0, ep+1] >= 0:
                    outData[ep, ch, :] = data[ep, ch, :]
                else:
                    outData[ep, ch, :] = -data[ep , ch, :]

        return outData

    def referenceDepolarization(self, data, refData):

        x = np.arange(0, data.shape[2], 1)

        outData = np.zeros(data.shape)
        for ch in range(data.shape[1]):
            stitchData = np.concatenate((refData[ch,:].reshape(1,-1), data[:,ch,:]), axis=0)
            corrMat = np.corrcoef(stitchData)
            for ep in range(data.shape[0]):
                if corrMat[0, ep+1] >= 0:
                    outData[ep, ch, :] = data[ep, ch, :]
                else:
                    outData[ep, ch, :] = -data[ep , ch, :]

        return outData

    def createMLset_pga(self, dataSegment):

        for epType in self.settings['epochTypes']:
            database = dataSegment[epType]
            data = database['flip']
            combSet = self.combinationsSet[data.shape[0]]

            testData = []
            for pgaSet in combSet['test']:
                group = data[pgaSet, :, :]
                dp_group = self.gaussDepolarization(group)
                testData.append(np.mean(dp_group, axis=0))
            database['test'] = np.array(testData)

            trainData = []
            for pgaSet in combSet['train']:
                group = data[pgaSet, :, :]
                dp_group = self.gaussDepolarization(group)
                trainData.append(np.mean(dp_group, axis=0))
            database['train'] = np.array(trainData)

            dataSegment[epType] = database

        return dataSegment

    def createMLset_unknown(self, dataSegment):

        for epType in self.settings['epochTypes']:
            database = dataSegment[epType]
            data = database['flip']
            combSet = self.combinationsSet[data.shape[0]]

            testData = []
            for pgaSet in combSet['test']:
                group = data[pgaSet, :, :]
                testData.append(np.mean(group, axis=0))
            database['test'] = np.array(testData)

            trainData = []
            for pgaSet in combSet['train']:
                group = data[pgaSet, :, :]
                trainData.append(np.mean(group, axis=0))
            database['train'] = np.array(trainData)

            dataSegment[epType] = database

        return dataSegment

    def createMLset_half_pga(self, dataSegment):

        for epType in self.settings['epochTypes']:
            database = dataSegment[epType]
            data = database['flip']
            combSet = self.combinationsSet[data.shape[0]]

            testSet = []
            for pgaSet in combSet['test']:
                testSet.extend(list(pgaSet))
            testSet = list(set(testSet))

            testData = data[testSet, :, :].copy()
            database['test'] = np.array(testData)

            trainData = []
            for pgaSet in combSet['train']:
                group = data[pgaSet, :, :]
                trainData.append(np.mean(group, axis=0))
            database['train'] = np.array(trainData)

            dataSegment[epType] = database

        return dataSegment

    def prepareDataForML(self):

        if self.TDT == "known" or self.TDT == "known1":
            for k in range(self.numberOfSegments):
                if self.dataSegmentList[k]['MLready'] == 'no':
                    self.dataSegmentList[k] = self.createMLset_known(self.dataSegmentList[k])
                    self.dataSegmentList[k]['MLready'] = 'yes'
        elif self.TDT == "classic":
            for k in range(self.numberOfSegments):
                if self.dataSegmentList[k]['MLready'] == 'no':
                    self.dataSegmentList[k] = self.createMLset_classic(self.dataSegmentList[k])
                    self.dataSegmentList[k]['MLready'] = 'yes'
        elif self.TDT == "pga_gauss" or self.TDT == "pga_ref":
            for k in range(self.numberOfSegments):
                if self.dataSegmentList[k]['MLready'] == 'no':
                    self.dataSegmentList[k] = self.createMLset_pga(self.dataSegmentList[k])
                    self.dataSegmentList[k]['MLready'] = 'yes'
        elif self.TDT == "pga":
            print('Only train flip data is available under current configuration.')
        elif self.TDT == "half_pga":
            print('Only train flip data is available under current configuration.')
        elif self.TDT == "half_pga1":
            for k in range(self.numberOfSegments):
                if self.dataSegmentList[k]['MLready'] == 'no':
                    self.dataSegmentList[k] = self.createMLset_half_pga(self.dataSegmentList[k])
                    self.dataSegmentList[k]['MLready'] = 'yes'
        elif self.TDT == "unknown" or self.TDT == "clustered" or self.TDT == "clustered1" or self.TDT == "clustered2":
            for k in range(self.numberOfSegments):
                if self.dataSegmentList[k]['MLready'] == 'no':
                    self.dataSegmentList[k] = self.createMLset_unknown(self.dataSegmentList[k])
                    self.dataSegmentList[k]['MLready'] = 'yes'

    def SaveData(self, fileName, folderPath):

        dataDump = Dump_File(self.settings["Sumitomo_Long_Short"])
        dataDump.push(self, "segments")
        dataDump.push(self.settings, "settings")

        WU_MEG_DP_lib.Save_Object(dataDump, fileName, folderPath)

    def PrintSegments(self):
        rangeList = []
        for dataSegment in self.dataSegmentList:
            rangeList.append(dataSegment['range'])
        print(rangeList)


class Sumitomo_Decoding_Short(Sumitomo_Machine):
    def __init__(self, idStartInd=3, idEndInd=6, dateStartInd=7, dateEndInd=13,
                 fixEventSequence='yes', exposureTime=300, exposure_err=0.15):
        super().__init__()

        self.idStartInd = idStartInd
        self.idEndInd = idEndInd
        self.dateStartInd = dateStartInd
        self.dateEndInd = dateEndInd
        self.fixEventSequence = fixEventSequence
        self.exposureTime = exposureTime
        self.exposure_err = exposure_err
        self.experimentType = None

    def data_Concatenation_Short(self, DataFileNames, inDataDir):

        raw_file = []
        raw_fileR = []
        for name in DataFileNames:
            raw_it = mne.io.read_raw_fif(join(inDataDir, name), preload=True)

            data = raw_it.get_data()
            if self.experimentType == "NRRN":
                if name[-10:-8] == "01" or name[-10:-8] == "04":
                    data[67, :] = 0
                    raw_it = mne.io.RawArray(data, raw_it.info)
                    raw_file.append(raw_it)
                elif name[-10:-8] == "02" or name[-10:-8] == "03":
                    data[67, :] = 1
                    raw_it = mne.io.RawArray(data, raw_it.info)
                    raw_fileR.append(raw_it)
                else:
                    raise Exception('Unable to configure event type.')

            elif self.experimentType == "RNNR":
                if name[-10:-8] == "01" or name[-10:-8] == "04":
                    data[67, :] = 1
                    raw_it = mne.io.RawArray(data, raw_it.info)
                    raw_fileR.append(raw_it)
                elif name[-10:-8] == "02" or name[-10:-8] == "03":
                    data[67, :] = 0
                    raw_it = mne.io.RawArray(data, raw_it.info)
                    raw_file.append(raw_it)
                else:
                    raise Exception('Unable to configure event type.')
        raw_file.extend(raw_fileR)

        if DataFileNames.__len__() > 1:
            raw = concatenate_raws(raw_file)
        else:
            raw = raw_it

        return raw

    def PrepareRawData(self, inDataDir, DataFileNames, outDataDir):

        # Input Check
        fileNameGroup = []
        if not os.path.exists(inDataDir):
            raise Exception('Read directory does not exist.')
        elif DataFileNames != "all":
            subNum = DataFileNames[0][self.idStartInd:self.idEndInd]
            expDate = DataFileNames[0][self.dateStartInd:self.dateEndInd]
            for name in DataFileNames:
                if not os.path.exists(join(inDataDir, name)):
                    raise Exception('At least one read file does not exist.')
                else:
                    if not subNum == name[self.idStartInd:self.idEndInd]:
                        warn("Concatenated data is from different subjects!")
                    elif not expDate == name[self.dateStartInd:self.dateEndInd]:
                        warn("Concatenated data has different date stamps!")
            fileNameGroup.append(DataFileNames)
        else:
            fileList = []
            for file in os.listdir(inDataDir):
                if file.endswith(".fif"):
                    fileList.append(file)

            while fileList != []:
                fileName = fileList[0]
                singleGroup = []
                for k in range(fileList.__len__()):
                    if fileName[self.idStartInd:self.dateEndInd] == fileList[k][self.idStartInd:self.dateEndInd]:
                        singleGroup.append(fileList[k])

                fileNameGroup.append(singleGroup)
                for fName in fileNameGroup[-1]:
                    fileList.remove(fName)

        if not os.path.exists(outDataDir):
            os.mkdir(outDataDir)

        for DataFileNames in fileNameGroup:

            subID = DataFileNames[0][self.idStartInd]
            if subID == "3":
                # Odd
                self.experimentType = "NRRN"
            elif subID == "4":
                # Even
                self.experimentType = "RNNR"
            else:
                raise Exception('Subject ID is not recognized. Expected 3XX or 4XX, got ' + subID + ' instead.')

            # Data Concatenation
            raw = self.data_Concatenation_Short(DataFileNames, inDataDir)

            # Data Normalization
            print("Normalization ON: \n\t Performing data normalization.")
            if self.DeviceNormalization == 'yes':
                raw = self.dataNormalization(raw)

            # Initialization
            raw = self.deviceInit(raw)
            print("Coil Type: ", raw.info["chs"][0]["coil_type"])

            # Event Sequence Correction
            if self.fixEventSequence == 'yes':
                data3 = raw.get_data()
                if (np.max(data3[66]) - np.min(data3[66])) < 1:
                    data3[66] = -10 ** 4 * data3[66]

                    baseLine = np.median(data3[66])
                    if baseLine < 0:
                        st_rat = np.abs(np.median(data3[66])) / (np.max(data3[66]) - np.median(data3[66]))
                        data3[66] = data3[66] * (1 + st_rat)
                    time_index = np.where(data3[66] < 0.25)
                    data3[66][time_index] = 0.0

                for k in range(64, 67):
                    baseLine = np.median(data3[k])
                    data3[k] = np.abs(data3[k] - baseLine)

                raw = mne.io.RawArray(data3, raw.info)

            # Event channels rectification
            raw = self.eventChannelsRectification(raw)

            outFileName = DataFileNames[0][:-10] + "00_raw.fif"
            raw.save(join(outDataDir, outFileName), overwrite=True)

    def eventConditions(self, filt, events_stim1, events_stim2, events_stim3, events_stim4,
                        tmin, tmax, preload, baseline, grad):
        printFlag = 1
        printNum = None
        events_conditions = {}
        epochs_conditions = {}
        reject = dict(grad=grad)  # T / m (gradiometers)
        event_log = Event_Log_Sumitomo(["STIM3", "STIM2", "STIM1", "MNE-All", "MNE-Norm", "MNE-Rev"])

        # Set 1: 0 0 1
        events_condition001 = np.asarray(
            [x for x in events_stim1 if (x[0] not in events_stim2[:, 0]) and (x[0] not in events_stim3[:, 0])])
        events_conditions['001'] = events_condition001
        if events_conditions['001'].__len__() > 0:
            events_conditions['001N'] = events_condition001[events_condition001[:, 0] < events_stim4[0, 0], :]
            events_conditions['001R'] = events_condition001[events_condition001[:, 0] >= events_stim4[0, 0], :]
        else:
            events_conditions['001N'] = events_conditions['001']
            events_conditions['001R'] = events_conditions['001']

        print('Set 1 - Image1     :      0        0       1      ', events_condition001.shape[0])
        event_log.addEntry("Image1", ["0", "0", "1",
                                      str(events_conditions['001'].shape[0]),
                                      str(events_conditions['001N'].shape[0]),
                                      str(events_conditions['001R'].shape[0])])
        if events_condition001.__len__() > 0:
            epochs_conditions['001'] = mne.Epochs(raw=filt, events=events_conditions['001'], tmin=tmin, tmax=tmax,
                                                   preload=preload, baseline=baseline, reject=reject)
            epochs_conditions['001N'] = mne.Epochs(raw=filt, events=events_conditions['001N'], tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            epochs_conditions['001R'] = mne.Epochs(raw=filt, events=events_conditions['001R'], tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)

            if printFlag:
                prtMessage = epochs_conditions['001N'].info['sfreq']
                printFlag = 0
                printNum = '001N'
        else:
            epochs_conditions['001'] = None
            epochs_conditions['001N'] = None
            epochs_conditions['001R'] = None

        # Set 2: 0 1 0
        events_condition010 = np.asarray(
            [x for x in events_stim2 if (x[0] not in events_stim1[:, 0]) and (x[0] not in events_stim3[:, 0])])
        events_conditions['010'] = events_condition010
        if events_conditions['010'].__len__() > 0:
            events_conditions['010N'] = events_condition010[events_condition010[:, 0] < events_stim4[0, 0], :]
            events_conditions['010R'] = events_condition010[events_condition010[:, 0] >= events_stim4[0, 0], :]
        else:
            events_conditions['010N'] = events_conditions['010']
            events_conditions['010R'] = events_conditions['010']

        print('Set 2 - Image2:      0        1       0      ', events_condition010.shape[0])
        event_log.addEntry("Image2", ["0", "1", "0",
                                      str(events_conditions['010'].shape[0]),
                                      str(events_conditions['010N'].shape[0]),
                                      str(events_conditions['010R'].shape[0])])
        if events_condition010.__len__() > 0:
            epochs_conditions['010'] = mne.Epochs(raw=filt, events=events_conditions['010'], tmin=tmin, tmax=tmax,
                                                   preload=preload, baseline=baseline, reject=reject)
            epochs_conditions['010N'] = mne.Epochs(raw=filt, events=events_conditions['010N'], tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            epochs_conditions['010R'] = mne.Epochs(raw=filt, events=events_conditions['010R'], tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)

            if printFlag:
                prtMessage = epochs_conditions['010N'].info['sfreq']
                printFlag = 0
                printNum = '010N'
        else:
            epochs_conditions['010'] = None
            epochs_conditions['010N'] = None
            epochs_conditions['010R'] = None

        # Set 3: 0 1 1
        events_condition011 = np.asarray(
            [x for x in events_stim1 if (x[0] in events_stim2[:, 0]) and (x[0] not in events_stim3[:, 0])])
        events_conditions['011'] = events_condition011
        if events_conditions['011'].__len__() > 0:
            events_conditions['011N'] = events_condition011[events_condition011[:, 0] < events_stim4[0, 0], :]
            events_conditions['011R'] = events_condition011[events_condition011[:, 0] >= events_stim4[0, 0], :]
        else:
            events_conditions['011N'] = events_conditions['011']
            events_conditions['011R'] = events_conditions['011']

        print('Set 3 - Image3:      0        1       1      ', events_condition011.shape[0])
        event_log.addEntry("Image3", ["0", "1", "1",
                                      str(events_conditions['011'].shape[0]),
                                      str(events_conditions['011N'].shape[0]),
                                      str(events_conditions['011R'].shape[0])])
        if events_condition011.__len__() > 0:
            epochs_conditions['011'] = mne.Epochs(raw=filt, events=events_conditions['011'], tmin=tmin, tmax=tmax,
                                                   preload=preload, baseline=baseline, reject=reject)
            epochs_conditions['011N'] = mne.Epochs(raw=filt, events=events_conditions['011N'], tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            epochs_conditions['011R'] = mne.Epochs(raw=filt, events=events_conditions['011R'], tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)

            if printFlag:
                prtMessage = epochs_conditions['011N'].info['sfreq']
                printFlag = 0
                printNum = '011N'
        else:
            epochs_conditions['011'] = None
            epochs_conditions['011N'] = None
            epochs_conditions['011R'] = None

        # Set 4: 1 0 0
        events_condition100 = np.asarray(
            [x for x in events_stim3 if (x[0] not in events_stim1[:, 0]) and (x[0] not in events_stim2[:, 0])])
        events_conditions['100'] = events_condition100
        if events_conditions['100'].__len__() > 0:
            events_conditions['100N'] = events_condition100[events_condition100[:, 0] < events_stim4[0, 0], :]
            events_conditions['100R'] = events_condition100[events_condition100[:, 0] >= events_stim4[0, 0], :]
        else:
            events_conditions['100N'] = events_conditions['100']
            events_conditions['100R'] = events_conditions['100']

        print('Set 4 - Image4:      1        0       0      ', events_condition100.shape[0])
        event_log.addEntry("Image4", ["1", "0", "0",
                                      str(events_conditions['100'].shape[0]),
                                      str(events_conditions['100N'].shape[0]),
                                      str(events_conditions['100R'].shape[0])])
        if events_condition100.__len__() > 0:
            epochs_conditions['100'] = mne.Epochs(raw=filt, events=events_conditions['100'], tmin=tmin, tmax=tmax,
                                                   preload=preload, baseline=baseline, reject=reject)
            epochs_conditions['100N'] = mne.Epochs(raw=filt, events=events_conditions['100N'], tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            epochs_conditions['100R'] = mne.Epochs(raw=filt, events=events_conditions['100R'], tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)

            if printFlag:
                prtMessage = epochs_conditions['100N'].info['sfreq']
                printFlag = 0
                printNum = '100N'
        else:
            epochs_conditions['100'] = None
            epochs_conditions['100N'] = None
            epochs_conditions['100R'] = None

        # Set 5: 1 0 1
        events_condition101 = np.asarray(
            [x for x in events_stim1 if (x[0] not in events_stim2[:, 0]) and (x[0] in events_stim3[:, 0])])
        events_conditions['101'] = events_condition101
        if events_conditions['101'].__len__() > 0:
            events_conditions['101N'] = events_condition101[events_condition101[:, 0] < events_stim4[0, 0], :]
            events_conditions['101R'] = events_condition101[events_condition101[:, 0] >= events_stim4[0, 0], :]
        else:
            events_conditions['101N'] = events_conditions['101']
            events_conditions['101R'] = events_conditions['101']

        print('Set 5 - Image5:      1        0       1      ', events_condition101.shape[0])
        event_log.addEntry("Image5", ["1", "0", "1",
                                      str(events_conditions['101'].shape[0]),
                                      str(events_conditions['101N'].shape[0]),
                                      str(events_conditions['101R'].shape[0])])
        if events_condition101.__len__() > 0:
            epochs_conditions['101'] = mne.Epochs(raw=filt, events=events_conditions['101'], tmin=tmin, tmax=tmax,
                                                   preload=preload, baseline=baseline, reject=reject)
            epochs_conditions['101N'] = mne.Epochs(raw=filt, events=events_conditions['101N'], tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            epochs_conditions['101R'] = mne.Epochs(raw=filt, events=events_conditions['101R'], tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)

            if printFlag:
                prtMessage = epochs_conditions['101N'].info['sfreq']
                printFlag = 0
                printNum = '101N'
        else:
            epochs_conditions['101'] = None
            epochs_conditions['101N'] = None
            epochs_conditions['101R'] = None

        # Set 6: 1 1 0
        events_condition110 = np.asarray(
            [x for x in events_stim2 if (x[0] not in events_stim1[:, 0]) and (x[0] in events_stim3[:, 0])])
        events_conditions['110'] = events_condition110
        if events_conditions['110'].__len__() > 0:
            events_conditions['110N'] = events_condition110[events_condition110[:, 0] < events_stim4[0, 0], :]
            events_conditions['110R'] = events_condition110[events_condition110[:, 0] >= events_stim4[0, 0], :]
        else:
            events_conditions['110N'] = events_conditions['110']
            events_conditions['110R'] = events_conditions['110']

        print('Set 6 - Image6:      1        1       0      ', events_condition110.shape[0])
        event_log.addEntry("Image6", ["1", "1", "0",
                                      str(events_conditions['110'].shape[0]),
                                      str(events_conditions['110N'].shape[0]),
                                      str(events_conditions['110R'].shape[0])])
        if events_condition110.__len__() > 0:
            epochs_conditions['110'] = mne.Epochs(raw=filt, events=events_conditions['110'], tmin=tmin, tmax=tmax,
                                                   preload=preload, baseline=baseline, reject=reject)
            epochs_conditions['110N'] = mne.Epochs(raw=filt, events=events_conditions['110N'], tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            epochs_conditions['110R'] = mne.Epochs(raw=filt, events=events_conditions['110R'], tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)

            if printFlag:
                prtMessage = epochs_conditions['110N'].info['sfreq']
                printFlag = 0
                printNum = '110N'
        else:
            epochs_conditions['110'] = None
            epochs_conditions['110N'] = None
            epochs_conditions['110R'] = None

        # Set 7: 1 1 1
        events_condition111 = np.asarray(
            [x for x in events_stim1 if (x[0] in events_stim2[:, 0]) and (x[0] in events_stim3[:, 0])])
        events_conditions['111'] = events_condition111
        if events_conditions['111'].__len__() > 0:
            events_conditions['111N'] = events_condition111[events_condition111[:, 0] < events_stim4[0, 0], :]
            events_conditions['111R'] = events_condition111[events_condition111[:, 0] >= events_stim4[0, 0], :]
        else:
            events_conditions['111N'] = events_conditions['111']
            events_conditions['111R'] = events_conditions['111']

        print('Set 7 - Image7:      1        1       1      ', events_condition111.shape[0])
        event_log.addEntry("Image7", ["1", "1", "1",
                                      str(events_conditions['111'].shape[0]),
                                      str(events_conditions['111N'].shape[0]),
                                      str(events_conditions['111R'].shape[0])])
        if events_condition111.__len__() > 0:
            epochs_conditions['111'] = mne.Epochs(raw=filt, events=events_conditions['111'], tmin=tmin, tmax=tmax,
                                                   preload=preload, baseline=baseline, reject=reject)
            epochs_conditions['111N'] = mne.Epochs(raw=filt, events=events_conditions['111N'], tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)
            epochs_conditions['111R'] = mne.Epochs(raw=filt, events=events_conditions['111R'], tmin=tmin, tmax=tmax,
                                             preload=preload, baseline=baseline, reject=reject)

            if printFlag:
                prtMessage = epochs_conditions['111N'].info['sfreq']
                printFlag = 0
                printNum = '111N'
        else:
            epochs_conditions['111'] = None
            epochs_conditions['111N'] = None
            epochs_conditions['111R'] = None

        events_conditions['log'] = event_log
        epochs_conditions['log'] = event_log

        return events_conditions, epochs_conditions, printNum, prtMessage

    def ResampleEpochsGroup(self, epochs_manager, new_freq):
        if epochs_manager['001N'].epochs_condition is not None:
            epochs_manager['001N'].epochs_condition.resample(new_freq)
            epochs_manager['001R'].epochs_condition.resample(new_freq)

        if epochs_manager['010N'].epochs_condition is not None:
            epochs_manager['010N'].epochs_condition.resample(new_freq)
            epochs_manager['010R'].epochs_condition.resample(new_freq)

        if epochs_manager['011N'].epochs_condition is not None:
            epochs_manager['011N'].epochs_condition.resample(new_freq)
            epochs_manager['011R'].epochs_condition.resample(new_freq)

        if epochs_manager['100N'].epochs_condition is not None:
            epochs_manager['100N'].epochs_condition.resample(new_freq)
            epochs_manager['100R'].epochs_condition.resample(new_freq)

        if epochs_manager['101N'].epochs_condition is not None:
            epochs_manager['101N'].epochs_condition.resample(new_freq)
            epochs_manager['101R'].epochs_condition.resample(new_freq)

        if epochs_manager['110N'].epochs_condition is not None:
            epochs_manager['110N'].epochs_condition.resample(new_freq)
            epochs_manager['110R'].epochs_condition.resample(new_freq)

        if epochs_manager['111N'].epochs_condition is not None:
            epochs_manager['111N'].epochs_condition.resample(new_freq)
            epochs_manager['111R'].epochs_condition.resample(new_freq)

    def EpockingProcedures(self, raw, resemple='yes', new_freq=500, tmin=-0.2, tmax=0.8,
                           baseline=(-0.2, 0), min_event_duration=0.1, grad=7000e-13, preload=True):

        filt = raw.copy()
        # Event Extraction
        events_stim1 = mne.find_events(filt, stim_channel='STIM1', min_duration=min_event_duration)
        events_stim2 = mne.find_events(filt, stim_channel='STIM2', min_duration=min_event_duration)
        events_stim3 = mne.find_events(filt, stim_channel='STIM3', min_duration=min_event_duration)
        events_stim4 = mne.find_events(filt, stim_channel='STIM4', min_duration=min_event_duration)

        # Conditions: STIM1 STIM2 STIM3
        events_conditions, epochs_conditions, printNum, prtMessage = \
            self.eventConditions(filt, events_stim1, events_stim2, events_stim3, events_stim4,
                                 tmin, tmax, preload, baseline, grad)

        eventLog = epochs_conditions['log']
        epochs_manager = {}
        epochs_manager['log'] = eventLog

        epochs_manager['001'] = DropEpochsManager(epochs_conditions['001'])
        epochs_manager['010'] = DropEpochsManager(epochs_conditions['010'])
        epochs_manager['011'] = DropEpochsManager(epochs_conditions['011'])
        epochs_manager['100'] = DropEpochsManager(epochs_conditions['100'])
        epochs_manager['101'] = DropEpochsManager(epochs_conditions['101'])
        epochs_manager['110'] = DropEpochsManager(epochs_conditions['110'])
        epochs_manager['111'] = DropEpochsManager(epochs_conditions['111'])

        epochs_manager['001N'] = DropEpochsManager(epochs_conditions['001N'])
        epochs_manager['010N'] = DropEpochsManager(epochs_conditions['010N'])
        epochs_manager['011N'] = DropEpochsManager(epochs_conditions['011N'])
        epochs_manager['100N'] = DropEpochsManager(epochs_conditions['100N'])
        epochs_manager['101N'] = DropEpochsManager(epochs_conditions['101N'])
        epochs_manager['110N'] = DropEpochsManager(epochs_conditions['110N'])
        epochs_manager['111N'] = DropEpochsManager(epochs_conditions['111N'])

        epochs_manager['001R'] = DropEpochsManager(epochs_conditions['001R'])
        epochs_manager['010R'] = DropEpochsManager(epochs_conditions['010R'])
        epochs_manager['011R'] = DropEpochsManager(epochs_conditions['011R'])
        epochs_manager['100R'] = DropEpochsManager(epochs_conditions['100R'])
        epochs_manager['101R'] = DropEpochsManager(epochs_conditions['101R'])
        epochs_manager['110R'] = DropEpochsManager(epochs_conditions['110R'])
        epochs_manager['111R'] = DropEpochsManager(epochs_conditions['111R'])

        if resemple == 'yes':
            self.ResampleEpochsGroup(epochs_manager, new_freq)
            print('Original sampling rate:  ', prtMessage, 'Hz')
            print('New sampling rate:       ', epochs_manager[printNum].epochs_condition.info['sfreq'], 'Hz')

        return events_conditions, epochs_manager, eventLog

    def PrepareEnvokedConditions(self, epochs_manager, eventLog):

        evoked_conditions = {}
        evoked_conditions['log'] = eventLog

        if epochs_manager['001'].epochs_condition is not None:
            evoked_conditions['001'] = epochs_manager['001'].epochs_condition.average()
            evoked_conditions['001N'] = epochs_manager['001N'].epochs_condition.average()
            evoked_conditions['001R'] = epochs_manager['001R'].epochs_condition.average()
        else:
            evoked_conditions['001'] = None
            evoked_conditions['001N'] = None
            evoked_conditions['001R'] = None

        if epochs_manager['010'].epochs_condition is not None:
            evoked_conditions['010'] = epochs_manager['010'].epochs_condition.average()
            evoked_conditions['010N'] = epochs_manager['010N'].epochs_condition.average()
            evoked_conditions['010R'] = epochs_manager['010R'].epochs_condition.average()
        else:
            evoked_conditions['010'] = None
            evoked_conditions['010N'] = None
            evoked_conditions['010R'] = None

        if epochs_manager['011'].epochs_condition is not None:
            evoked_conditions['011'] = epochs_manager['011'].epochs_condition.average()
            evoked_conditions['011N'] = epochs_manager['011N'].epochs_condition.average()
            evoked_conditions['011R'] = epochs_manager['011R'].epochs_condition.average()
        else:
            evoked_conditions['011'] = None
            evoked_conditions['011N'] = None
            evoked_conditions['011R'] = None

        if epochs_manager['100'].epochs_condition is not None:
            evoked_conditions['100'] = epochs_manager['100'].epochs_condition.average()
            evoked_conditions['100N'] = epochs_manager['100N'].epochs_condition.average()
            evoked_conditions['100R'] = epochs_manager['100R'].epochs_condition.average()
        else:
            evoked_conditions['100'] = None
            evoked_conditions['100N'] = None
            evoked_conditions['100R'] = None

        if epochs_manager['101'].epochs_condition is not None:
            evoked_conditions['101'] = epochs_manager['101'].epochs_condition.average()
            evoked_conditions['101N'] = epochs_manager['101N'].epochs_condition.average()
            evoked_conditions['101R'] = epochs_manager['101R'].epochs_condition.average()
        else:
            evoked_conditions['101'] = None
            evoked_conditions['101N'] = None
            evoked_conditions['101R'] = None

        if epochs_manager['110'].epochs_condition is not None:
            evoked_conditions['110'] = epochs_manager['110'].epochs_condition.average()
            evoked_conditions['110N'] = epochs_manager['110N'].epochs_condition.average()
            evoked_conditions['110R'] = epochs_manager['110R'].epochs_condition.average()
        else:
            evoked_conditions['110'] = None
            evoked_conditions['110N'] = None
            evoked_conditions['110R'] = None

        if epochs_manager['111'].epochs_condition is not None:
            evoked_conditions['111'] = epochs_manager['111'].epochs_condition.average()
            evoked_conditions['111N'] = epochs_manager['111N'].epochs_condition.average()
            evoked_conditions['111R'] = epochs_manager['111R'].epochs_condition.average()
        else:
            evoked_conditions['111'] = None
            evoked_conditions['111N'] = None
            evoked_conditions['111R'] = None

        return evoked_conditions

    def PrepareStage1Data(self, Settings):
        inDataDir = Settings["inDataDir"]
        DataFileNames = Settings["DataFileNames"]
        outDataDir = Settings["outDataDir"]

        # Input Check
        fileList = []
        if not os.path.exists(inDataDir):
            raise Exception('Read directory does not exist.')

        elif DataFileNames != "all":
            for name in DataFileNames:
                if not os.path.exists(join(inDataDir, name)):
                    raise Exception('File does not exist in the specified folder.')
            fileList.extend(DataFileNames)

        else:
            for file in os.listdir(inDataDir):
                if file.endswith(".fif"):
                    fileList.append(file)

        if not os.path.exists(outDataDir):
            os.mkdir(outDataDir)

        # Print current settings
        print(pd.DataFrame.from_dict(Settings, orient='index', columns=["Settings"]))

        for name in fileList:
            raw = mne.io.read_raw_fif(join(Settings["inDataDir"], name), preload=Settings["preload"])

            # Perform BPF and Detrending on the raw data.
            if Settings["RawDataFiltration"] == 'yes':
                filt = self.FilterRawData_BPF(raw, cutoff_l=Settings["cutoff_l"], cutoff_h=Settings["cutoff_h"])
                filt = self.FilterRawData_DT(filt, DT=Settings["DT"], DT_param=Settings["DT_param"])
            else:
                filt = raw.copy()

            # Preparation of events and epochs for further analysis.
            events_conditions, epochs_manager, eventLog = \
                self.EpockingProcedures(filt, resemple=Settings["resemple"], new_freq=Settings["new_freq"],
                                      tmin=Settings["tmin"], tmax=Settings["tmax"], baseline=Settings["baseline"],
                                      min_event_duration=Settings["min_event_duration"], grad=Settings["grad"],
                                      preload=Settings["preload"])

            # Print current logfile
            print('Data file: ' + name)
            eventLog.printLog()

            # Create data Dump object
            dataDump = Dump_File(Settings["Sumitomo_Long_Short"])
            # Fill Dump object
            dataDump.push(Settings, "settings")
            if Settings["fullDump"] == 'yes':
                dataDump.push(events_conditions, "events")
            dataDump.push(epochs_manager, "epochs")

            outFileName = name[:Settings["idEndInd"]] + "_Stage1.pkl"
            self.Save_Object(dataDump, outFileName, Settings["outDataDir"])



# Electra Device Modules
class Electra_Machine(WU_MEG_DP_lib):
    def __init__(self, DeviceNormalization=0):
        super().__init__()

        self.DeviceNormalization = DeviceNormalization


class Electra_Prediction(Electra_Machine):
    def __init__(self):
        super().__init__()

