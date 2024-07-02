# Abbiamo due dicom di due ecocardigrammi, proviamo ad aprirli

import pydicom
import matplotlib.pyplot as plt

# Si chiamano 1.dcm e 2.dcm
dcm1 = pydicom.dcmread('1.dcm')