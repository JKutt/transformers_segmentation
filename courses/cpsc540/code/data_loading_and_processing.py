# ---------------------------------------------------------------------------

# Code snippit for loading dciptimeseries data

# ----------------------------------------------------------------------------

from scipy import signal as sig
from datfiles_lib_parallel import *
from os import listdir
dir_path_dc = r"C:\Users\johnk\Documents\projects\ubc\cpsc540\projects\report\johnsift\\"
only_files = [f for f in listdir(dir_path_dc)]     # gets files # remember 8
# print(only_files)
idx = 0

test_data = []

for idx in range(len(only_files)):
    
    if only_files[idx].split(".")[-1] == "DAT":
        
        node = dir_path_dc + only_files[idx]
#         print(node)
        fIn = open(node, 'r', encoding="utf8", errors=""'ignore')
        linesFIn = fIn.readlines()
        fIn.close()

        time, data = read_data(linesFIn)
        
        # stack the data ===================================================
        num_half_T = int(np.floor(data.size / 600))
        new_trim = int(num_half_T * 600)
        xt_data = data[:new_trim]
        
        xt_data = np.reshape(xt_data, (600, num_half_T))
        
        test_data += [xt_data]



# ---------------------------------------------------------------------------

# Code snippit for processing dciptimeseries data

# ----------------------------------------------------------------------------

import EMtools as DCIP
import matplotlib.pyplot as plt

def getTime():
    timeFrom = [2040., 2060., 2080., 2120., 2160., 2200.,
                2240., 2320., 2400.,
                2480., 2560., 2640.,
                2720., 2800., 2960.,
                3120., 3280., 3440.,
                3600., 3760.]
    timeTo = [2060., 2080., 2120., 2160., 2200., 2240.,
              2320., 2400., 2480., 2560., 2640., 2720.,
              2800., 2960., 3120., 3280., 3440.,
              3600., 3760., 3920.]
    return timeFrom, timeTo

# ----------------------------------------------------------------------------

# stack the data 

#

timeFrom, timeTo = getTime()

mid_time = (np.asarray(timeTo) + np.asarray(timeFrom)) / 2

num_half_T = np.floor(data.size / 600)
new_trim = int(num_half_T * 600)
xt = y_notched[:new_trim]
xt_data = data[:new_trim]

print(num_half_T, new_trim, xt.size)
start_vp = 50                           # start of Vp calculation (%)
end_vp = 90                             # end of Vp calculation (%)
window = DCIP.createHanningWindow(num_half_T)   # creates filter window
# window = DCIP.createChebyshevWindow(int(num_half_T), 500)
window2 = DCIP.createKaiserWindow(int(num_half_T), 150)

tHK = DCIP.filterKernel(filtershape=window)     # creates filter kernal
tHK2 = DCIP.filterKernel(filtershape=window)     # creates filter kernal
print("half T: {0} window: {1} Kernel: {2}".format(num_half_T, window.size, tHK.kernel.size))
# print(xt.size)
# # eHK = DCIP.ensembleKernal(filtershape=window3,
# #                           number_half_periods=num_half_T)
dkernal = DCIP.decayKernel(num_windows=np.asarray(timeTo).size,
                           window_starts=np.asarray(timeFrom),
                           window_ends=np.asarray(timeTo),
                           window_weight=501,
                           window_overlap=0.99,
                           output_type="Vs")  # creates decay kernal
stack = tHK * xt                               # stack data
stack2 = tHK * xt_data                               # stack data

decay = dkernal * (tHK * xt)         # calculates the decay
decay2 = dkernal * (tHK * xt_data)         # calculates the decay

# ----------------------------------------------------------------------------

# plot data

#

plt.figure()
plt.plot(stack)
plt.plot(stack2, 'r')
plt.xlabel("stack number")
plt.show()
plt.figure()
plt.plot(mid_time, decay)
plt.plot(mid_time, decay2, 'r')
plt.ylabel("voltage (mV)")
plt.xlabel("off-time (ms)")
plt.show()


# plt.plot(np.abs(DCIP.getFrequnceyResponse(window)))
# plt.show()