import datfiles_lib_parallel as datlib
import signalAnalysis_lib_parallel as signal
import multiprocessing
import glob
import numpy as np
from itertools import product
from classes import Record, Node
import utm
import time
import shutil
import matplotlib.pylab as plt
import io_lib as io


def worker(num):
    """worker function"""
    print('Worker', num)

def get_content_folder(project):

    log_file = project.open_friday_log()

    nodeList = datlib.list_nodes(project.data_path)
    log_file.write("Checking node library for missing files and empty folders\n")
    logFile = open(project.path + 'checkDataFolder.log', 'w')
    #if len(glob(project.path + nodeMessagesFile)):
    #    print("Using the node messages file to build a library from acquisition messages")
    #    struct, nodeId = datlib.read_node_messages(dataRoot + nodeMessagesFile)
    #else:
    struct = []
    nodeId = []
    for node in nodeList:
        logFile.write("Checking node: " + node + "\n")
        if node in nodeId:
            ind = nodeId.index(node)
            if len(ind):
                list_files = struct[ind]
            else:
                list_files = []
        logFile.write("Checking node: " + node + '\n')
        datNode = datlib.get_list_file(project.data_path, node)
        datlib.check_list_files(datNode, logFile)
    print("Done! You can now check the file " + project.path + "checkDataFolder.log")
    logFile.close()

    project.close_friday_log(log_file)

    return nodeList

def get_records(project):

    log_file = project.open_friday_log()
    # Checking if previous library file is existing
    if len(glob.glob(project.path + 'nodeLibrary.dat')):
        records = datlib.read_library_file(project.path + 'nodeLibrary.dat')
        records, project.list_nodes = datlib.add_new_nodes(records, project.list_nodes)
    else:
        records = [[] for i in range(len(project.list_nodes))]

    # Reading recording file
    injections = datlib.read_log_file(project.recording_file)

    # Using a pool of worker to read each node folder
    max_number_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(max_number_processes - 2)
    r = []
    print("\tStarting workers pool: " + str(max_number_processes - 2) + " workers reading files.")

    for i in range(len(project.list_nodes)):
        #res = pool.map(datlib.read_node_library, )
        # r.append(datlib.read_node_library(project.data_path, project.list_nodes[i], records, i))
        r.append(pool.apply_async(datlib.read_node_library, (project.data_path, project.list_nodes[i], records, i)))
    pool.close()
    pool.join()
    for i in range(len(r)):
        try:
            tmp = r[i].get()
            records[i] = tmp[0]
            project.write_messages_to_friday(log_file, tmp[1])
        except:
            print('[WARNING] file read issue likely')
            pass
    datlib.make_library_file(project.list_nodes, records, project.path)
    # Comparing recording file and list of records
    for injection in injections:
        for nodeIndex in range(len(project.list_nodes)):
            for record in records[nodeIndex]:
                test = datlib.is_node_active(injection, record)
                if test:
                    try:
                        tmp = record.getUtm()
                        injection.list_nodes.append(record.node_id)
                        injection.list_gps.append([tmp[0], tmp[1], record.altitude])
                        if record.node_type == 'C':
                            injection.list_type.append('C')
                        else:
                            injection.list_type.append(record.relay_state)
                        print(record.name)
                        injection.list_files.append(record.name)
                    except:
                        print['[ERROR] LISTGPS']
                        pass
    project.close_friday_log(log_file)

    return records, injections


def harmonic_vp_analysis_injection(project, injection):
    print(injection)
    log_file = project.open_friday_log()
    sample_freq = 150
    sample_half_t = 600.0

    # Creating synthetic current to get frequencies of harmonics
    time_s, data_s = signal.synthetic_current(sample_freq, 2, 16384)
    # Getting frequencies of harmonics

    periods = signal.get_maxima(1,
                                len(data_s),
                                np.linspace(0, 1, len(data_s)) * sample_freq,
                                np.abs(np.fft.fft(data_s)))

    # Small outlier removal
    index_2 = signal.get_frequency_index(np.linspace(0, 1, len(data_s)) * sample_freq, periods)
    ps_values_2 = signal.get_harmonics_amplitude(index_2,
                                                np.linspace(0, 1, len(data_s)) * sample_freq,
                                                np.abs(np.fft.fft(data_s)))
    ps_values_2, periods = signal.remove_outliers(ps_values_2, periods)

    # Initializing arrays for harmonics power and Vp values storage
    nodes = [Node(id="",
             location=[],
             file_name="",
             harmonics=[np.nan for i in range(len(periods))],
             freq_harmonics=[np.nan for i in range(len(periods))],
             Vp=np.nan,
             adf=1,
             score=0,
             coherence=[]) for i in range(len(injection.list_gps))]

    values = [np.nan for i in range(len(injection.list_gps))]
    Vp = [np.nan for i in range(len(injection.list_gps))]
    # Switching to UTM for distances calculations
    utm_coord = [[], []]
    for i in range(len(injection.list_gps)):
        nodes[i].location = injection.list_gps[i][:]

    # Getting on_time from current recorders
    if 'C' in injection.list_type:
        for i in range(len(injection.list_type)):
            if injection.list_type[i] == 'C':
                fIn = open(injection.list_files[i], 'r', encoding="utf8", errors='ignore')
                linesFIn = fIn.readlines()
                fIn.close()
                time_current, data_current = datlib.read_data(linesFIn)
                on_time, index_current = datlib.get_on_time(time_current, data_current, 2, 150)

                variability = signal.get_current_stability(data_current[index_current:], 150, 2)
                moving_variance = signal.moving_variance_signal(data_current, 2, 150)
                if variability[0] > 10 or variability[1] > 10:
                    _, _ = signal.trim_variance(time_current, data_current, 150, 2, moving_variance)
                    print('Warning: Current variation above 10% detected.')
    else:
        on_time = -1
    # Starting loop on node files for each injection
    for i in range(len(injection.list_gps)):
        if injection.list_type[i] == 'A':       # Only getting nodes power
            nodes[i].id = injection.list_nodes[i]
            nodes[i].file_name = injection.list_files[i]


    r = [[] for i in range(len(nodes))]

    max_number_processes = multiprocessing.cpu_count()
    # print("\tStarting workers pool: " + str(max_number_processes - 2) + " workers; " + str(len(nodes)) + " files to process.")
    # pool = multiprocessing.Pool(max_number_processes - 2)            # PARALLEL IMPLEMENTATION

    # index_pool = []
    # for i in range(len(nodes)):
    #     if injection.list_type[i] == 'A':       # Only getting nodes power
    #     # Using a pool of worker to analyze each file
    #         r[i] = pool.apply_async(signal.get_harmonics_and_vp, (nodes[i], on_time, periods, injection))
    #         index_pool.append(i)
    # pool.close()
    # pool.join()
    # """ SERIAL FOR TESTS
    for i in range(len(nodes)):
        if injection.list_type[i] == 'A':       # Only getting nodes power
        # Using a pool of worker to analyze each file
            r[i] = signal.get_harmonics_and_vp(nodes[i], on_time, periods, injection)
    # """
    for ind in range(len(nodes)):
        tmp = r[ind]  #  .get()
        nodes[ind] = tmp[0]
        project.write_messages_to_friday(log_file, tmp[1])
    project.close_friday_log(log_file)
    return nodes


def coherency_analysis(project, injection, nodes):

    log_file = project.open_friday_log()
    sample_freq = 150
    sample_half_t = 600.0

    # Creating synthetic current to get frequencies of harmonics
    time_s, data_s = signal.synthetic_current(sample_freq, 2, 16384)
    # Getting frequencies of harmonics

    periods = signal.get_maxima(1,
                                len(data_s),
                                np.linspace(0, 1, len(data_s)) * sample_freq,
                                np.abs(np.fft.fft(data_s)))

    # Small outlier removal
    index_2 = signal.get_frequency_index(np.linspace(0, 1, len(data_s)) * sample_freq, periods)
    ps_values_2 = signal.get_harmonics_amplitude(index_2,
                                                np.linspace(0, 1, len(data_s)) * sample_freq,
                                                np.abs(np.fft.fft(data_s)))
    ps_values_2, periods = signal.remove_outliers(ps_values_2, periods)

    #dataCouples = [[] for i in range(len(nodes))]
    r = [[] for i in range(len(nodes))]

    max_number_processes = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(max_number_processes - 2)            # PARALLEL IMPLEMENTATION
    # index_pool = []
    # print("\tStarting workers pool: " + str(max_number_processes - 2) + " workers; " + str(len(nodes)) + " files to process.")
    # for i in range(len(nodes)):
    #     if injection.list_type[i] == 'A':       # Only getting nodes power
    #         # Using a pool of worker to analyze each file
    #         r[i] = pool.apply_async(signal.get_coherency, (nodes[i], nodes, periods, injection))
    #         index_pool.append(i)
    # pool.close()
    # pool.join()

    for i in range(len(nodes)):
        if injection.list_type[i] == 'A':       # Only getting nodes power
            r[i] = signal.get_coherency(nodes[i], nodes, periods, injection)

    for ind in range(len(nodes)):
        tmp = r[ind]  # .get()
        nodes[ind].coherence = tmp[0]
        project.write_messages_to_friday(log_file, tmp[1])
        for j in range(len(nodes[ind].coherence)):
            nodes[ind].coherence[j].mem = injection.num
            nodes[ind].coherence[j].node_index.extend([ind])

    project.close_friday_log(log_file)
    return nodes


def statistical_analysis(project, nodes):

    log_file = project.open_friday_log()
    xscore = []
    xaxis = []
    xnumber = []
    index = 0
    for node in nodes:
        score_list = []
        for couple in node.coherence:
            for value in couple.coherency:
                if not np.isnan(value):
                    score_list.append(value)
        if len(score_list):
            node.score = np.sum(score_list) / len(score_list)
            xaxis.append(node.id)
            xnumber.append(index)
            xscore.append(node.score)
            index += 1
        else:
            node.score = np.nan

    tmp = [node.score for node in nodes]

    #fig, (ax1, ax2) = plt.subplots(2, 1)
    #ax1.set_xticks(xnumber, xaxis)
    #ax1.plot(xnumber, [xscore[i] for i in range(len(xscore))], 'ro')

    #ax1.plot([xnumber[i] for i in range(len(xscore)) if np.abs(xscore[i] - np.mean(xscore)) > np.std(xscore) and xscore[i] < np.mean(xscore)],
    #         [xscore[i] for i in range(len(xscore)) if np.abs(xscore[i] - np.mean(xscore)) > np.std(xscore) and xscore[i] < np.mean(xscore)], '*b')

    # First pass
    tested_nodes = []
    for node in nodes:
        if not np.isnan(node.score):
            if np.abs(node.score - np.mean(xscore)) > np.std(xscore) and node.score < np.mean(xscore):
                tested_nodes.append(node.file_name)

    xscore = []
    xaxis = []
    xnumber = []
    index = 0
    # Second pass
    for node in nodes:
        score_list = []
        for couple in node.coherence:
            if couple.couple[1] not in tested_nodes:
                for value in couple.coherency:
                    if not np.isnan(value):
                        score_list.append(value)
        if len(score_list):
            node.score = np.sum(score_list) / len(score_list)
            xaxis.append(node.id)
            xnumber.append(index)
            xscore.append(node.score)
            index += 1
        else:
            node.score = np.nan

    print('Nodes found statistically outliers:')
    message = 'Statistics: Mean coherence: ' + str(np.mean(xscore)) + '; STD: '  + str(np.std(xscore))
    project.write_messages_to_friday(log_file, message)
    message = 'Nodes found statistically outliers:\n'
    project.write_messages_to_friday(log_file, message)

    for i in range(len(nodes)):
        if np.abs(nodes[i].score - np.mean(xscore)) > np.std(xscore) and nodes[i].score < np.mean(xscore):
            message = '\t' + nodes[i].file_name
            project.write_messages_to_friday(log_file, message)
            print('\t', nodes[i].file_name)
            nodes[i].is_quarantine = 1

    #ax2.set_xticks(xnumber, xaxis)
    #ax2.plot(xnumber, [xscore[i] for i in range(len(xscore))], '*r')
    #ax2.plot([xnumber[i] for i in range(len(xscore)) if np.abs(xscore[i] - np.mean(xscore)) > np.std(xscore) and xscore[i] < np.mean(xscore)],
    #         [xscore[i] for i in range(len(xscore)) if np.abs(xscore[i] - np.mean(xscore)) > np.std(xscore) and xscore[i] < np.mean(xscore)], 'ob')
    #plt.show()

    project.close_friday_log(log_file)

    return nodes



def vp_harmonics_check(project, nodes):

    log_file = project.open_friday_log()

    r = [[] for i in range(len(nodes))]
    """
    for i in range(len(nodes)):
        if nodes[i].is_quarantine == 1:       # Only checking outliers
            signal.check_vp_outliers(nodes[i], nodes)
    """
    max_number_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(max_number_processes - 2)            # PARALLEL IMPLEMENTATION
    index_pool = []
    print("\tStarting workers pool: " + str(max_number_processes - 2) + " workers; " + str(len(nodes)) + " files to check.")
    for i in range(len(nodes)):
        if nodes[i].is_quarantine == 1:       # Only checking outliers
            # Using a pool of worker to analyze each file
            r[i] = pool.apply_async(signal.check_vp_outliers, (nodes[i], nodes))
            index_pool.append(i)
    pool.close()
    pool.join()

    for i in range(len(nodes)):
        if nodes[i].is_quarantine == 1:       # Only checking outliers
            tmp = r[i].get()
            nodes[i] = tmp[0]
            project.write_messages_to_friday(log_file, tmp[1])

    project.close_friday_log(log_file)
    return nodes


def quarantine(project, index):

    log_file = project.open_friday_log()

    # creating directory for quarantine
    io.create_quarantine_directory(project)
    # Using a pool of worker to move each file
    max_number_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(max_number_processes - 2)
    r = []
    for node in project.node_data[index]:
        if node.is_quarantine:
            r.append(pool.apply_async(io.quarantine_file, (project, node)))
    pool.close()
    pool.join()

    for message in r:
        project.write_messages_to_friday(log_file, message.get())

    project.close_friday_log(log_file)