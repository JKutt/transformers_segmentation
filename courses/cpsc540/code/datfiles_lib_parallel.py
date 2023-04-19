## IMPORTS
from classes import Record, Injection, Project
import numpy as np
import datetime
from jdcal import gcal2jd, jd2gcal
from os.path import isdir, getsize
from glob import glob
from sys import exit
import matplotlib.pylab as plt
import math
import utm
from scipy.signal import detrend, correlate, convolve
import signalAnalysis_lib_parallel as signalAnalysis
from sys import platform
import multiprocessing

## Written by H. LARNIER, Oct 2018 - DIAS GEOPHYSICAL LTD.

if platform == 'linux1' or platform == 'linux2' or platform == 'darwin':
    separator = '/'
elif platform == 'win32':
    separator = '\\'
else:
    print('Unknown platform')


def read_node_library(path, node, library, num):
    messages = []
    print(multiprocessing.current_process().name + " taking care of node " + node)
    messages.append(multiprocessing.current_process().name + " taking care of node " + node + '\n')
    records_node = []
    datNode = get_list_file(path, node)
    fileIndex = 0
    for datFile in datNode:  # loop on DAT files in folder
        test = is_file_in_library(library, path + node + '/' + datFile, num)
        #test = 1
        if test == 1:
            messages.append(multiprocessing.current_process().name + "\tReading file: " + datFile + "\n")
            fIn = open(path + node + '/' + datFile, 'r', encoding="utf8", errors='ignore')
            linesFIn = fIn.readlines()
            fIn.close()
            info = get_dat_info(linesFIn)               # Getting DAT file info from header
            timeStamp = get_start_end_time(linesFIn)    # Getting time stamp
            gpsLocations = get_average_gps(linesFIn)    # Getting average GPS location from DAT file
            try:
                records_node.append(Record(node_id=info[0],
                                                 node_type=info[3],
                                                 name=path + node + '/' + datFile,
                                                 mem=info[1],
                                                 start_date=timeStamp[0],
                                                 end_date=timeStamp[1],
                                                 relay_state=info[2],
                                                 northing=gpsLocations[0],
                                                 easting=gpsLocations[1],
                                                 altitude=gpsLocations[2]))
            except IndexError:
                messages.append(multiprocessing.current_process().name + " ERROR detected for file: " + datFile + '\n')
                pass
        else:
            messages.append(multiprocessing.current_process().name + "\t File: " + datFile + " is already in the library" + "\n")

        fileIndex += 1
    records_node.extend(library[num])

    return records_node, messages


def get_julian_datetime(date):
    """
    Convert a datetime object into julian float.
    Args:
        date: datetime-object of date in question

    Returns: float - Julian calculated datetime.
    Raises:
        TypeError : Incorrect parameter type
        ValueError: Date out of range of equation
    """

    # Ensure correct format
    if not isinstance(date, datetime.datetime):
        raise TypeError('Invalid type for parameter "date" - expecting datetime')
    elif date.year < 1801 or date.year > 2099:
        raise ValueError('Datetime must be between year 1801 and 2099')

    # Perform the calculation
    julian_datetime = 367 * date.year - int((7 * (date.year + int((date.month + 9) / 12.0))) / 4.0) + int(
        (275 * date.month) / 9.0) + date.day + 1721013.5 + (
                          date.hour + date.minute / 60.0 + date.second / math.pow(60,
                                                                                  2)) / 24.0 - 0.5 * math.copysign(
        1, 100 * date.year + date.month - 190002.5) + 0.5

    return julian_datetime

def gps_from_minutes_to_decimal(gps_value):
    # Get decimal coordinates from decimal minutes
    # Inputs:
    # gps_value: gps coordinates in decimal minutes
    # Output:
    # gps value in decimal value
    degree = np.floor(gps_value)
    tmp = (gps_value - degree) * 100
    minutes = np.floor(tmp)
    seconds = ((tmp - minutes) * 60.)

    return degree + minutes / 60. + seconds / 3600.


def get_list_file(path, node_id):
    # List all dat files for the node "node_id" in the folder "path"
    # Inputs:
    # path: string containing the path to the node to investigate
    # node_id: string containing the node id (two characters)
    # Outputs:
    # dat_list: list of DAT files


    dat_list = glob(path + separator + node_id + separator + '*.DAT')
    to_remove = []
    # Getting empty files
    for i in range(len(dat_list)):
        size = getsize(dat_list[i])
        if size == 0:
            to_remove.append(dat_list[i])

    # Removing empty files
    for i in range(len(to_remove)):
        dat_list.remove(to_remove[i])

    # Only keeping the file name without the path
    for i in range(len(dat_list)):
        tmp = dat_list[i].split(separator)
        dat_list[i] = tmp[-1]

    return dat_list


def check_list_files(list_files, log):
    # Simple checks on the list of files
    # Input:
    # list_files: List of files in node folder
    # log: pointer on log file to write stuff inside
    # Outputs:
    # None
    num = []
    if len(list_files) > 1:
        ## Comparing node_messages_list and file list
        #if len(node_messages_list) > 0:
        #    print()

        # Recovering numbers
        for file in list_files:
            tmp = file.split('.')
            num.append(int(tmp[0][2:]))

        diff = [num[i + 1] - num[i] for i in range(len(num) - 1)]
        if np.max(diff) > 1:
            tmp = [i for i in range(len(diff)) if diff[i] > 1]
            for i in range(len(tmp)):
                log.write('\tWarning: Possible missing files between: ' + str(list_files[tmp[i]]) +
                          ' and ' + str(list_files[tmp[i] + 1]) + '\n')
        else:
            pass
    else:
        log.write('\tWarning: Empty folder' + '\n')
    return


def get_dat_info(lines):
    # Get information of DAT file from header
    # Input:
    # lines: list of lines from DAT file
    # Output:
    # dat_info: list of information (Unit ID, MEM number, Relay state, if current of potential)
    dat_info = ['', 0, '', '', 0., 0.]
    for line in lines:
        if not line[0] == '#':
            break
        else:
            if line[1:5] == 'Unit':
                tmp = line.split(':')
                dat_info[0] = tmp[1][:-1].replace(" ", "")
            if line[1:4] == 'Mem':
                tmp = line.split(':')
                dat_info[1] = int(tmp[1][:-1])
            if line[1:6] == 'Relay':
                tmp = line.split(':')
                relay_state = tmp[1][:-1].replace(" ", "")
                dat_info[2] = relay_state
            if line[1:8] == 'Current':
                dat_info[3] = 'C'
            if line[1:8] == 'Voltage':
                dat_info[3] = 'V'
            if line[1:17] == 'Override Easting':
                tmp = line.split(':')
                dat_info[5] = float(tmp[1][:-1])
            if line[1:18] == 'Override Northing':
                tmp = line.split(':')
                dat_info[4] = float(tmp[1][:-1])
    return dat_info


def get_gps_constellation(lines):
    # Read the GPS constellation from a DAT file
    # Inputs:
    # dat_file: list of lines from read .DAT file
    # Outputs:
    # list of north, west and elevation

    alt = []
    north = []
    west = []
    for line in lines:
        if line[0:6] == '$GNGGA' or line[0:6] == '$GPGGA':
            [test_n, test_w, test_a] = [0, 0, 0]
            tmp = line.split(',')
            try:
                north.append(gps_from_minutes_to_decimal(float(tmp[2]) / 100.))
                if tmp[3] == 'S':
                    north[-1] *= -1
            except ValueError:
                test_n = 1
                pass
            except IndexError:
                test_n = 1
                pass
            try:
                west.append(gps_from_minutes_to_decimal(float(tmp[4]) / 100.))
                if tmp[5] == 'W':
                    west[-1] *= -1
            except ValueError:
                test_w = 1
                pass
            except IndexError:
                test_w = 1
                pass
            try:
                alt.append(float(tmp[9]))
            except ValueError:
                test_a = 1
                pass
            except IndexError:
                test_a = 1
                pass
            if test_n and len(north):
                del north[-1]
            if test_w and len(west):
                del west[-1]
            if test_a and len(alt):
                del alt[-1]

    if not len(north):
        return []
    else:
        return [north, west, alt]


def get_average_gps(lines):
    # Return the median gps location from the constellation
    # Inputs:
    # dat_file: .DAT file to read
    # Outputs:
    # list of average gps [north, west, elev]
    gps_constellation = get_gps_constellation(lines)
    if len(gps_constellation):
        north = np.median(gps_constellation[0])
        west = np.median(gps_constellation[1])
        elev = np.median(gps_constellation[2])
        return [north, west, elev]
    else:
        return []


def get_time(lines):
    # Read the time from a DAT file
    # Inputs:
    # dat_file: .DAT file read
    # Outputs:
    # list of north, west and elevation

    [year, month, day, hour, minute, sec] = [[], [], [], [], [], []]
    time = []
    for line in lines:

        if line[0:7] == '$GNRMC,' or line[0:7] == '$GPRMC,':
            time_tmp = get_date_from_gps_value(line)
            if not time_tmp == datetime.datetime(2000, 1, 1, 1, 1, 1):
                time.append(time_tmp)
            """
            tmp = line.split(',')
            tmp2 = tmp[]
            tmp = tmp[1]
            if len(tmp) == 9 or len(tmp) == 10:
                try:
                    hour_tmp = int(tmp[0:2])
                except ValueError:
                    exit()
                try:
                    minute_tmp = int(tmp[2:4])
                except ValueError:
                    exit()
                try:
                    sec_tmp = int(tmp[4:6])
                except ValueError:
                    exit()
                try:
                    time.append(datetime.datetime(year, month, day, hour_tmp, minute_tmp, sec_tmp))
                except TypeError:
                    pass
                except ValueError:
                    pass
                """
    return time


def get_start_end_time(lines):
    # Return the end and start date from a DAT file
    # Input:
    # lines: lines from a DAT file
    # Output:
    # List of end and start time from the DAT file
    time = get_time(lines)
    try:
        start_end = [time[0], time[-1]]
        return start_end
    except:
        print('Error with the .DAT file.')
        return []


def read_log_file(log_file):
    # Reads the log file and return injection number + time stamp
    # Input:
    # log_file: Name of the log file
    # Output:
    # injection: list of injection (class defined above)
    f_in = open(log_file, 'r')
    lines = f_in.readlines()
    f_in.close()
    injections = []
    for i in range(len(lines)):
        if lines[i][0:5] == 'Start':
            tmp = lines[i].split(':')
            num1 = tmp[1]
            date = tmp[2]
            date = date.split('T')
            tmp1 = date[0].split('-')
            tmp2 = date[1].split(':')
            date_start = datetime.datetime(int(tmp1[0]), int(tmp1[1]), int(tmp1[2]),
                                           int(tmp2[0]), int(tmp[3]), int(tmp[4][0:2]))
            tmp = lines[i + 1].split(':')
            num2 = tmp[1]
            date = tmp[2]
            date = date.split('T')
            tmp1 = date[0].split('-')
            tmp2 = date[1].split(':')
            date_end = datetime.datetime(int(tmp1[0]), int(tmp1[1]), int(tmp1[2]),
                                         int(tmp2[0]), int(tmp[3]), int(tmp[4][0:2]))
            tmp = lines[i + 2].split(':')
            num = tmp[1]
            if num == num1 and num == num2:
                valid = tmp[2][:-1]
                injections.append(Injection(valid=valid,
                                            num=int(num),
                                            start_date=date_start,
                                            end_date=date_end,
                                            list_files=[],
                                            list_current=[],
                                            list_current_gps=[],
                                            list_nodes=[],
                                            list_gps=[],
                                            list_type=[],
                                            list_short=[],
                                            list_short_gps=[])
                                            )
            else:
                injections.append(Injection(valid='Bad',
                                            num=int(num),
                                            start_date=date_start,
                                            end_date=date_end,
                                            list_files=[],
                                            list_current=[],
                                            list_current_gps=[],
                                            list_nodes=[],
                                            list_gps=[],
                                            list_type=[],
                                            list_short=[],
                                            list_short_gps=[])
                                            )

    return injections


def list_nodes(data_folder):
    # Get list of nodes from DATA folder
    # Input:
    # data_folder: String containing the DATA folder location
    # Output:
    # node_list: list of nodes (have to be only two characters long)
    node_list = glob(data_folder + '*')
    not_nodes = []
    for line in node_list:
        if not isdir(line):
            not_nodes.append(line)
    for i in range(len(not_nodes)):
        node_list.remove(not_nodes[i])

    for i in range(len(node_list)):
        tmp = node_list[i].split(separator)
        node_list[i] = tmp[-1]

    return node_list


def is_node_active(inj, rec):
    # Return boolean indicating if the considered node is active during the considered injection
    # Inputs:
    # inj: injection item (class Injection)
    # rec: recording item (class Record)
    # Output:
    # is_active: boolean value indicating if node active during recording or not
    if inj.start_date > rec.end_date or inj.end_date < rec.start_date:
        is_active = False
    else:
        is_active = True

    return is_active


def get_date_from_gps_value(line):
    tmp = line.split(',')
    time = datetime.datetime(2000, 1, 1, 1, 1, 1)
    if len(tmp) >= 10:
        tmp2 = tmp[9]
        tmp = tmp[1]
        try:
            year = int("20" + tmp2[4:])
            month = int(tmp2[2:4])
            day = int(tmp2[0:2])
        except ValueError:
            return time
        try:
            hour_tmp = int(tmp[0:2])
        except ValueError:
            return time
        try:
            minute_tmp = int(tmp[2:4])
        except ValueError:
            return time
        try:
            sec_tmp = int(tmp[4:6])
        except ValueError:
            return time
        try:
            time = datetime.datetime(year, month, day, hour_tmp, minute_tmp, sec_tmp)
            return time
        except TypeError:
            return time
        except ValueError:
            return time
    else:
        return time


def assign_lines(lines):
    # Read data from DAT file
    # Input:
    # lines: list of lines from the DAT file
    # Output:
    # data: list of data

    data = []           # list of data values
    time_pps = []           # list of time values assigned to data list
    factor = 1

    dated_lines = []

    count = 0
    count_sample = 0
    adc_offset = 0.
    have_timing = 0
    gps_count = 0
    threshold = 20000000.
    assigned_pps = 0
    time_shift = 0.0
    start_time = -1
    last_gps_time = 0.
    have_s = 0.
    l_pps = 0.
    sample_rate = 20.
    gps_count = 0.
    f_pps = 0
    hold_data = 0
    val = 0
    lsval = 0
    for line in lines:
        
        if line[0] == '+' or line[0] == '-' or line[0] == ' ':
            
            count_sample += 1
            dated_lines += [(count_sample / 150) / 86400]
        
        else:

            dated_lines += [-1]

    return dated_lines


def read_data(lines):
    # Read data from DAT file
    # Input:
    # lines: list of lines from the DAT file
    # Output:
    # data: list of data

    data = []           # list of data values
    time_pps = []           # list of time values assigned to data list
    factor = 1

    count = 0
    adc_offset = 0.
    have_timing = 0
    gps_count = 0
    threshold = 20000000.
    assigned_pps = 0
    time_shift = 0.0
    start_time = -1
    last_gps_time = 0.
    have_s = 0.
    l_pps = 0.
    sample_rate = 20.
    gps_count = 0.
    f_pps = 0
    hold_data = 0
    val = 0
    lsval = 0
    for line in lines:
        if line[1:4] == "Time":
            tmp = line.split(':')
            time_shift = (int(tmp[1][:-1]) / 1000.0) / (60. * 60. *24.)
        if line[1:11] == 'Conversion':
            tmp = line.split(':')
            factor = float(tmp[1][:-1])
        if line[1:4] == 'ADC':
            tmp = line.split(':')
            try:
                adc_offset = float(tmp[1][:-1])
            except:
                adc_offset = 0.
        if line[1:7] == 'Sample':
            tmp = line.split(':')
            sample_rate = int(tmp[1][:-1])              # Frequency sample rate
            t_inc = (1. / sample_rate) / (60 * 60 * 24)  # Time increment between two samples (in years)
        if line[:-1] == 'PPS':
            assigned_pps = 1
            #print("Found PPS")
        if line[0] == 'S':
            #print("Found second")
            tmp = line.split('S')
            val = float(tmp[1][:-1])
            if val < 360000000:
                hold_data = 0
                have_timing += 1
                if have_timing == 0 or gps_count == 0:
                    f_pps = count
                    l_pps = count
                    lsval = count
                else:
                    obs_samplerate = count - l_pps
                    have_s = 1
                    #print("Sample rates", obs_samplerate, sample_rate)
                    if not obs_samplerate == sample_rate:
                        spacing = val - lsval
                        if spacing > 0:
                            samples = count - l_pps                 # Number of samples
                            exp_samples = spacing * sample_rate     # Number of samples from file
                            missing = int(exp_samples - samples)         # Difference is missing samples
                            if missing > 0:
                                tmp = data[-1]
                                for i in range(missing):
                                    data.append(tmp)
                                    count += 1
                            else:
                                count += missing

                l_pps = count
                lsval = val
            else:
                hold_data = 1
        if line[0] == '$':
            tmp = line.split(',')
            if tmp[0] == "$GNRMC" or tmp[0] == "$GPRMC" or tmp[0] == "$PSRFC" \
                    or tmp[0] == "$PSRMC" or tmp[0] == "$PPRMC" :
                #print('Reading GPS String')
                gps_valid = 0
                if have_timing > 2 and assigned_pps == 1:
                    assigned_pps = 0
                    if len(tmp) >= 10:  # Check if line is complete
                        try:
                            time_tmp = get_date_from_gps_value(line)
                            time_tmp = get_julian_datetime(time_tmp)
                            #time_tmp += ((time_shift / 1000.) / 60. * 1 / 60. * 1 / 24.)
                            gps_count += 1
                        except ValueError:
                            pass
                        gps_valid = 1
                    if gps_valid == 1:
                        if start_time == -1:
                            start_time = time_tmp - (t_inc * f_pps)
                            time_pps.append(start_time)
                        else:
                            time_pos = l_pps + 1
                            if time_pos < len(data) and have_s == 1:
                                if time_tmp == last_gps_time:
                                    time_tmp = time_tmp + 1. / 86400.
                                if len(time_pps) < l_pps:
                                    while len(time_pps) < l_pps:
                                        time_pps.append(0.)
                                    time_pps[l_pps - 1] = time_tmp
                                else:
                                    time_pps[l_pps - 1] = time_tmp
                                    have_s = 0
                                    last_gps_time = time_tmp
        if line[0] == '+' or line[0] == '-' or line[0] == ' ':
            if have_timing > 0:
                if hold_data == 0:
                    if len(line) <= 12:
                        try:
                            if int(line[:-1]) / factor - adc_offset < threshold:
                                count += 1
                                data.append(int(line[:-1]))
                        except:
                            pass
    s_time = start_time
    if len(data) > 0:
        if len(time_pps):
            for i in range(len(time_pps)):
                """
                if not time_pps[i] == 0:
                    s_time = time_pps[i]
                else:
                    s_time = s_time + t_inc
                    time_pps[i] = s_time
                """
                s_time = s_time + t_inc
                time_pps[i] = s_time

    if not time_shift == 0:
        for i in range(len(time_pps)):
            time_pps[i] += time_shift

    for i in range(len(data)):
        data[i] = (data[i] / factor) - adc_offset

    data = data[0:len(time_pps)]
    time = time_pps[0:]

    time, data = trimTimeSeries(time, data, 150., 2.)

    return time, data


def trimTimeSeries(time, data, sample_rate, timebase):
    advance = 2.
    time = np.asarray(time)
    data = np.asarray(data)
    # begin by fixing the start time of the record
    t_inc = (1 / sample_rate) * 2
    # start time of the record
    starttime = time[0] + ((advance) / (3600.0 * 24))
    #  create date vector from advanced start time
    dv = serial_gregarion(starttime)

    # identify the start time in seconds
    start_t = (dv[4] * 60.0) + dv[5]

    # identify time of next sample
    next_t = np.ceil(start_t / (timebase * 4.0)) * 4.0 * timebase
    # update the date vector
    dv[4] = np.floor(next_t / 60.0)
    dv[5] = next_t % 60
    # now retrieve this new time's julian date
    new_start_time = serial_julian_date(dv[1], dv[2], dv[0], dv[3], dv[4], dv[5])
    # now deal with the end time
    # by converting to gregarion
    dv_end = serial_gregarion(time[time.size - 1])
    # // now get end time in total seconds using mins + secs
    end_t = (dv_end[4] * 60.0) + dv_end[5]
    # // get next end time sample
    next_endt = np.floor(end_t / (timebase * 4.0)) * 4.0 * timebase
    # // update the datavector dv_end
    dv_end[4] = np.floor(next_endt / 60)
    dv_end[5] = next_endt % 60
    # // determine the new julian endtime
    new_end_time = serial_julian_date(dv_end[1], dv_end[2], dv_end[0], dv_end[3], dv_end[4], dv_end[5]) - (t_inc / (60.0 * 60.0 * 24.0 * 2.0))
    # // now lets check for shifting
    pluscycle = new_end_time + 4 * (timebase / (60.0 * 60.0 * 24.0))
    shift = pluscycle - time[time.size - 1]

    advance_pos = time >= new_start_time
    data = data[advance_pos]
    time = time[advance_pos]
    trim_end = time <= new_end_time
    data = data[trim_end]
    time = time[trim_end]

    return time, data


def serial_julian_date(month, days, year, hrs, mins, secs):
    nhrs = hrs / 24
    nmin = mins / 60 * 1 / 24
    nsec = secs / 60 * 1 / 60 * 1 / 24

    day = days + +nhrs + nmin + nsec

    if ((month == 1) or (month == 2)):
        yearp = year - 1
        monthp = month + 12
    else:
        yearp = year
        monthp = month

    # // this checks where we are in relation to October 15, 1582, the beginning
    # // of the Gregorian calendar.
    if ((year < 1582) or ((year == 1582) and (month < 10)) or ((year == 1582) and ((month == 10) and (day < 15)))):
        # // before start of Gregorian calendar
        B = 0
    else:
        # // after start of Gregorian calendar
        A = np.trunc(yearp / 100.0)
        B = 2 - A + np.trunc(A / 4.0)

    if (yearp < 0):
        C = np.trunc((365.25 * yearp) - 0.75)
    else:
        C = np.trunc(365.25 * yearp)

    D = np.trunc(30.6001 * (monthp + 1))

    ndate = B + C + D + day + 1720994.5

    # //cout << "B: " << B <<  " C: " << C << " D: " << D << " day: " << day << endl;
    # //cout << "yearp: " << yearp << endl;

    return ndate


def serial_gregarion(jdate):

    jdate = jdate + 0.5

    fracp, intpart = np.modf(jdate)
    I_ = intpart

    A = np.trunc((I_ - 1867216.25) / 36524.25)
    if I_ > 2299160:
        B = I_ + 1 + A - np.trunc(A / 4.0)
    else:
        B = I_

    C = B + 1524

    D = np.trunc((C - 122.1) / 365.25)

    E = np.trunc(365.25 * D)

    G = np.trunc((C - E) / 30.6001)

    day = C - E + fracp - np.trunc(30.6001 * G)

    if (G < 13.5):
        month = G - 1
    else:
        month = G - 13

    if (month > 2.5):
        year = D - 4716
    else:
        year = D - 4715

    #  now determine hrs, mins, secs
    hrs = (day - np.floor(day)) * (24.0)
    mins = (hrs - np.floor(hrs)) * (60.0)
    secs = (mins - np.floor(mins)) * (60.0)
    # printf("wtf: %f\n", (day - floor(day))*24);
    #  create the datavector
    datevec = np.zeros(6)
    datevec[0] = year
    datevec[1] = month
    datevec[2] = np.floor(day)
    datevec[3] = np.floor(hrs)
    datevec[4] = np.floor(mins)
    datevec[5] = secs

    return datevec


def is_common_ts(time_1, time_2, sample_freq):
    test = 1
    if time_1[-1] < time_2[0] or time_1[0] > time_2[-1]:
        #print('No common time series is possible')
        test = 0
    else:
        if time_1[0] > time_2[0]:
            begin = time_1[0]
        else:
            begin = time_2[0]

        if time_1[-1] > time_2[-1]:
            end = time_1[-1]
        else:
            end = time_2[-1]
        time_span = (end - begin).total_seconds()
        time_span *= sample_freq
    return test, time_span

def get_common_ts(time_1, time_2, data_1, data_2):

    test = 1
    if time_1[-1] < time_2[0] or time_1[0] > time_2[-1]:
        #print('No common time series is possible')
        test = 0
    else:
        time_min = np.max([time_1[0], time_2[0]])
        time_max = np.min([time_1[-1], time_2[-1]])

        data_1 = [data_1[i] for i in range(len(time_1)) if time_1[i] > time_min and time_1[i] < time_max]
        time_1 = [time_1[i] for i in range(len(time_1)) if time_1[i] > time_min and time_1[i] < time_max]
        data_2 = [data_2[i] for i in range(len(time_2)) if time_2[i] > time_min and time_2[i] < time_max]
        time_2 = [time_2[i] for i in range(len(time_2)) if time_2[i] > time_min and time_2[i] < time_max]

    return time_1, time_2, data_1, data_2, test


def get_on_time(time, data, time_base, sample_freq):

    N = int(time_base * sample_freq * 4)

    _, synthetic = signalAnalysis.cycle_current(150., 2, N)
    test = convolve(synthetic[:], data[:N])

    index = [i for i in range(len(test)) if test[i] == np.max(test)]
    index = index[0] - 1
    on_time = time[index]

    return on_time, index


def get_on_time_old(time_current, data_current, time_base, sample_freq):

    N = time_base * sample_freq

    data_current = detrend(data_current - np.median(data_current))
    on_time = -1
    index = 0
    if np.median(data_current[:int(N / 2)]) > 5:
        if np.median(data_current[int(N / 2):N]) > 5:
            on_time = time_current[0]
            index = 0
        else:
            on_time = time_current[int(N / 2) + 3 * N]
            index = int(N / 2) + 3 * N
    elif np.median(data_current[:int(N / 2)]) < -5:
        if np.median(data_current[int(N / 2):N]) <= -5:
            on_time = time_current[2 * N]
            index = 2 * N
        else:
            on_time = time_current[int(N / 2) + N]
            index = int(N / 2) + N
    else:
        if np.abs(np.median(data_current[int(N / 2):N])) <= 5:
            if np.median(data_current[N:N + int(N / 2)]) > 5:
                on_time = time_current[N]
                index = N
            elif np.median(data_current[N:N + int(N / 2)]) < -5:
                on_time = time_current[3 * N]
                index = 3 * N
        else:
            if np.median(data_current[int(N / 2):N]) > 5:
                on_time = time_current[int(N / 2)]
                index = int(N / 2)
            elif np.median(data_current[N:N + int(N / 2)]) < -5:
                on_time = time_current[2 * N + int(N / 2)]
                index = 2 * N + int(N / 2)
    plt.plot(time_current, data_current)
    plt.plot(time_current[index], data_current[index], 'ro')
    plt.show()
    return on_time, index


def trim_injection_time(time, data, start_date, end_date):

    start_date = get_julian_datetime(start_date)
    end_date = get_julian_datetime(end_date)
    data = [data[i] for i in range(len(time)) if time[i] >= start_date and time[i] <= end_date]
    time = [time[i] for i in range(len(time)) if time[i] >= start_date and time[i] <= end_date]

    return time, data


def trim_on_time(time, data, on_time, time_base, sample_freq):
    if on_time > 0:
        julian_second = 1.1574074074074073e-05

        loc = 0
        decal = (time[0] - on_time) / (julian_second * 4 * time_base)
        #print(decal)
        #tmp = np.median(data)
        if decal >= 0: # Node recording starts after current recorder,
            decal = int(np.ceil(decal))
        else:
            decal = -1 * int(np.abs(np.ceil(decal)))
        closest_on_time = on_time + decal * julian_second * 4 * time_base

        for loc in range(len(time)):
            if np.abs(time[loc] - closest_on_time) < (julian_second / sample_freq):
                break
        #while np.abs(time[loc] - closest_on_time) > (julian_second / sample_freq) and loc < len(time):
        #    loc += 1
            #print(multiprocessing.current_process().name, loc - 1, len(time), time[loc])


        #plt.plot(time, data)
        #plt.plot(closest_on_time, tmp, 'bo')
        if loc < len(time):
            time = time[loc:]
            data = data[loc:]
        else:
            time = []
            data = []
        #plt.plot(time[loc], data[loc], 'ok')
    return time, data

def write_node_entry(rec, file):
    # Description goes here

    start_date = str(rec.start_date.year) + '-' + str(rec.start_date.month) + '-' + str(rec.start_date.day) \
        + 'T' + str(rec.start_date.hour) + ':' + str(rec.start_date.minute) + ':' + str(rec.start_date.second)
    end_date = str(rec.end_date.year) + '-' + str(rec.end_date.month) + '-' + str(rec.end_date.day) \
        + 'T' + str(rec.end_date.hour) + ':' + str(rec.end_date.minute) + ':' + str(rec.end_date.second)
    file.write('\t<file_name:' + rec.name + '>\n')
    file.write('\t\t<node_type:' + rec.node_type + '>\n')
    file.write('\t\t<node_mem:' + str(rec.mem) + '>\n')
    file.write('\t\t<start_date:' + start_date + '>\n')
    file.write('\t\t<end_date:' + end_date + '>\n')
    file.write('\t\t<relay_state:' + rec.relay_state + '>\n')
    file.write('\t\t<northing:' + str(rec.northing) + '>\n')
    file.write('\t\t<easting:' + str(rec.easting) + '>\n')
    file.write('\t\t<altitude:' + str(rec.altitude) + '>\n')

    return


def make_library_file(node_list, recs, root):
    # Make library file so nothing has to be redone again

    node_library = open(root + 'nodeLibrary.dat', 'w')
    for i in range(len(node_list)):
        if len(recs[i]):
            node_library.write('<' + node_list[i] + '>\n')
            for rec in recs[i]:
                write_node_entry(rec, node_library)
    node_library.close()

    return


def get_date_from_library(date):
    # Description goes here
    date = date.split('T')
    tmp1 = date[0].split('-')
    tmp2 = date[1].split(':')
    date_datetime = datetime.datetime(int(tmp1[0]), int(tmp1[1]), int(tmp1[2]),
                                      int(tmp2[0]), int(tmp2[1]), int(tmp2[2]))

    return date_datetime


def get_node_list_from_library(lines):

    node_list = []
    for line in lines:
        if line[0] == '<':  # new node
            node_list.append(line[1:-2])

    return node_list


def read_library_file(library_file):

    file = open(library_file, 'r')
    lines = file.readlines()
    node_list = get_node_list_from_library(lines)
    recs = [[] for i in range(len(node_list))]
    count = 0
    for line in lines:
        if line[0] == '<':  # new node
            node_id = line[1:-2]
            tmp = [i for i in range(len(node_list)) if node_id == node_list[i]]
            node_index = tmp[0]
        else:  # same node, new file, \t,
            if count == 0:
                tmp = line.split('file_name:')
                file_name = tmp[1][:-2]
            elif count == 1:
                tmp = line.split('node_type:')
                node_type = tmp[1][:-2]
            elif count == 2:
                tmp = line.split('node_mem:')
                node_mem = tmp[1][:-2]
            elif count == 3:
                tmp = line.split('start_date:')
                start_date = get_date_from_library(tmp[1][:-2])
            elif count == 4:
                tmp = line.split('end_date:')
                end_date = get_date_from_library(tmp[1][:-2])
            elif count == 5:
                tmp = line.split('relay_state:')
                relay_state = tmp[1][:-2]
            elif count == 6:
                tmp = line.split('northing:')
                northing = tmp[1][:-2]
            elif count == 7:
                tmp = line.split('easting:')
                easting = tmp[1][:-2]
            elif count == 8:
                tmp = line.split('altitude:')
                altitude = tmp[1][:-2]
            count += 1
        if count == 9:
            try:
                recs[node_index].append(Record(node_id=node_id,
                                               node_type=node_type,
                                               name=file_name,
                                               mem=int(node_mem),
                                               start_date=start_date,
                                               end_date=end_date,
                                               relay_state=relay_state,
                                               northing=float(northing),
                                               easting=float(easting),
                                               altitude=float(altitude)))
            except IndexError:
                pass
            count = 0
    file.close()

    return recs


def get_node_list_from_records(records):

    node_list = []
    for i in range(len(records)):
        if len(records[i]):
            node_list.append(records[i][0].node_id)

    return node_list


def add_new_nodes(records, node_list):

    node_list_library = get_node_list_from_records(records)

    for i in range(len(node_list)):
        j = 0
        while node_list[i] > node_list_library[j]:
            j += 1
        if not node_list[i] == node_list_library[j]:
            node_list_library.insert(j, node_list[i])
            records.insert(j, [])

    return records, node_list_library


def is_file_in_library(records, dat_file, node_index):

    test = 1
    for i in range(len(records[node_index])):
        if dat_file == records[node_index][i].name:
            test = 0
            break

    return test

def get_closest(id, nodes, distances, values):

    arg_distances = np.argsort(distances)
    count = 0
    while nodes[arg_distances[count]].id == id and nodes[arg_distances[count + 1]].id == id:
        count += 1

    return [distances[arg_distances[count]], distances[arg_distances[count + 1]]], [values[arg_distances[count]], values[arg_distances[count + 1]]]


def read_node_messages(path):
    """
    Read node messages file and return a list of files created during
    acquisition
    """
    fIn = open(path, 'r', encoding="utf8", errors='ignore')
    lines = fIn.readlines()
    fIn.close()
    files = []
    for line in lines:
        if '.dat' in line:
            tmp = line.split(',')
            for i in range(len(tmp)):
                if '.dat' in tmp[i]:
                    files.append(tmp[i])

    files = sorted(files)
    # Organizing with node id
    node_id = []
    for i in range(len(files)):
        id = files[i][0:2]
        if id not in node_id:
            node_id.append(id)
    node_id = sorted(node_id)

    struct = [[] for i in range(len(node_id))]
    id_prev = files[0][0:2]
    index = 0
    for i in range(len(files)):
        id = files[i][0:2]
        if not id == id_prev:
            index += 1
        if files[i] not in struct[index]:
            struct[index].append(files[i])
        id_prev = id[:]
    return struct, node_id
