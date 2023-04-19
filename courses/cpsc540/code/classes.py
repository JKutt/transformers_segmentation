# Classes used
import glob
import sys
import utm


class Project:

    def __init__(self,
                 path=None,
                 data_path=None,
                 recording_file=None,
                 list_nodes=None,
                 injections=None,
                 records=None,
                 node_data=None,
                 log_file=None):

        self.path = path
        self.data_path = data_path
        self.recording_file = recording_file
        self.list_nodes = list_nodes
        self.injections = injections
        self.records = records
        self.node_data = node_data
        self.log_file = log_file

        self.check_path()
        self.initialize_friday_log()

    def check_path(self):
        tmp = glob.glob(self.path)
        if len(tmp) == 0:
            print("Project path doesn't exist.")
            sys.exit()

        tmp = glob.glob(self.recording_file)
        if len(tmp) == 0:
            print("Recording file doesn't exist.")
            sys.exit()

    def get_node_list(self):
        list_nodes = []
        for i in range(len(self.records)):
            if len(self.records[i]):
                list_nodes.append(self.records[i][0].node_id)
        self.list_nodes = list_nodes

    def initialize_node_data(self):
        self.node_data = [[] for i in range(len(self.injections))]

    def initialize_friday_log(self):
        file = open(self.path + 'friday.log', 'w')
        file.write('Friday says:\n')
        file.close()

    def open_friday_log(self):
        file = open(self.path + 'friday.log', 'a')
        return file

    def close_friday_log(self, file):
        file.close()

    def write_messages_to_friday(self, file, messages):
        for line in messages:
            file.write(line)
        file.flush()


class Injection:
    """ List attributes of an injection.

    attributes:
    valid: Noise, Good or Bad
    num: Mem number
    start_date/end_date
    list_nodes: All nodes active during that injection
    list_files: All files related to that injection
    list_gps: GPS coordinates for all files related to that injection
    list_type: Node type for all files related to that injection

    """

    def __init__(self,
                 valid=None,
                 num=None,
                 start_date=None,
                 end_date=None,
                 list_nodes=None,
                 list_current=None,
                 list_files=None,
                 list_short=None,
                 list_gps=None,
                 list_type=None,
                 list_current_gps=None,
                 list_short_gps=None):
        
        self.valid = valid
        self.num = num
        self.start_date = start_date
        self.end_date = end_date
        self.list_current = list_current
        self.list_nodes = list_nodes
        self.list_files = list_files
        self.list_short = list_short
        self.list_gps = list_gps
        self.list_current_gps = list_current_gps
        self.list_short_gps = list_short_gps
        self.list_type = list_type
        self.transmit_selection = None


class Record:
    def __init__(self,
                 name=None,
                 node_id=None,
                 node_type=None,
                 mem=None,
                 start_date=None,
                 end_date=None,
                 relay_state=None,
                 northing=None,
                 easting=None,
                 altitude=None):
        self.name = name
        self.node_id = node_id
        self.node_type = node_type
        self.mem = mem
        self.start_date = start_date
        self.end_date = end_date
        self.relay_state = relay_state
        self.northing = northing
        self.easting = easting
        self.altitude = altitude

    def getUtm(self):
        tmp = utm.from_latlon(self.northing, self.easting)
        return tmp[0], tmp[1]


class Node:
    def __init__(self, id=None,
                       file_name=None,
                       mem=None,
                       location=None,
                       harmonics=None,
                       freq_harmonics=None,
                       Vp=None,
                       adf=None,
                       score=None,
                       coherence=None,
                       is_quarantine=0):
        self.id = id
        self.mem = mem
        self.file_name = file_name
        self.location = location
        self.harmonics = harmonics
        self.freq_harmonics = freq_harmonics
        self.Vp = Vp
        self.adf = adf
        self.score = score
        self.coherence = coherence
        self.is_quarantine = is_quarantine

class DataCouple:
    def __init__(self,
                 distance=None,
                 couple=None,
                 coherency=None,
                 cluster=None,
                 mem=None,
                 node_index=None):
        self.distance = distance
        self.couple = couple
        self.coherency = coherency
        self.cluster = cluster
        self.mem = mem
        self.node_index = node_index


class Data:
    def __init__(self,
                 distance=None,
                 harmonics=None,
                 Vp=None,
                 file_name=None,
                 cluster=None,
                 mem=None):
        self.distance = distance
        self.harmonics = harmonics
        self.Vp = Vp
        self.file_name = file_name
        self.cluster = cluster
        self.mem = mem
