import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import utm
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import logging
logging.basicConfig(filename='qc.log' , format='%(asctime)s | %(levelname)s: %(message)s', level=logging.NOTSET)
fontprops = fm.FontProperties(size=18)


def run_line_spacing_qc_mag(

    line_seperation:float = 200,
    data_by_line: xr.core.groupby.DatasetGroupBy = None,
    number_of_data_in_file:int = 0,
    output_file:str = 'mag_default_line_seperation_qc.csv',
    output_png:bool = False,

)->None:
    """
        Method to run quality control item line seperation on an mag data file

        :param line_seperation: the maximum line seperation allowed
        :type line_seperation: float
        :param data: mag data file contents
        :type data: xr.core.groupby.DatasetGroupBy
        :param number_of_data_in_file: item total in file
        :type number_of_data_in_file: int
        :param output_file: output qc file
        :type output_file: str
        :param output_png: output qc file
        :type output_png: bool

    """


def run_line_spacing_qc_agg(
    
    line_seperation:float = 200,
    data_by_line: xr.core.groupby.DatasetGroupBy = None,
    time_constant: float = 0.18
    number_of_data_in_file:int = 0,
    output_file:str = 'default_qc',
    output_png:bool = False,
    instrument_dropout: bool = True
    
    )->None:
    """
        Method to run quality control item line seperation on an AGG data file

        :param line_seperation: the maximum line seperation allowed
        :type line_seperation: float
        :param data: AGG data file contents
        :type data: xr.core.groupby.DatasetGroupBy
        :param line_seperation: the maximum line seperation allowed
        :type time_constant: float
        :param time_constant: the time constant of the demodulation filter
        :type number_of_data_in_file: int
        :param output_file: output qc file
        :type output_file: str
        :param output_png: output qc file
        :type output_png: bool
        :param instrument_dropout: run qc on instrument dropout
        :type instrument_dropout: bool
    
    """

    # initiate problem historian
    problem_hist = {'line_error': []}

    # initiate matrix of qc dta
    check_data = np.zeros((number_of_data_in_file, 3))

    # create figure for plotting
    fig, ax = plt.subplots(1,1, figsize=(10, 10))

    # open file to write csv info
    csv_qc = open(output_file, 'w+')
    csv_qc.write('line_id,easting,northing\n')

    csv_dropout_qc = open(output_file, 'w+')
    csv_dropout_qc.write('line_id,easting1,northing1,easting2,northing2\n')

    # extract line ids
    line_ids = []

    for group, group_ds in data_by_line:
        variable_keys = group_ds.keys()
        line_ids.append(group)

    # loop through the data
    for ii in range(0, len(data_by_line)):
        try:

            try:
                x, y, zone, letter = utm.from_latlon(
                    data_by_line[line_ids[ii]].Latitude.to_numpy(),
                    data_by_line[line_ids[ii]].Longitude.to_numpy()
                )

                # also grab the time
                difference_between_samples = np.diff(data_by_line[line_ids[ii]].UTC_Time1980)
            
            except utm.error.OutOfRangeError as e:

                logging.info(f"found gap in data causing error: {e}")
                
                # find where we have the nan's and remove them
                remove_logging_gaps = np.isnan(data_by_line[line_ids[ii]].Latitude).to_numpy()

                latitude_gaps_removed = data_by_line[line_ids[ii]].Latitude.to_numpy()[~remove_logging_gaps]
                longitude_gaps_removed = data_by_line[line_ids[ii]].Longitude.to_numpy()[~remove_logging_gaps]
                time_gaps_removed = data_by_line[line_ids[ii]].UTC_Time1980.to_numpy()[~remove_logging_gaps]

                x, y, zone, letter = utm.from_latlon(latitude_gaps_removed, longitude_gaps_removed)

                difference_between_samples = np.diff(time_gaps_removed)

            samples = np.vstack([x, y]).T

            if output_png:
                ax.plot(x, y, '--k', alpha=0.2)

            if ii > 0:

                # also check if line is tie-line or regular
                if 'L' in line_ids[ii-1] and 'L' in line_ids[ii]:
                
                    try:
                        x2, y2, zone2, letter2 = utm.from_latlon(
                            data_by_line[line_ids[ii-1]].Latitude.to_numpy(), 
                            data_by_line[line_ids[ii-1]].Longitude.to_numpy()
                        )
                    
                    except utm.error.OutOfRangeError as e:

                        logging.info(f"found gap in data causing error: {e}")
                        
                        # find where we have the nan's and remove them
                        remove_logging_gaps = np.isnan(data_by_line[line_ids[ii-1]].Latitude).to_numpy()

                        latitude_gaps_removed = data_by_line[line_ids[ii-1]].Latitude.to_numpy()[~remove_logging_gaps]
                        longitude_gaps_removed = data_by_line[line_ids[ii-1]].Longitude.to_numpy()[~remove_logging_gaps]

                        x2, y2, zone2, letter2 = utm.from_latlon(latitude_gaps_removed, longitude_gaps_removed)
                    
                    samples2 = np.vstack([x2, y2]).T

                    # in case of lines shorter than the other need use proper order to get distance
                    xmax = np.max(x)
                    xmin = np.min(x)
                    ymax = np.max(y)
                    ymin = np.min(y)

                    dist = np.sqrt(np.sum(np.array([(xmax - xmin), (ymax - ymin)])**2))

                    xmax2 = np.max(x2)
                    xmin2 = np.min(x2)
                    ymax2 = np.max(y2)
                    ymin2 = np.min(y2)

                    dist2 = np.sqrt(np.sum(np.array([(xmax2 - xmin2), (ymax2 - ymin2)])**2))

                    if output_png:
                        ax.plot(x2, y2, '--k', alpha=0.2)

                    if dist > dist2:
                        # fit longer line and query the smallest
                        neigh = NearestNeighbors(n_neighbors=1)
                        neigh.fit(samples)
                        idx_pick = neigh.kneighbors(samples2)

                    else:
                        # use opposite line assignment to determine line deviation
                        # fit longer line and query the smallest
                        neigh = NearestNeighbors(n_neighbors=1)
                        neigh.fit(samples2)
                        idx_pick = neigh.kneighbors(samples)

                    # check to see if any points are over 1.5x distance
                    if (idx_pick[0] > (line_seperation * 1.5)).sum() > 0:
                        
                        try:
                            # find where they are
                            find_excess_distance = np.where(idx_pick[0] > (line_seperation* 1.5))

                            # grab the offending points
                            # if dist > dist2:
                            excess_points = samples[idx_pick[1][find_excess_distance]]
                            # else:
                            #     excess_points = samples2[idx_pick[1][find_excess_distance]]

                            # add the points to the plot
                            if output_png:
                                ax.plot(excess_points[:, 0], excess_points[:, 1], '.r')
                                ax.annotate(line_ids[ii], xy=(excess_points[0, 0], excess_points[0, 1]))

                            for jj in range(excess_points.shape[0]):
                                csv_qc.write(f'{line_ids[ii]},{excess_points[jj, 0]},{excess_points[jj, 1]}\n')

                        except IndexError as e:

                            print(e)

            # now do instrument dropout
            if instrument_dropout:

                

                f.write(f'{line_ids[ii]},{time_excess_points_start[kk, 0]},{time_excess_points_start[kk, 1]},{time_excess_points_end[kk, 0]},{time_excess_points_start[kk, 1]}\n')
        except Exception as e:
            problem_hist['line_error'] += [line_ids[ii]]
            logging.info(f'line: {line_ids[ii]} with error: {e}')
    
    csv_qc.close()

    if output_png:
        ax.axis('equal')
        fig.savefig('test.png')

if __name__ == '__main__':

    if False:

        # ---------------------------------------------------------------------------------------------

        # AGG data

        #

        # data file to view
        agg_data = r"C:\Users\johnk\Documents\projects\kyle\j0002\test_data\2200168_agg_preliminary.csv"
        output_filepath = r'agg_preliminary_line_qc.csv'

        # line seperation parameter
        line_spacing = 200

        # load data into panadas then to xarray dataframe
        data_obj = pd.read_csv(agg_data).to_xarray()

        # group data by lines
        data_by_line = data_obj.groupby("Line")

        # run the data qc
        run_line_spacing_qc_agg(
            
            line_seperation=line_spacing,
            data_by_line=data_by_line,
            number_of_data_in_file=data_obj.dims['index'],
            output_file=output_filepath,
            output_png=True,
        
        )

    if True:
        # ---------------------------------------------------------------------------------------------

        # MAG data

        #
        mag_data = r"C:\Users\johnk\Documents\projects\kyle\j0002\test_data\2200168_mag_ags_preliminary.csv"
        output_filepath = r'mag_preliminary_line_qc.csv'

        # line seperation parameter
        line_spacing = 200

        # load data into panadas then to xarray dataframe
        data_obj = pd.read_csv(mag_data).to_xarray()

        # group data by lines
        data_by_line = data_obj.groupby("Line")

        # run the data qc
        run_line_spacing_qc_agg(
            
            line_seperation=line_spacing,
            data_by_line=data_by_line,
            number_of_data_in_file=data_obj.dims['index'],
            output_file=output_filepath,
            output_png=True,
        
        )
