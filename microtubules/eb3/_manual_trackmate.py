"""
Import TrackMate XML data to create pandas DataFrames for track
coordinates.

Below is the justification for using pandas DataFrames instead of
numpy Arrays.

The TrackMate XML file uses SQL table-style data structures in three
groups:
1. All "spots" of a track which contain the coordinate information.
2. Each "track" containing multiple edges, where an edge maps to a
   start and end spot ID.
3. The "edge" is relates spots to tracks, and also contains duration,
   distance, etc.
See the <FeatureDeclarations> at the beginning of the XML file for all
the types of data provided.

We need to use the `merge' function in pandas to map the spots
coordinates to to tracks.  There does not seem to be a simple
equivalent in numpy to do an SQL table-style merge.  Also pandas data
structures are more easily read by data scientists since they don't
have to understand axes indices to see what is referenced; even numpy
structured arrays are not as transparent as pandas multi-indexing.

Adapted from Pariksheet Nanda
https://github.uconn.edu/pan14001
https://github.uconn.edu/pan14001/cell-bio-lab-2015/blob/master/fig/run_manual_trackmate.py

"""
import logging
import xml.etree.ElementTree as et
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def read(filename):
    """Read the XML file and create track DataFrames.

    Returns tuple of track coordinates, edges and metrics DataFrames.
    If relative=False, return absolute XY coordiantes.

    """

    # Read in the raw XML file.
    log.info('Reading XML file, {0}'.format(filename))
    tree = et.parse(filename)
    root = tree.getroot()
    # Create references to the tags of interest.  Edges are implicitly
    # part of tracks.
    spot_elements = list(root.iter('Spot'))
    track_elements = list(root.iter('Track'))

    # Create 5 pandas DataFrames based on max sizes, namely:
    # 1. `all_spot_coordinates' to read the `x_um', `y_um' and `time_s'.
    #    Indexed by `spot_ids'.
    # 2. `track_spot_ids' to read `spot_id' and `track_id'.  No explicit
    #    index.
    # 3. `track_edges' to read `edge_time_s', `edge_displacement_um',
    #    `velocity_umps', and to calculate `slope'.  Index by `track_id'
    #    and sub-index by `edge_order'.
    # 4. `track_metrics' to read `track_duration_s',
    #    `track_displacement_um' and calculate `persistence'.  Indexed
    #    by `track_ids'.
    # 5. `track_coordinates' by merging `all_spot_coordinates' and
    #    `track_spot_ids'.  Indexed by 'track_ids' and sub-indexed by
    #    `spot_order'.
    log.info('Parsing spot information')
    # Spot parsing.
    spot_ids = [int(spot.attrib['ID']) for spot in spot_elements]
    x_um = [float(spot.attrib['POSITION_X']) for spot in spot_elements]
    y_um = [float(spot.attrib['POSITION_Y']) for spot in spot_elements]
    time_s = [float(spot.attrib['POSITION_T']) for spot in spot_elements]
    frame = [int(spot.attrib['FRAME']) for spot in spot_elements]
    # Track parsing.
    track_ids = [int(track.attrib['TRACK_ID']) for track in track_elements]
    track_names = [track.attrib['name'] for track in track_elements]
    track_duration_s = [float(track.attrib['TRACK_DURATION']) for track in track_elements]
    track_displacement_um = [float(track.attrib['TRACK_DISPLACEMENT']) for track in track_elements]

    # Edge parsing requires iterating through XML tracks and storing them
    # into DataFrames, so that will be done later after allocating the
    # DataFrames.

    # Allocate DataFrames.
    #
    # Fully populated DataFrame.
    all_spot_coordinates = pd.DataFrame(
        {'x_um': x_um, 'y_um': y_um, 'frame': frame, 'time_s': time_s, 'spot_id': spot_ids}
    )

    # Empty DataFrames.
    track_spot_ids = pd.DataFrame(
        columns={'spot_id', 'track_id', 'track_name'},
    )

    log.info('Parsing edge information from tracks')
    ntracks = len(track_elements)
    track_nspots = [int(track.attrib['NUMBER_SPOTS']) for track in track_elements]
    for t in range(ntracks):
        track = track_elements[t]
        nedges = track_nspots[t] - 1
        log.debug('\r[{0:{digits}d}/{1:{digits}d}]'.format(t + 1, ntracks, digits=len(str(ntracks))))
        one_track_spot_ids = np.full(nedges * 2, np.nan, dtype='u8')
        track_edge_set = pd.DataFrame(
            columns={'edge_time_s', 'edge_displacement_um', 'velocity_umps'},
        )
        for i in range(nedges):
            edge = track.getchildren()[i]
            # Collect information to fill `track_edges'.
            one_track_edge = pd.DataFrame({
                'edge_time_s':
                    [float(edge.attrib['EDGE_TIME'])],
                'edge_displacement_um':
                    [float(edge.attrib['DISPLACEMENT'])],
                'velocity_umps':
                    [float(edge.attrib['VELOCITY'])],
            })
            track_edge_set = track_edge_set.append(one_track_edge)
            # Get spot IDs.
            source_id = int(edge.attrib['SPOT_SOURCE_ID'])
            target_id = int(edge.attrib['SPOT_TARGET_ID'])
            one_track_spot_ids[2 * i:2 * (i + 1)] = np.array([source_id, target_id])
        one_track_spot_ids = np.unique(one_track_spot_ids)
        assert len(one_track_spot_ids) == nedges + 1
        # Append to `track_spot_ids'.
        one_track_spot_ids = pd.DataFrame({
            'spot_id': one_track_spot_ids,
            'track_id': track_ids[t],
            'track_name': track_names[t],
        })
        track_spot_ids = track_spot_ids.append(one_track_spot_ids)
        # Append to `track_edges' with appropriate multi-index.
        track_edge_set = track_edge_set.sort_values(
            by=['edge_time_s'],
        )
        track_edge_set.index = [
            np.ones(nedges, dtype='u8') * track_ids[t],  # track id
            np.arange(nedges, dtype='u8'),  # edge order
        ]
        track_edge_set.index.names = ['track_id', 'edge_order']
        if t == 0:
            track_edges = track_edge_set.copy()
        else:
            track_edges = track_edges.append(track_edge_set)

    # For each track, get the coordinates and sort by time.  Tracks list
    # their edges in an arbitrary order, and so sorting by time orders
    # them correctly.
    track_spot_ids["spot_id"] = track_spot_ids["spot_id"].astype(int)
    all_spot_coordinates["spot_id"] = all_spot_coordinates["spot_id"].astype(int)
    track_coordinates = pd.merge(
        all_spot_coordinates, track_spot_ids,
        left_on='spot_id', right_on='spot_id',
        sort=False,
    )
    track_coordinates = (track_coordinates
                         .sort_values(by=['track_id', 'time_s'])
                         .reset_index(drop=True)
                         )

    # return coords, track_edges, track_metrics
    return track_coordinates


def directionality(edges, min_time_s=300):
    """Return edge directionality."""
    edge_set = edges[edges.edge_time_s >= min_time_s]
    directionality = np.cos(np.deg2rad(edge_set.angle))
    return directionality
