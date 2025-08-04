"""
ObsPy tutorial script.

This script demonstrates how to:
- Read seismogram data using obspy.read
- Plot seismograms using matplotlib
- Apply low-pass filtering to the data
- Downsample (decimate) traces
- Merge multiple waveform segments
- Download data from an FDSN service (e.g., IRIS)

Each function below performs a specific operation. You can run this script
as a standalone program; it will execute all examples sequentially.  If you
are using this in an interactive environment (e.g., Jupyter), you can
import individual functions and run them as needed.
"""

import obspy
import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.clients.fdsn import Client


def read_and_plot_example():
    """Read a seismogram and display basic information and a plot."""
    # Read an example file from the ObsPy repository
    st = obspy.read('http://examples.obspy.org/RJOB_061005_072159.ehz.new')
    tr = st[0]

    # Print summaries
    print(st)
    print(tr)
    print(tr.stats)

    # Plot the seismogram
    st.plot()


def filter_example():
    """Apply a low-pass filter to a trace and compare with the raw data."""
    st = obspy.read('https://examples.obspy.org/RJOB_061005_072159.ehz.new')
    tr = st[0]

    # Copy the trace to preserve original data
    tr_filt = tr.copy()

    # Apply a low-pass filter (1 Hz cutoff, two corners, zero-phase)
    tr_filt.filter('lowpass', freq=1.0, corners=2, zerophase=True)

    # Generate time axis
    t = np.arange(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.delta)

    # Plot raw and filtered data
    plt.subplot(2, 1, 1)
    plt.plot(t, tr.data, 'k')
    plt.ylabel('Raw Data')

    plt.subplot(2, 1, 2)
    plt.plot(t, tr_filt.data, 'k')
    plt.ylabel('Low‑passed Data')
    plt.xlabel('Time [s]')
    plt.suptitle(str(tr.stats.starttime))
    plt.show()


def decimate_example():
    """Downsample a trace and compare with a low-pass filtered trace."""
    st = obspy.read('https://examples.obspy.org/RJOB_061005_072159.ehz.new')
    tr = st[0]

    # Downsample (factor=4) with automatic pre-filtering
    tr_dec = tr.copy()
    tr_dec.decimate(factor=4, strict_length=False)

    # Apply an equivalent low-pass filter without decimation
    tr_filt = tr.copy()
    tr_filt.filter('lowpass', freq=0.4 * tr.stats.sampling_rate / 4.0)

    # Generate time axes
    t = np.arange(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.delta)
    t_dec = np.arange(0, tr_dec.stats.npts / tr_dec.stats.sampling_rate, tr_dec.stats.delta)

    # Plot comparisons
    plt.plot(t, tr.data, 'k', label='Raw', alpha=0.3)
    plt.plot(t, tr_filt.data, 'b', label='Low‑pass', alpha=0.7)
    plt.plot(t_dec, tr_dec.data, 'r', label='Low‑pass/Decimated', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.xlim(82, 83.5)
    plt.title(str(tr.stats.starttime))
    plt.legend()
    plt.show()


def merge_example():
    """Merge multiple waveform segments into a single trace and plot."""
    # Read three contiguous MiniSEED segments
    st = obspy.read('https://examples.obspy.org/dis.G.SCZ.__.BHE')
    st += obspy.read('https://examples.obspy.org/dis.G.SCZ.__.BHE.1')
    st += obspy.read('https://examples.obspy.org/dis.G.SCZ.__.BHE.2')

    # Sort by start time
    st.sort(['starttime'])

    # Plot each segment relative to the start time of the first trace
    t0 = st[0].stats.starttime
    fig, axes = plt.subplots(nrows=len(st) + 1, sharex=True)
    for tr, ax in zip(st, axes[:-1]):
        ax.plot(tr.times(reftime=t0), tr.data)

    # Merge segments
    st.merge(method=1)

    # Plot merged data
    axes[-1].plot(st[0].times(reftime=t0), st[0].data, 'r')
    axes[-1].set_xlabel(f'seconds relative to {t0}')
    plt.show()


def download_example():
    """Download waveform data from an FDSN service and remove the instrument response."""
    client = Client('IRIS')
    starttime = UTCDateTime('2020-01-01T00:00:00')
    endtime = UTCDateTime('2020-01-01T00:02:00')
    network = 'IU'
    station = 'ANMO'
    location = '00'
    channel = 'BH*'

    # Download waveform and station metadata
    st = client.get_waveforms(network, station, location, channel, starttime, endtime)
    inventory = client.get_stations(
        network=network,
        station=station,
        starttime=starttime,
        endtime=endtime,
        level='response'
    )

    # Remove instrument response and plot
    st.remove_response(inventory=inventory)
    st.plot()


if __name__ == '__main__':
    # Call each example in sequence. Feel free to comment out
    # functions you do not wish to run.
    read_and_plot_example()
    filter_example()
    decimate_example()
    merge_example()
    download_example()
