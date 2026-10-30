Notes

1. When processing EEG and eye-tracking data, it is recommended to discard data from the initial period (this period marks the beginning of data recording but not the start of the experiment, for instance, the first 24 seconds) to ensure the quality of EEG data.

2. Although the recording of EEG and eye movement data starts simultaneously, they might not end at the same moment due to operational constraints during the experiment. Therefore, when aligning EEG with eye movement data, it is advisable to truncate the data to the shortest length between the two to ensure synchronization.

3. You can use the mne package (python) to read the '.edf' files.  All the channel information can be found in the source files.