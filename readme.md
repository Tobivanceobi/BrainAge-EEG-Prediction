# Brain age prediction on EEG-Data
___
## Data Set

**Subject:** timeseries of EEG recording (Eyes opened/closed)

**Predict:** age of subject
___
### Training Data

- 2400 raw EEG files (MNE)
- 1200 subjects
- 129 electrodes per subject
- 500 Hz for 20s (eyes open) or 40s (eyes closed)
- Target: 1200 age values
___
#### Data Dimensions:
```
1200 x 129 x 10000 (open)
1200 x 129 x 20000 (close)
```
___
### Test Data

400 subjects (800 EEG files)
