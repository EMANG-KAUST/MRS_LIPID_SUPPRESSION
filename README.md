# A novel method for Magnetic Resonance Spectroscopy lipid signal suppression using Semi-classical signal analysis and Bidirectional Long short-term memory

This is a novel method for lipid suppression for MRS signal. The method combines a semi-classical signal analysis method and a bidirectional long short term memory technique.

# Pre-requisite

Install the next packages on your local machine:
pip install keras-tuner -q
pip install BaselineRemoval

## Bayesian_optimization

This program makes the Bayesian search for the Hyperparameters Optimization of the BiLSTM.
### Inputs:
- SCSA_path: The path of the file used to train the model with SCSA already applied
- MRS_C_path: The path of the file with the Ground Truth to train the model 
- ppm_path: The path of the ppm file
### Output:
- model: is the best architecture found for the model already trained with 5000 
epochs and batch_size of 300 signals.

## Test_method

This code trains and validates the optimized model obtained from the Bayesian optimization.
### Input:
- SCSA_path: The path of the file with the MRS with SCSA already applied
- MRS_C_path is the path of the file with the ground truth of the MRS Spectras 
- ppm_path is the path of the ppm file
### Output:
- model: The model with the optimun hyperparameters trained with data base of 2000 MRS Spectra 
for 5000 epochs and batch_size of 300.

## Test_method

This program tests the method proposed for the lipid peak suppression, calculates
the usefulness of the SCSA in the method by calculating the amplitude Ratio (AR) between 
the Ground Truth and the output of the BiLSTM, in MRS spectra with different levels 
of noise= 5,10,15,20,50 dB. Also calculates the signal Intensity Ratio (SIR) of 2 stes of 
MRS spectra with different levels of noise (5, 10 and 15 dB) and different alpha values
(2, 3 and 5).

### Inputs:
- ppm_path: The path of the ppm file
- model: The optimized model obtained from the Bayesian Search already trained
- MRSC_path_test: The path of the file with the ground truth for the SCSA Test
- MRS_path_test: The path of the file of MRS spectra without SCSA for the SCSA Test
- SCSA_path_test: The path of the file of MRS spectra with SCSA for the SCSA Test
- MRSC_path_si_test: The path of the file with the ground truth for the SI test
- SCSA_path_si_test: The path of the file with of MRS spectra with SCSA for the SI test
- MRSC_path_noise_test: The path of the file with the ground truth for the noise test
- SCSA_path_noise_test: The path of the file with of MRS spectra with SCSA for the noise test

### Output:
    
- AR and SIR with SCSA and SIR AR without SCSA 
- Meand and Std of the AR and SIR with and without SCSA with different levels of noise= 5,10,15,20,50 dB.
- SIR Lac1 is the Signal Intensity Ratio of the proximal peak (Lac1) 
- SIR Lac2 is the Signal Intensity Ratio of the distal peak (Lac2)
- Mean and Std of the SIR of lac1 and lac2, for the test with different levels of noise (5, 10 and 15 dB) and different alpha values (2, 3 and 5).

# Acknowledgment

Research reported in this publication was supported by King Abdullah University of Science and Technology (KAUST) with the Base Research Fund (BAS/1/1627-01-01).
