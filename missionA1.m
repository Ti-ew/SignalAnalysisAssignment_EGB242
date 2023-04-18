%% EGB242 Assignment 1 %%
% This file is a template for your MATLAB solution.
%
% Before starting to write code, record your test audio with the record
% function as described in the assignment task.

%% Load recorded test audio into workspace
clear all; 
close all;
load DataA1;
% Begin writing your MATLAB solution below this line.

%1.3 
%{
Create a time vector t for the audio signal. Plot the audio signal against t. Comment on
your observations, and how they relate to any audible characteristics of the signal.
%}
%------------------------------------------------------------------------------------
N = length(audio);
AudioInSeconds=(length(audio)/fs);
t = linspace(0, AudioInSeconds, N);
%1.5 
%{
Using −5 ≤ n ≤ 5 (i.e., 5 harmonics), generate a vector cn in MATLAB which contains
% cn evaluated at each value of n. List the values of these coefficients in your report.
%}
%---------------------------------------------------------------------------------------
T = 2; % Period
f0 = 1/T; %Fundamental Freq
syms x; %Create a symbolic variable 'x' for our function to use, 't' is gonna get used alot
s_t = piecewise(1<=x & x<2, -3*x+8, 0<=x & x<1, 5*exp(4*(x-1))); %Create the piecewise function
% Calculate C0
C0 = (int(s_t,x,0,2)/T);
C0 = simplify(C0);
% Check C0
c0Check = simplify(int(s_t*exp(-1j*2*pi*0*f0*x),x,0,2)/T); %It matches hand solution mashallah
% Check Cn
% Calculate Cn for -5 <= n <= 5
harmonic = -5:5;
cn = zeros(size(harmonic));
for i = 1:length(harmonic)
    n = harmonic(i);
    CnthValue = (int(s_t*exp(-1j*2*pi*n*f0*x),x,0,2)/T);
    cn(i) = CnthValue;
end 


%1.6 
%{
Using the cn vector, generate an approximation of the noise signal nApprox for the full
time vector t. Plot your recorded audio and your generated noise signal approximation.
%}
nApprox = ApproximationValues(harmonic, cn, t, f0);
% Plot recorded audio and noise signal approximation
figure;
subplot(4,6,1);
plot(t, audio);
ylim([0 6]);
title('Recorded audio');
xlabel('Time (s)');
ylabel('Amplitude');
subplot(4,6,2);
plot(t, real(nApprox));
title('Noise approximation');
xlabel('Time (s)');
ylabel('Amplitude');

%1.7
%{
De-noise the recorded audio by reversing the additive noise process (Figure 2) using your
Fourier series approximation, and store the de-noised signal in audioClean. Listen to
the clean signal, and plot it.
%}

audioClean = audio - nApprox; %Remove noise
audioClean = real(audioClean); %Unable to use sound function without real
if max(abs(audioClean)) > 1
    disp('Clipping detected');
else
    disp('No clipping detected');
end
% Plot clean signal
subplot(4,6,3);
plot(t, audioClean);
xlabel('Time (s)');
ylabel('Amplitude');
title('Clean Audio Signal: (s)');


%1.8
%{
Is using 5 harmonics in your noise signal approximation enough to adequately de-noise
the audio? Experiment with the number of harmonics to determine a suitable value, and
justify your choice both qualitatively and quantitatively.
%}


%2.1
%{
Plot the magnitude spectrum of the clean audio signal, using an appropriate frequency
vector f
%}
%Frequency domain for audio clean
N = length(audioClean);
frequencyVector = (0:N-1)*(fs/N);
fftAudioClean = fft(audioClean);
fftAudioClean = abs(fftAudioClean);
subplot(4,6,4);
plot(frequencyVector,(fftAudioClean));
ylim([0, max(fftAudioClean)]);
xlim([0, fs/2]);
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Clean Audio: (Hz)')

%2.2
%{
“Listen” to the channel before transmitting your signal through it. You can “listen” to the
channel before transmitting anything by passing a vector of zeroes through the channel
function.
%}


%2.3
%{
Plot the time and frequency domain of channelQuiet in order to find an empty band of
frequencies that you can transmit your audio on. State your selected range of frequencies
and the center frequency. Justify these parameter choices.
%}
%Time domain for channel
channelQuiet = channel(10541977, zeros(size(t)));

subplot(4,6,5);
plot(t, channelQuiet);
xlabel('Time (s)');
ylabel('Amplitude');
title('Channel Pre-Transmission');

%Frequency domain for channel
frequencyVector2 = (0:N-1)*(fs/N);
xlim([0 2]);
fftChannelQuiet = abs(fft(channelQuiet));
subplot(4,6,6);
plot(frequencyVector2,(fftChannelQuiet));
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Channel Pre-Transmission');


%2.4
%{
Modulate your audio signal using the carrier frequency you have selected.
%}
%Begin Modulation
fc = 60000;
% Y = 0.1
AudioModulated = cos(2*pi*fc*t) .* audioClean;

%2.5
%{
Simulate the transmission of your modulated signal, providing it as input to the channel
function, and plot the frequency domain of the input and output signals.
%}
AudioModulated_FFT = fft(AudioModulated);
fVec = linspace(-fs/2, fs/2, length(t));
subplot(4,6,7);
plot(fVec, abs(AudioModulated_FFT));
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('AudioModulated Input');
AudioModulatedOutput = channel(10541977, AudioModulated);
AudioModulatedOutput_FFT = fft(AudioModulatedOutput);
subplot(4,6,8);
plot(fVec, abs(AudioModulatedOutput_FFT));
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('AudioModulated Output');

%2.6
%{
Demodulate your audio signal from the channel output created in 2.5. View the demodu-
lated signal in the frequency domain. Filter the demodulated signal to isolate your audio
signal. Use the lowpass function in MATLAB to simulate an analogue filter, and store
the received audio as audioReceived.
%}
AudioDeModulated = AudioModulatedOutput .* cos(2*pi*fc*t);
f_demod = linspace(-fs/2, fs/2, length(AudioDeModulated));
AudioDeModulated_FFT = fft(AudioDeModulated);
subplot(4,6,9);
plot(f_demod, abs(AudioDeModulated_FFT));
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('AudioDeModulated');

%FILTER
audioReceived = lowpass(AudioDeModulated, 100, fc);
audioReceived = real(audioReceived);
audioReceived = audioReceived / max(abs(audioReceived));
% Check for clipping
if max(abs(audioReceived)) > 1
    disp('Clipping detected');
else
    disp('No clipping detected');
end

subplot(4,6,10);
plot(t, audioReceived);
xlabel('Time (s)');
ylabel('Amplitude');
title('Filtered + Demod');

%3.1
%{
Use MATLAB’s resample function to resample your received audio signal at a valid
sampling rate (from Table 1) closest to its Nyquist rate. Store your new sampling rate as
fs2 and your resampled audio as audioResampled.
%}

fs2 = 48000;
audioResampled = resample(audioReceived, fs2, fs);

%3.2
%{
Listen to and comment on the resampled audio.
%}



%3.3
%{
With an appropriate quantiser (mid-tread or mid-riser), quantise audioResampled
using 16 quantisation levels and store the result as audioQuantised. Listen to and
plot the quantised audio and comment on any changes.
%}
N = length(audioResampled);
AudioInSeconds=(length(audioResampled)/fs2);
t = linspace(0, AudioInSeconds, N);
xmax = 1; xmin = -1;
L = 16;
deltaMR = (xmax-xmin/L);
deltaMT = (xmax-xmin/L-1);
audioMT = deltaMT* floor(audioResampled/deltaMT + 1/2);
audioMR = deltaMR* (floor(audioResampled/deltaMR) + 1/2);
audioMT(audioMT>=xmax)= xmin+deltaMT*(L-1);
audioMR(audioMR>=xmax)= xmin+deltaMR*(L-1/2);

subplot(4,6,11); hold on; grid on;
plot(t, audioMR);
plot(t, audioResampled);
xlabel('Time (s)');
ylabel('Amplitude');
title('Mid-Riser: N=16');
subplot(4,6,12); hold on; grid on;
plot(t, audioMT);
plot(t, audioResampled);
xlabel('Time (s)');
ylabel('Amplitude');
title('Mid-Tread: N=16');




subplot(4,6,13);
plot(t, audioMR);
ylim([-1 1]);
xlabel('Time (s)');
ylabel('Amplitude');
title('AudioQuantised: MR');


subplot(4,6,14);
plot(t, audioMT);
ylim([-1 1]);
xlabel('Time (s)');
ylabel('Amplitude');
title('AudioQuantised: MT');
%3.4
%{
Experiment with using 2, 4, 8 and 32 quantisation levels. Listen to the quantised audio
for each case, and select an appropriate number of quantisation levels for the final system.
Justify your choice.
%}
%N=2
L = 2;
deltaMR = (xmax-xmin/L);
deltaMT = (xmax-xmin/L-1);
audioMT_2 = deltaMT* floor(audioResampled/deltaMT + 1/2);
audioMR_2 = deltaMR* (floor(audioResampled/deltaMR) + 1/2);

audioMT_2(audioMT_2>=xmax)= xmin+deltaMT*(L-1);
audioMR_2(audioMR_2>=xmax)= xmin+deltaMR*(L/2);

subplot(4,6,15); hold on; grid on;
plot(t, audioMR_2);
plot(t, audioResampled);
xlabel('Time (s)');
ylabel('Amplitude');
title('Mid-Riser: N=2');
subplot(4,6,16); hold on; grid on;
plot(t, audioMT_2);
plot(t, audioResampled);
xlabel('Time (s)');
ylabel('Amplitude');
title('Mid-Tread: N=2');

%N=4
L = 4;
deltaMR = (xmax-xmin/L);
deltaMT = (xmax-xmin/L-1);
audioMT_4 = deltaMT* floor(audioResampled/deltaMT + 1/2);
audioMR_4 = deltaMR* (floor(audioResampled/deltaMR) + 1/2);

audioMT_4(audioMT_4>=xmax)= xmin+deltaMT*(L-1);
audioMR_4(audioMR_4>=xmax)= xmin+deltaMR*(L-1/2);


subplot(4,6,17); hold on; grid on;
plot(t, audioMR_4);
plot(t, audioResampled);
xlabel('Time (s)');
ylabel('Amplitude');
title('Mid-Riser: N=4');
subplot(4,6,18); hold on; grid on;
plot(t, audioMT_4);
plot(t, audioResampled);
xlabel('Time (s)');
ylabel('Amplitude');
title('Mid-Tread: N=4');

%N=8
L = 8;
deltaMR = (xmax-xmin/L);
deltaMT = (xmax-xmin/L-1);
audioMT_8 = deltaMT* floor(audioResampled/deltaMT + 1/2);
audioMR_8 = deltaMR* (floor(audioResampled/deltaMR) + 1/2);

audioMT_8(audioMT_8>=xmax)= xmin+deltaMT*(L-1);
audioMR_8(audioMR_8>=xmax)= xmin+deltaMR*(L-1/2);


subplot(4,6,19); hold on; grid on;
plot(t, audioMR_8);
plot(t, audioResampled);
xlabel('Time (s)');
ylabel('Amplitude');
title('Mid-Riser: N=8');
subplot(4,6,20); hold on; grid on;
plot(t, audioMT_8);
plot(t, audioResampled);
xlabel('Time (s)');
ylabel('Amplitude');
title('Mid-Tread: N=8');

%N=32
L = 8;
deltaMR = (xmax-xmin/L);
deltaMT = (xmax-xmin/L-1);
audioMT_32 = deltaMT* floor(audioResampled/deltaMT + 1/2);
audioMR_32 = deltaMR* (floor(audioResampled/deltaMR) + 1/2);

audioMT_32(audioMT_32>=xmax)= xmin+deltaMT*(L-1);
audioMR_32(audioMR_32>=xmax)= xmin+deltaMR*(L-1/2);


subplot(4,6,21); hold on; grid on;
plot(t, audioMR_32);
plot(t, audioResampled);
xlabel('Time (s)');
ylabel('Amplitude');
title('Mid-Riser: N=32');
subplot(4,6,22); hold on; grid on;
plot(t, audioMT_32);
plot(t, audioResampled);
xlabel('Time (s)');
ylabel('Amplitude');
title('Mid-Tread: N=32');

function ApproxVector= ApproximationValues(HarmonicVector, cnVector, t, f0)
    nApprox = zeros(size(t));
    for i = 1:length(HarmonicVector)
        n = HarmonicVector(i);
        nApprox = nApprox + cnVector(i)*exp(1j*2*pi*n*f0*t);
    end
    ApproxVector = nApprox;
end

