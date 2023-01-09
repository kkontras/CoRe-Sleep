import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import random
import torch
import einops

###################################################################
# Wake
def my_wave(A,omega,t):
    x_volts = A*np.sin(omega*t)
    return x_volts

def get_wake(fs, N, split):
    t = np.arange(N) / float(fs)
    t_split = np.split(t, split)

    A = []
    for i in range(t_split.__len__()):
        a = (4-0.5)*random.random()-0.5  # I am not sure about this values
        A = np.append(A, a)  # Also not sure about integers

    alphas = random.randint(6,10) #>50%
    omega = []
    for i in range(alphas):  # 60-40 sample play also with this!
        o = (13-8)*random.random()-8
        omega = np.append(omega, o)

    thetas = 10 - alphas
    for i in range(thetas):
        o = (7-4)*random.random()-4
        omega = np.append(omega, o)

    random.shuffle(omega)

    x_volts = []
    for i in range(t_split.__len__()):
        x = my_wave(A[i], omega[i], t_split[i])
        x_volts = np.append(x_volts, x)

    return x_volts

def get_n3(fs, N, split):
    t = np.arange(N) / float(fs)
    t_split = np.split(t, split)

    A = []
    for i in range(t_split.__len__()):
        a = (4-0.5)*random.random()-0.5  # I am not sure about this values
        A = np.append(A, a)  # Also not sure about integers

    deltas = random.randint(3, 10)  # >20%
    omega = []
    for i in range(deltas):
        o = (4-0.5)*random.random()-0.5
        omega = np.append(omega, o)

    thetas = 10 - deltas
    for i in range(thetas):
        o = (7-4)*random.random()-4
        omega = np.append(omega, o)

    random.shuffle(omega)

    x_volts = []
    for i in range(t_split.__len__()):
        x = my_wave(A[i], omega[i], t_split[i])
        x_volts = np.append(x_volts, x)

    return x_volts

def signal_with_noise(x_volts, snr):
    x_watts = x_volts ** 2
    # Adding noise using target SNR
    # Set a target SNR
    target_snr_db = snr
    # Calculate signal power and convert to dB
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
    # Noise up the original signal
    y_volts = x_volts + noise_volts
    return y_volts

fs = 100
N=2800
split = 10
wake_clean = get_wake(fs, N, split)
snr = 10
wake_volts = signal_with_noise(wake_clean, snr)
t = np.arange(N) / float(fs)

plt.plot(t, wake_volts)
plt.title('Wake')
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
plt.show()


###############################################################################
# N3
n3_clean = get_n3(fs, N, split)
snr = 20
n3_volts = signal_with_noise(n3_clean, snr)
t = np.arange(N) / float(fs)

plt.plot(t, n3_volts)
plt.title('N3')
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
plt.show()

######################################################################
# Data set
n = 336
fs = 100
N=2800
split = 10
#Wake sampling
snr = 10
wake_volts = []
for i in range(n):
    wake_clean = get_wake(fs, N, split)
    noise = signal_with_noise(wake_clean, snr)
    wake_volts = np.append(wake_volts, noise, axis=0)
#wake_volts.resize((1000, 336))
wake_volts.shape = (n, N)

# N3 sampling
snr = 15
n3_volts = []
for i in range(n):
    n3_clean = get_n3(fs, N, split)
    noise = signal_with_noise(n3_clean, snr)
    n3_volts = np.append(n3_volts, noise, axis=0)
#wake_volts.resize((1000, 336))
n3_volts.shape = (n, N)

twake_volts = torch.from_numpy(wake_volts)
tn3_volts = torch.from_numpy(n3_volts)
wake_labels = torch.zeros(n, dtype=int)
n3_labels = torch.ones(n, dtype=int)
# 0 FOR WAKE, 1 FOR N3
wake_n3_volts = torch.cat((twake_volts, tn3_volts), 0)
labels = torch.cat((wake_labels, n3_labels))


torch.manual_seed(0)
wake_n3_volts=wake_n3_volts[torch.randperm(wake_n3_volts.size()[0])]
torch.manual_seed(0)
labels=labels[torch.randperm(labels.size()[0])]

############################################################################
# STFT
x_watts = wake_n3_volts ** 2
x_db = 10 * np.log10(x_watts)

data = torch.zeros(672,129,23)
for i in range(2*n):
    f,t,Zxx = scipy.signal.stft(x_db[i], fs=100, nperseg = 256)
    data[i] = torch.from_numpy(np.abs(Zxx))

#labels[0] =1
t = np.arange(0, data.shape[-1])
f = np.arange(0, data.shape[-2])
plt.pcolormesh(t, f, data[0], vmin=wake_n3_volts.min(),
               vmax=wake_n3_volts.max(), shading='gouraud')
plt.title('STFT Magnitude for N3')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar()
plt.show()

#labels[2] =0
plt.pcolormesh(t, f, data[2], vmin=wake_n3_volts.min(),
               vmax=wake_n3_volts.max(), shading='gouraud')
plt.title('STFT Magnitude for Wake')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar()
plt.show()

data = einops.rearrange(data, '(b m) f t -> b m f t', b=32, m=21)


###############################################################################
