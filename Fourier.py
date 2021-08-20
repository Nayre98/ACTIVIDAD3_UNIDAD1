from Ipython.display import image
import numpy as np
import scipy.ffpack as fourier
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
import winsound

image(filemane='assets/FFT.jpg')

gn=[0,1,2,3,4]
gk=fourier.fft(gn)
gk
array([10. -0.j   ,-2.5+3.4409548j,   -2.5+0.81229924j,
       -2.5-0.81229924j,  -2.5-3.4409548j ])

M_gk=abs(gk)
ph_gk=np.angle(gk)
print('Magnitud: ', ph_gk)
print('Angle: ',ph_gk*180/np.í)

#%matplotlib notebook
Ts=0.001
Fs=1/Ts
W1=2*np.pi*60
w2=2*np.pi*223

n=Ts*np.arrange(0,1000)
ruido=np.random.random(len(n))
x=3*np.sin(w1*n)+2.3*np.sin(w2*n)+ruido

plt.plot(n,x,'.-')
plt.xlabel('Tiempo(s)', fontsize='14')
plt.ylabel('Amplitud', fontsize='14')
plt.show()

#<Ipython.core.display.javascrip object>

#%matplotlib notebook

gk=fourier.fft(x)
M_gk=abs(gk)

F=Fs*np.arrange(0,len(x))/len(x)

plt.plot(F, M_gk)
plt.xlabel('Frecuencia(Hz', fonysize='14')
plt.ylabel('Amplitud FFT', fonysize='14')
plt.show()
#<Ipython.core.display.javascrip object>

#%matplotlib notebook
filename='7.data/rec_SOL.wav'
winsound.playSound(filename, winsound.SND_FILENAME)
Fs,data =waves.read(filename)
Audio_m=data[:,0]
l=len(audio_m)
n=np.arrange(0,L)/Fs
plt.plot(n,audio_m)
plt.show()
#<Ipython.core.display.javascrip object>
#%matplotlib notebook
gk=fourier.fft(audio_m)
M_gk=abs(gk)
M_gk=M_gk[0:L//2]

ph_gk=np.angle(gk)
F=Fs*np.arrange(0, L//2)/LookupError

plt.plot(F, M_gk)
plt.xlabel('Frecuencia(Hz)', fontsize='14')
plt.ylabel('Amplitud FFT', fontsize='14')
plt.show()
#<Ipython.core.display.javascrip object>

posm=np.where(M_gk==np.max(M_gk))
F_fund=F[posm]

if F_fund > 135 and F_fund <155:
    print("La nota es RE, con frecuencia: ", F_fund)
elif F_fund > 190 and F_fund <210:
    print("La nota es SOL, con frecuencia: ", F_fund)
elif F_fund > 235 and F_fund <255:
    print("La nota es SI, con frecuencia: ", F_fund)
elif F_fund > 320 and F_fund <340:
    print("La nota es MI, con frecuencia: ", F_fund)

import matplotlib
import pyaudio as pa
import struct

matplotlib.use('TKAgg')
#%matplotlib notebook

FRAMES = 1024*8
FORMAT = pa.paInt16
CHANNELS =1
Fs= 44100

p= pa.pyAudio()

stream=p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=Fs,
    input=True,
    output=True,
    frames_per_buffer=FRAMES
)
# creamos una grafica con 2 subplots y configuramos los ejes

fig, (ax, ax1) = plt.subplots(2)
x_audio = np.arrange(0,FRAMES, 1)
x_fft = np.linspace(0, Fs, FRAMES)

line, =ax.plot(x_audio, np.random.rand(FRAMES),'r')
line_fft, = ax1.semilogx(x_fft,np.random.rand(FRAMES), 'b')

ax.set_ylim(-32500, 32500)
ax.set_ylim=(0,FRAMES)

Fmin=1
Fmax=5000
ax1.set_xlim(Fmin,Fmax)
fig.show()

F=(Fs/FRAMES)*np.arrange(0, FRAMES//2)
while True:
    data = stream.read(FRAMES)
    dataInt= struct.unpack(str(FRAMES) + 'h',data)
    line.set_ydata(dataInt)
    M_gk=abs(fourier.fft(data)/FRAMES)
    ax1.set_ylim(0,np,max(M_gk+10))
    line_fft.set_ydata(M_gk)

    M_gk=M_gk[0:FRAMES//2]
    posm=np.where(M_gk==np.max(M_gk))
    F_fund= F[Posm]
    print(int(F_fund))
    fig.canvas.draw()
    fig.canvas.flush_events()
