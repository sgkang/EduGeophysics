import numpy as np
import matplotlib.pyplot as plt
from DigitalFilter import setFrequency, transFiltImpulse

def ColeColeFplot(siginf, eta, tau, c, fmin, fmax):
    eta = eta
    tau = 10**(tau)
    c = c
    siginf = 10**(siginf)
    omega = np.logspace(fmin, fmax, 56)*2.*np.pi
    sigmaColeF = siginf*np.ones(omega.size)-siginf*eta*(1./(1.+(1.-eta)*(1j*omega*tau)**c))

    time = np.logspace(-6, 0, 64)
    wt, tbase, omega_int = setFrequency(time)
    sigmaColeF_temp = siginf-siginf*eta*(1./(1.+(1.-eta)*(1j*omega_int*tau)**c))
    sigmaColeT = transFiltImpulse(sigmaColeF_temp, wt, tbase, omega_int, time, tol=1e-12)
    fig, ax = plt.subplots(1,2, figsize = (16, 5))
    ax[0].semilogx(omega/2./np.pi, sigmaColeF.real, 'ko-')
    ax[0].semilogx(omega/2./np.pi, sigmaColeF.imag, 'ro-')
    ax[0].set_ylim(0., siginf*1.2)
    ax[0].grid(True)
    ax[0].set_xlabel("Frequency (Hz)", fontsize = 16)
    ax[0].set_ylabel("Frequency domain Cole-Cole (S/m)", fontsize = 16)

    ax[1].semilogx(time, sigmaColeT.real, 'ko-')
    ax[1].set_ylim(-siginf*0.5*time.min()**(-0.75), 0.)
    ax[1].grid(True)
    ax[1].set_xlabel("Time (s)", fontsize = 16)
    ax[1].set_ylabel("Time domain Cole-Cole (S/m)", fontsize = 16)
    ax[0].set_title("$\sigma_{\infty}$= "+str(siginf)+"$, \eta = $"+str(eta)+"$, \
                     \\tau = $"+str(tau)+"and c="+str(c), fontsize = 14)

if __name__ == '__main__':
    ColeColeFplot(0.1, 0.1, 0.01, 0.5, -5, 5)
