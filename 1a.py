import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time() # Used to calculate the runtime

# Constants
A = 1
T = 10**(-6)
omega_0 = 2*np.pi*10**5
phi = np.pi/8
n_0 = -256
N = 513
ITERATIONS = 100

# SNR and variance
SNR_list = np.array([-10, 0, 10, 20, 30, 40, 50, 60])
sigma2_list = (A**2)/(2*10**(SNR_list/10))

# CRLB
CRLB_omega = (12*(sigma2_list))/((A**2)*(T**2)*N*((N**2)-1))
P = (N*(N-1))/2
Q = (N*(N-1)*(2*N-1))/6
CRLB_phi = (12*sigma2_list*(n_0**2 * N + 2*n_0*P+Q))/(A**2 *N**2 * (N**2 - 1))

# Computes the population variance of the values in a with a given mean
def var(a, mean):
    var = 0
    for val in a:
        var += (val-mean)**2
    return var/(len(a))

# Computes the MLE a given number of times
def computeMLE(iterations):
    # Signal generation
    s = []
    for n in range(n_0, n_0+N):
        s.append(A*np.exp(np.complex(0,1)*(omega_0*n*T+phi)))

    M_list = [2**10, 2**12, 2**14, 2**16, 2**18, 2**20] # The values of M for the M-point FFT
    omega_hat_list_M = [] # Contains omega_hat per M per SNR for all of the iterations
    phi_hat_list_M = []   # Contains phi_hat per M per SNR for all of the iterations
    for M in M_list:
        omega_hat_list_SNR = [] # Contains omega_hat per M for each SNR
        phi_hat_list_SNR = []   # Contains phi_hat per M for each SNR
        for sigma2 in sigma2_list:
            omega_hat_list_i = [] # Contains omega_hat for iteration i
            phi_hat_list_i = []   # Contains phi_hat for iteration i
            for i in range(iterations):
                # Adds noise to signal
                x = s + (np.complex(1,1))*np.random.normal(0, np.sqrt(sigma2), size=N)

                # Computation of the estimators
                FFT = np.fft.fft(x, M)
                m_star = np.argmax(np.abs(FFT))
                omega_hat = (2*np.pi*m_star)/(M*T)
                phi_hat = np.angle(np.exp(np.complex(0,-1)*omega_hat*n_0*T)*FFT[m_star])
                omega_hat_list_i.append(omega_hat)
                phi_hat_list_i.append(phi_hat)

            omega_hat_list_SNR.append(omega_hat_list_i)
            phi_hat_list_SNR.append(phi_hat_list_i)

        omega_hat_list_M.append(omega_hat_list_SNR)
        phi_hat_list_M.append(phi_hat_list_SNR)
    return omega_hat_list_M, phi_hat_list_M

# Obtaining the MLEs. The values are sorted per M per SNR
omega_hat_list, phi_hat_list = computeMLE(ITERATIONS)

# Computes the variance and mean error of the estimates
omega_var = []    
omega_error = [] 
phi_var = []  
phi_error = []
for M in range(len(omega_hat_list)):
    omega_var_SNR = []
    phi_var_SNR = []
    omega_error_SNR = []
    phi_error_SNR = []
    for SNR in range(len(omega_hat_list[M])):
        omega_error_SNR.append(omega_0 - np.mean(omega_hat_list[M][SNR]))
        phi_error_SNR.append(phi - np.mean(phi_hat_list[M][SNR]))
        omega_var_SNR.append(var(omega_hat_list[M][SNR], omega_0))
        phi_var_SNR.append(var(phi_hat_list[M][SNR], phi))
    omega_var.append(omega_var_SNR)
    phi_var.append(phi_var_SNR)
    omega_error.append(omega_error_SNR)
    phi_error.append(phi_error_SNR)

# Prints the runtime
print("Runtime: " + "{:.0f}".format(time.time() - start_time) + " seconds")

# Plot for the mean error of omega
for i in range(len(omega_error)):
    plt.plot(SNR_list, omega_error[i], label="M = 2^" + str(2*i+10))
plt.legend()
plt.xlabel("SNR [dB]")
plt.ylabel("Frequency [Hz]")
plt.title("The mean error of " + str(ITERATIONS) + " iterations of omega_hat")
plt.grid(axis="y")
plt.show()
plt.clf()

# Plot for the variance of omega
for i in range(len(omega_var)):
    plt.plot(SNR_list, omega_var[i], label="M = 2^" + str(2*i+10))
plt.plot(SNR_list, CRLB_omega, label="CRLB", color="darkred", linestyle="dashed")
plt.yscale("log")
plt.legend()
plt.xlabel("SNR [dB]")
plt.ylabel("Variance [Hz^2]")
plt.title("The variance of " + str(ITERATIONS) + " iterations of omega_hat")
plt.grid(axis="y")
plt.show()
plt.clf()

# Plot for the mean error of phi
for i in range(len(phi_error)):
    plt.plot(SNR_list, phi_error[i], label="M = 2^" + str(2*i+10))
plt.legend()
plt.xlabel("SNR [dB]")
plt.ylabel("Angle [rad]")
plt.title("The mean error of " + str(ITERATIONS) + " iterations of phi_hat")
plt.grid(axis="y")
plt.show()
plt.clf()

# Plot for the variance of phi
for i in range(len(phi_var)):
    plt.plot(SNR_list, phi_var[i], label="M = 2^" + str(2*i+10))
plt.plot(SNR_list, CRLB_phi, label="CRLB", color="darkred", linestyle="dashed")
plt.yscale("log")
plt.legend()
plt.xlabel("SNR [dB]")
plt.ylabel("Variance [rad^2]")
plt.title("The variance of " + str(ITERATIONS) + " iterations of phi_hat")
plt.grid(axis="y")
plt.show()
plt.clf()