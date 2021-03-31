import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import time

start_time = time.time() # Used to calculate the runtime

# Constants
A = 1
T = 10**(-6)
omega_0 = 2*np.pi*10**5
phi = np.pi/8
n_0 = -256
N = 513
M = 2**10
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

# Mean Square Error
def mse(a1, a2):
    return np.mean(np.square(np.subtract(a1, a2)))

# Function to be minimized by scipy.optimize.minimize
def functionToBeMinimized(a):
    omega_guess = a[0]
    phi_guess = a[1]
    # Generates a signal with the guessed frequency without noise
    s_guess = []
    for n in range(n_0, n_0+N):
        s_guess.append(A*np.exp(np.complex(0,1)*(omega_guess*n*T+phi_guess)))
    sFFT = np.fft.fft(s_guess, M)
    # We minimize the mean square error between sFFT and xFFT
    return mse(np.concatenate([np.real(sFFT), np.imag(sFFT), np.abs(sFFT), np.angle(sFFT)]), np.concatenate([np.real(xFFT), np.imag(xFFT), np.abs(xFFT), np.angle(xFFT)]))

# Computation of omega_hat
omega_hat_list = []
phi_hat_list = []
for i in range(len(sigma2_list)):
    # Signal generation
    s = []
    for n in range(n_0, n_0+N):
        s.append(A*np.exp(np.complex(0,1)*(omega_0*n*T+phi)))
    x = s + (np.complex(1,1))*np.random.normal(0, np.sqrt(sigma2_list[i]), size=N)
    xFFT = np.fft.fft(x, M) # Computes the FFT of the original signal with noise
    
    # Initial guess for the frequency and phase, using the same approach as in 1a
    m_star = np.argmax(np.abs(xFFT))
    omega_guess = m_star*(2*np.pi)/(M*T)
    phi_guess = np.angle(np.exp(np.complex(0,-1)*omega_guess*n_0*T)*xFFT[m_star])

    omega_hat_row = []
    phi_hat_row = []
    for i in range(ITERATIONS):
        # Numerically minimises the MSE until it hits a threshhold or has ran for 20 iterations
        result = scipy.optimize.minimize(functionToBeMinimized, [omega_guess, phi_guess], method="Nelder-Mead", options={"maxiter": 20}) 
        omega_hat_row.append(result.x[0])
        phi_hat_row.append(result.x[1])
    omega_hat_list.append(omega_hat_row)
    phi_hat_list.append(phi_hat_row)

# Computation of the mean error and variance of the estimators sorted per SNR
omega_error_list = []
omega_var_list = []
phi_error_list = []
phi_var_list = []
for i in range(len(omega_hat_list)):
    omega_error_list.append(omega_0 - np.mean(omega_hat_list[i]))
    omega_var_list.append(var(omega_hat_list[i], omega_0))
    phi_error_list.append(phi - np.mean(phi_hat_list[i]))
    phi_var_list.append(var(phi_hat_list[i], phi))

# Prints the runtime
print("Runtime: " + "{:.0f}".format(time.time() - start_time) + " seconds")

# # Plot for the mean error of omega
plt.plot(SNR_list, omega_error_list, label="omega_0 - omega_hat")
plt.legend()
plt.xlabel("SNR [dB]")
plt.ylabel("Frequency [Hz]")
plt.title("The mean error of " + str(ITERATIONS) + " iterations of omega_hat")
plt.grid(axis="y")
plt.show()
plt.clf()

# Plot for the variance of omega
plt.plot(SNR_list, omega_var_list, label="Var[omega_hat]")
plt.plot(SNR_list, CRLB_omega, linestyle="dashed", label="CRLB")
plt.legend()
plt.yscale("log")
plt.xlabel("SNR [dB]")
plt.ylabel("Variance [Hz^2]")
plt.title("The variance of " + str(ITERATIONS) + " iterations of omega_hat")
plt.grid(axis="y")
plt.show()

# Plot for the mean error of phi
plt.plot(SNR_list, phi_error_list, label="phi - phi_hat")
plt.legend()
plt.xlabel("SNR [dB]")
plt.ylabel("Frequency [Hz]")
plt.title("The mean error of " + str(ITERATIONS) + " iterations of phi_hat")
plt.grid(axis="y")
plt.show()
plt.clf()

# Plot for the variance of phi
plt.plot(SNR_list, phi_var_list, label="Var[phi_hat]")
plt.plot(SNR_list, CRLB_phi, linestyle="dashed", label="CRLB")
plt.legend()
plt.yscale("log")
plt.xlabel("SNR [dB]")
plt.ylabel("Variance [Hz^2]")
plt.title("The variance of " + str(ITERATIONS) + " iterations of phi_hat")
plt.grid(axis="y")
plt.show()