
import time
import sys
import os

import numpy as np
from numpy import cos, sin, zeros, linspace, loadtxt, eye, arange, exp, maximum
from numpy.lib.scimath import sqrt, log
from numpy.random import rand

from scipy import dot, float64, ones, array, float64, outer, pi
from scipy.linalg import det, eigh, inv, expm
from scipy.special import logsumexp, digamma
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import pylab
# from matplotlib import pyplot as pl

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--random_seed", help = "random seed", type = int, default = 0)
parser.add_argument("--max_annealing_01", help = "max annealing 01", type = int, default = 10)
parser.add_argument("--max_annealing_02", help = "max annealing 02", type = int, default = 50)
parser.add_argument("--N_iter_start", help = "N iter start", type = int, default = 0)
parser.add_argument("--N_iter_end", help = "N iter end", type = int, default = 1)
parser.add_argument("--num_class", help = "num class", type = int, default = 20)
parser.add_argument("--num_max_iter", help = "num max iter", type = int, default = 3000)
parser.add_argument("--beta_ini", help = "beta_ini", type = float, default = 30.000)
parser.add_argument("--s_ini", help = "s_ini", type = float, default = 1.000)
parser.add_argument("--mode_E", help = "mode E: QA, SA, VB", type = str, default = "QA")
parser.add_argument("--mode_M", help = "mode M: SA, VB", type = str, default = "SA")
parser.add_argument("--mode_QA", help = "mode QA: Farhi, Kadowaki", type = str, default = "Farhi")
parser.add_argument("--mode_write", help = "mode write: ON, OFF", type = str, default = "ON")
parser.add_argument("--mode_quantum_term", help = "mode quantum term: sub_diagonal, all_one", type = str, default = "sub_diagonal")
parser.add_argument("--mode_read_input_file", help = "read input file: read, random, uniform", type = str, default = "random")
parser.add_argument("--directory_dataset", help = "directory dataset", type = str, default = "dataset_input_dim_002_v_001_n_0100_seed_0000")

args = parser.parse_args()

random_seed = args.random_seed
max_annealing_01 = args.max_annealing_01
max_annealing_02 = args.max_annealing_02
N_iter_start = args.N_iter_start
N_iter_end = args.N_iter_end
num_class = args.num_class
num_max_iter = args.num_max_iter
beta_ini = args.beta_ini
s_ini = args.s_ini
mode_E = args.mode_E
mode_M = args.mode_M
mode_QA = args.mode_QA
mode_write = args.mode_write
mode_quantum_term = args.mode_quantum_term
mode_read_input_file = args.mode_read_input_file
directory_dataset = args.directory_dataset

#
if max_annealing_02 > 0:
    pass
else:
    print("max_annealing must be 1 or larger.")
    sys.exit()

#
np.random.seed(random_seed)  # global optimum
N_iter = N_iter_end - N_iter_start
beta_ini_E = beta_ini
beta_ini_M = beta_ini
s_ini_E = s_ini

criteria_break = 10 ** (-10)
criteria_QA = 10 ** (-10)
directory_input = "."

#
directory_output_main = "output_VB"
if not os.path.exists(directory_output_main):
    os.makedirs(directory_output_main)

directory_output_sub = "output_" + directory_dataset + "/" + "output_nc_" + str(num_class).zfill(4) + "_max_" + str(num_max_iter).zfill(8) + "_as1_" + str(max_annealing_01).zfill(5) + "_as2_" + str(max_annealing_02).zfill(5) + "_s0_" + "{0:07.4f}".format(s_ini) + "_beta0_" + "{0:08.4f}".format(beta_ini)
if not os.path.exists(directory_output_main + "/" + directory_output_sub):
    os.makedirs(directory_output_main + "/" + directory_output_sub)

####

def faithful():
    try:
        return loadtxt("./datasets/" + directory_dataset + "/mixture_of_gauss_001_obs.txt")
    except IOError:
        print("Data cannot be read!")
        sys.exit()

# def init_figure():
#     print("Here!")
#     pylab.ion()
#     figsize = [8, 6]
#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_subplot(111)
#     return ax

# def preview_stage(ax, x, pi, mu, var):
#     num_class = pi.shape[0]
#     ax.clear()
#     ax.plot(x[:, 0], x[:, 1], '+')
# 
#     weight = pi * 2
#     for k in range(num_class):
#         if pi[k] > 10 ** (-3):
#             ax.plot(mu[k, 0], mu[k, 1], 'o')
#             rbuf = make_ring(var[k], ndiv=50)
#             ax.plot(rbuf[:, 0] + mu[k, 0], rbuf[:, 1] + mu[k, 1], 'b')
#             ax.plot(rbuf[:, 0] * weight[k] + mu[k, 0],
#                     rbuf[:, 1] * weight[k] + mu[k, 1], '0.8')
#     # ax.set_xlim(-3, 3)
#     # ax.set_xlim(-2, 2)
#     # ax.set_ylim(-3, 3)
#     ax.set_xlim(-10, 20)
#     ax.set_ylim(-20, 20)
#     # ax.set_xlim(-5, 5)
#     # ax.set_ylim(-15, 15)
#     ax.figure.canvas.draw()
#     ax.figure.canvas.draw()

def faithful_norm():
    dat = faithful()
    return dat

def unit_ring(ndiv=20):
    angles = linspace(0, 2 * pi, ndiv)
    ring = zeros([ndiv, 2], dtype=float64)
    ring[:, 0] = cos(angles)
    ring[:, 1] = sin(angles)
    return ring


def calc_unit(cov):
    assert cov.ndim == 2
    (vals, vecs) = eigh(cov)
    vals = sqrt(vals)

    buf = zeros([2, 2], dtype=float64)
    buf[0] = vals[0] * vecs[0]
    buf[1] = vals[1] * vecs[1]
    return buf


def make_ring(cov, ndiv=20):
    ring = unit_ring(ndiv)
    units = calc_unit(cov)
    buf = zeros([ndiv, 2], dtype=float64)
    for i in range(ndiv):
        buf[i, :] = units[0, :] * ring[i, 0] + units[1, :] * ring[i, 1]
    return buf

def expect_pi(alpha):
    return alpha / alpha.sum()

def expect_lpi(alpha):
    return digamma(alpha) - digamma(alpha.sum())

def expect_llambda(W, nu):
    ndim = W.shape[0]
    arr = np.array(float64(nu - arange(ndim)) / 2)
    return digamma(arr).sum() + ndim * log(2) + log(det(W))

def expect_lambda(W, nu):
    return W * nu

def quad(A, x):
    num, ndim = x.shape
    ret = zeros(num, dtype=float64)

    for i in range(ndim):
        for j in range(ndim):
            ret += A[i, j] * x[:, i] * x[:, j]
    return ret

def quad2(A, x):
    num, ndim = x.shape
    ret = zeros(num, dtype=float64)

    for i in range(num):
        ret[i] = x[i].dot(A).dot(x[i])
    return ret

def expect_quad(x, m, beta, W, nu):
    ndim = x.shape[1]
    return ndim / beta + nu * quad(W, x - m[None, :])

def expect_log(x, m, beta, W, nu):
    ndim = x.shape[1]

    ex_llambda = expect_llambda(W, nu)
    ex_quad = expect_quad(x, m, beta, W, nu)
    return (ex_llambda - ndim * log(2 * pi) - ex_quad) / 2

def normalize_response(lrho, beta_E, s_E, mode_Estep_local):
    num, num_class = lrho.shape
    ret = zeros([num, num_class], dtype=float64)

    if mode_Estep_local == "VB":
        for i in range(num):
            lr = lrho[i] - logsumexp(lrho[i])
            ret[i] = exp(lr)
    elif mode_Estep_local == "SA":
        for i in range(num):
            lrho_sa = lrho[i] * beta_E
            lr = lrho_sa - logsumexp(lrho_sa)
            ret[i] = exp(lr)
    elif mode_Estep_local == "QA":
        H1 = np.zeros((num_class, num_class))
        H2 = np.zeros((num_class, num_class))

        if mode_QA == "Kadowaki":
            print("Kadowaki ", end="")
            if mode_quantum_term == "sub_diagonal":
                for j in range(num_class):
                    H2[j, (j + 1) % num_class] = s_E
                    H2[(j + 1) % num_class, j] = s_E
            elif mode_quantum_term == "all_one":
                for j1 in range(num_class):
                    for j2 in range(num_class):
                        if j1 != j2:
                            H2[j1, j2] = s_E
            else:
                print("error")
                sys.exit()

        elif mode_QA == "Farhi":
            print("Farhi ", end="")
            mode_quantum_term = "sub_diagonal"
            if mode_quantum_term == "sub_diagonal":
                for j in range(num_class):
                    H2[j, (j + 1) % num_class] = 1.0
                    H2[(j + 1) % num_class, j] = 1.0
            elif mode_quantum_term == "all_one":
                for j1 in range(num_class):
                    for j2 in range(num_class):
                        if j1 != j2:
                            H2[j1, j2] = 1.0
            else:
                print("error")
                sys.exit()

        else:
            print("error")
            import sys
            sys.exti()

        chemical_pot_cl = 0.0
        chemical_pot_qu = 0.0

        if abs(s_E) < criteria_QA: 
            for i in range(num):
                lrho_sa = lrho[i] * beta_E
                lr = lrho_sa - logsumexp(lrho_sa)
                ret[i] = exp(lr)

        else:
            for i in range(num):
                for j in range(num_class):
                    H1[j, j] = - lrho[i][j]

                if mode_QA == "Kadowaki":
                    expH = expm(- beta_E * ((H1 - chemical_pot_cl * np.identity(num_class)) + (H2 - chemical_pot_qu * s_E * np.identity(num_class))))
                elif mode_QA == "Farhi":
                    expH = expm(- beta_E * ((1 - s_E) * (H1 - chemical_pot_cl * np.identity(num_class)) + s_E * (H2 - chemical_pot_qu * np.identity(num_class))))
                else:
                    import sys
                    sys.exit()

                sum_expH = 0.0
                for j in range(num_class):
                    sum_expH += expH[j, j]
                lr = np.array([0.0 for j in range(num_class)])
                for j in range(num_class):
                    lr[j] = log(expH[j, j]) - log(sum_expH)
                ret[i] = exp(lr)

    else:
        sys.exit()

    likelihood = 0.0
    for i in range(num):
        likelihood_dummy01 = 0.0
        for k in range(num_class):
            likelihood_dummy01 = likelihood_dummy01 + np.exp(lrho[i][k])
        likelihood = likelihood + np.log(likelihood_dummy01)

    ret = maximum(ret, 1e-10)
    ret /= ret.sum(1)[:, None]

    return ret

def calc_xbar(x, r_nk):
    num, ndim = x.shape
    num, num_class = r_nk.shape
    ret = zeros([num_class, ndim], dtype=float64)

    for k in range(num_class):
        clres = r_nk[:, k]
        for i in range(ndim):
            ret[k, i] = (clres * x[:, i]).sum()
        ret[k, :] /= clres.sum()

    return ret

def calc_S(x, xbar, r_nk):
    num, ndim = x.shape
    num, num_class = r_nk.shape
    ret = zeros([num_class, ndim, ndim], dtype=float64)

    for k in range(num_class):
        clres = r_nk[:, k]
        for i in range(ndim):
            diff_i = x[:, i] - xbar[k, i]
            for j in range(ndim):
                diff_j = x[:, j] - xbar[k, j]
                ret[k, i, j] = (clres * diff_i * diff_j).sum()
        ret[k] /= clres.sum()

    return ret


#
#   平均のハイパーパラメータを計算します．
#
def calc_m(xbar, Nk, m0, beta0, beta, beta_prior, beta_M):
    num_class, ndim = xbar.shape
    ret = zeros([num_class, ndim], dtype=float64)

    for k in range(num_class):
        ret[k] = (beta_prior * beta0 * m0 + beta_M * Nk[k] * xbar[k]) / beta[k]

    return ret

def calc_W(xbar, Sk, Nk, m0, beta0, inv_W0, beta_prior_local, beta_M_local):
    num_class, ndim = xbar.shape
    ret = zeros([num_class, ndim, ndim], dtype=float64)

    for k in range(num_class):
        ret[k] = beta_prior_local * inv_W0 + beta_M_local * Nk[k] * Sk[k]
        fact = beta0 * Nk[k] / (beta0 + Nk[k])
        diff = xbar[k] - m0
        for i in range(ndim):
            for j in range(ndim):
                term = diff[i] * diff[j]
                ret[k, i, j] += beta_M_local * beta_prior_local * fact * term
        ret[k] = inv(ret[k])

    return ret

def estimate(x, iteration):

    num, ndim = x.shape

    # # Graph
    # if iteration == 0:
    #     print("YES")
    #     preview_on = 1
    # else:
    #     preview_on = 0

    # if preview_on == 1:
    #     print("YES", preview_on)
    #     ax = init_figure()

    alpha0 = 1e-3
    beta0 = 1e-3
    m0 = zeros(ndim)
    W0 = eye(ndim)
    nu0 = ndim - 1.0 # 1.0
    inv_W0 = inv(W0)

    if nu0 < ndim - 1:
        print("inappropriate nu0 is set!")
        import sys
        sys.exit()

    if mode_read_input_file == "read":
        r_nk = zeros((num, num_class), dtype=float64)
        try:
            f_input_02 = open(directory_input + '/input_current/output_vbgmm_002_' + str(iteration).zfill(4) + '.txt')
        except IOError:
            print("Data cannot be read!")
            sys.exit()

        line = f_input_02.readlines() 

        for k in range(num):
            for i in range(num_class):
                line_splitted = line[k].split()
                r_nk[k][i] = line_splitted[i]
        f_input_02.close
    elif mode_read_input_file == "random":
        r_nk = rand(num, num_class)
        r_nk /= r_nk.sum(1)[:, None]
    elif mode_read_input_file == "uniform":
        r_nk = ones((num, num_class))
        r_nk /= r_nk.sum(1)[:, None]
    else:
        import sys
        sys.exit()

    if mode_write == "ON":
        f_output_002 = open(directory_output_main + "/" + directory_output_sub + "/output_vbgmm_002_" + str(iteration).zfill(4) + ".txt", "w")

        f_output_005_mu = open(directory_output_main + "/" + directory_output_sub + "/output_vbgmm_005_mu_" + str(iteration).zfill(4) + ".txt", "w")
        f_output_005_var = open(directory_output_main + "/" + directory_output_sub + "/output_vbgmm_005_var_" + str(iteration).zfill(4) + ".txt", "w")
        f_output_005_pi = open(directory_output_main + "/" + directory_output_sub + "/output_vbgmm_005_pi_" + str(iteration).zfill(4) + ".txt", "w")

        f_output_006_alpha = open(directory_output_main + "/" + directory_output_sub + "/output_vbgmm_006_alpha_" + str(iteration).zfill(4) + ".txt", "w")
        f_output_006_beta = open(directory_output_main + "/" + directory_output_sub + "/output_vbgmm_006_beta_" + str(iteration).zfill(4) + ".txt", "w")
        f_output_006_nu = open(directory_output_main + "/" + directory_output_sub + "/output_vbgmm_006_nu_" + str(iteration).zfill(4) + ".txt", "w")
        f_output_006_m = open(directory_output_main + "/" + directory_output_sub + "/output_vbgmm_006_m_" + str(iteration).zfill(4) + ".txt", "w")
        f_output_006_W = open(directory_output_main + "/" + directory_output_sub + "/output_vbgmm_006_W_" + str(iteration).zfill(4) + ".txt", "w")

        f_output_009 = open(directory_output_main + "/" + directory_output_sub + "/output_vbgmm_009_" + str(iteration).zfill(4) + ".txt", "w")

        for k in range(num):
            for i in range(num_class):
                f_output_002.write(str(r_nk[k][i]))
                f_output_002.write(" ")
            f_output_002.write("\n")

        f_output_006_alpha.write(str(0))
        f_output_006_alpha.write(" ")
        f_output_006_alpha.write(str(alpha0))
        f_output_006_alpha.write("\n")

        f_output_006_beta.write(str(0))
        f_output_006_beta.write(" ")
        f_output_006_beta.write(str(beta0))
        f_output_006_beta.write("\n")

        f_output_006_nu.write(str(0))
        f_output_006_nu.write(" ")
        f_output_006_nu.write(str(nu0))
        f_output_006_nu.write("\n")

        f_output_006_m.write(str(0))
        f_output_006_m.write(" ")
        for i in range(ndim):
            f_output_006_m.write(str(m0[i]))
            f_output_006_m.write(" ")
        f_output_006_m.write("\n")

        f_output_006_W.write(str(0))
        f_output_006_W.write(" ")
        for i in range(ndim):
            for j in range(ndim):
                f_output_006_W.write(str(W0[i][j]))
                f_output_006_W.write(" ")
            f_output_006_W.write(" ")
        f_output_006_W.write("\n")

    likelihood_before = -1000000

    for iiter in range(num_max_iter+1):
        print(" ")
        print(N_iter, " ", end="")
        print("{0:4d}".format(iteration), " ", end="")
        print(num_max_iter, " ", end="")
        print("{0:4d}".format(iiter), " ", end="")

        iteration_annealing_01 = max_annealing_01
        iteration_annealing_02 = max_annealing_01 + max_annealing_02

        s_E = s_ini_E * max(1.0 - 1.0 * iiter / iteration_annealing_01, 0.0)

        if iiter < iteration_annealing_01:
            beta_E = beta_ini_E
            beta_M = beta_ini_M
        else:
            beta_E = 1.0 + (beta_ini_E - 1.0) * max((1.0 - 1.0 * (iiter - iteration_annealing_01) / (iteration_annealing_02 - iteration_annealing_01)), 0.0)
            beta_M = 1.0 + (beta_ini_M - 1.0) * max((1.0 - 1.0 * (iiter - iteration_annealing_01) / (iteration_annealing_02 - iteration_annealing_01)), 0.0)
            
        beta_ini_prior = 1.0
        beta_prior = 1.0

        print("a1", "{0:4d}".format(max_annealing_01), " ", end="")
        print("a2", "{0:4d}".format(max_annealing_02), " ", end="")

        print("bE", "{0:2.4f}".format(beta_ini_E), " ", end="")
        print("{0:2.4f}".format(beta_E), " ", end="")

        print("bM", "{0:2.4f}".format(beta_ini_M), " ", end="")
        print("{0:2.4f}".format(beta_M), " ", end="")

        print("GE", "{0:2.4f}".format(s_ini_E), " ", end="")
        print("{0:2.4f}".format(s_E), " ", end="")

        print("mE ", mode_E, " ", end="")
        print("mM ", mode_M, " ", end="")

        if mode_M == "VB":
            Nk = r_nk.sum(0)
            alpha = alpha0 + Nk

            xbar = calc_xbar(x, r_nk)
            Sk = calc_S(x, xbar, r_nk)
            beta = beta0 + Nk
            mk = calc_m(xbar, Nk, m0, beta0, beta, 1.0, 1.0)
            W = calc_W(xbar, Sk, Nk, m0, beta0, inv_W0, 1.0, 1.0)
            nu = nu0 + Nk

        elif mode_M == "SA":
            Nk = r_nk.sum(0)
            alpha = beta_prior * alpha0 + beta_M * Nk - beta_prior + 1.0

            xbar = calc_xbar(x, r_nk)
            Sk = calc_S(x, xbar, r_nk)
            beta = beta_prior * beta0 + beta_M * Nk
            mk = calc_m(xbar, Nk, m0, beta0, beta, beta_prior, beta_M)
            W = calc_W(xbar, Sk, Nk, m0, beta0, inv_W0, beta_prior, beta_M)
            nu = beta_prior * nu0 + beta_M * Nk + (1.0 - beta_prior) * ndim

        elif mode_M == "QA":
            import sys
            sys.exit()

        else:
            import sys
            sys.exit()

        ex_lpi = expect_lpi(alpha)
        ex_log = zeros([num, num_class], dtype=float64)
        for k in range(num_class):
            ex_log[:, k] = expect_log(x, mk[k], beta[k], W[k], nu[k])
        lrho = ex_lpi[None, :] + ex_log
        r_nk = normalize_response(lrho, beta_E, s_E, mode_E)

        likelihood = 0.0
        for i in range(num):
            likelihood_dummy01 = 0.0
            for k in range(num_class):
                likelihood_dummy01 = likelihood_dummy01 + np.exp(lrho[i][k])
            likelihood = likelihood + np.log(likelihood_dummy01)

        print("LH", "{0:2.5f}".format(likelihood), " ", end="")

        ex_pi = expect_pi(alpha)
        ex_lambda = zeros([num_class, ndim, ndim], dtype=float64)
        for k in range(num_class):
            ex_lambda[k] = expect_lambda(W[k], nu[k])

        inv_ex_lambda = zeros([num_class, ndim, ndim], dtype=float64)
        for k in range(num_class):
            inv_ex_lambda[k] = inv(ex_lambda[k])

        num_class_current = 0
        for i in range(ex_pi.shape[0]):
            if ex_pi[i] > 10 ** (-3):
                num_class_current = num_class_current + 1
        print("num_cl", "{0:2d}".format(num_class_current), end="")

        # if preview_on == 1:
        #     print("YES YES !!")
        #     preview_stage(ax, x, ex_pi, mk, inv_ex_lambda)

        if mode_write == "ON":
            f_output_005_mu.write(str(iiter + 1))
            f_output_005_mu.write(" ")
            for k in range(num_class):
                for i in range(ndim):
                    f_output_005_mu.write(str(mk[k][i]))
                    f_output_005_mu.write(" ")
            f_output_005_mu.write("\n")

            f_output_005_var.write(str(iiter + 1))
            f_output_005_var.write(" ")
            for k in range(num_class):
                for i in range(ndim):
                    for j in range(ndim):
                        f_output_005_var.write(str(inv_ex_lambda[k][i][j]))
                        f_output_005_var.write(" ")
            f_output_005_var.write("\n")

            f_output_005_pi.write(str(iiter + 1))
            f_output_005_pi.write(" ")
            for k in range(num_class):
                f_output_005_pi.write(str(ex_pi[k]))
                f_output_005_pi.write(" ")
            f_output_005_pi.write("\n")

            f_output_006_alpha.write(str(iiter + 1))
            f_output_006_alpha.write(" ")
            for k in range(num_class):
                f_output_006_alpha.write(str(alpha[k]))
                f_output_006_alpha.write(" ")
            f_output_006_alpha.write("\n")

            f_output_006_beta.write(str(iiter + 1))
            f_output_006_beta.write(" ")
            for k in range(num_class):
                f_output_006_beta.write(str(beta[k]))
                f_output_006_beta.write(" ")
            f_output_006_beta.write("\n")

            f_output_006_nu.write(str(iiter + 1))
            f_output_006_nu.write(" ")
            for k in range(num_class):
                f_output_006_nu.write(str(nu[k]))
                f_output_006_nu.write(" ")
            f_output_006_nu.write("\n")

            f_output_006_m.write(str(iiter + 1))
            f_output_006_m.write(" ")
            for k in range(num_class):
                for i in range(ndim):
                    f_output_006_m.write(str(mk[k][i]))
                    f_output_006_m.write(" ")
                f_output_006_m.write(" ")
            f_output_006_m.write("\n")

            f_output_006_W.write(str(iiter + 1))
            f_output_006_W.write(" ")
            for k in range(num_class):
                for i in range(ndim):
                    for j in range(ndim):
                        f_output_006_W.write(str(W[k][i][j]))
                        f_output_006_W.write(" ")
                    f_output_006_W.write(" ")
            f_output_006_W.write("\n")

            f_output_009.write(str(iiter + 1))
            f_output_009.write(" ")
            f_output_009.write(str(likelihood))
            f_output_009.write(" ")
            f_output_009.write(str(likelihood - likelihood_before))
            f_output_009.write(" ")
            f_output_009.write(str(num_class_current))
            f_output_009.write("\n")

        mode_early_stop = "yes"
        if mode_early_stop == "yes":
            if iiter > 0:
                print("{0:8.4f}".format(likelihood - likelihood_before), criteria_break, end=" ")
                if abs(likelihood - likelihood_before) < criteria_break and mode_E == "VB" and mode_M == "VB":
                    break
                elif abs(likelihood - likelihood_before) < criteria_break and iiter > max_annealing_01 + max_annealing_02:
                    break

        likelihood_before = likelihood

    if mode_write == "ON":
        f_output_002.close()
        f_output_005_mu.close()
        f_output_005_var.close()
        f_output_005_pi.close()
        f_output_006_alpha.close()
        f_output_006_beta.close()
        f_output_006_nu.close()
        f_output_006_m.close()
        f_output_006_W.close()
        f_output_009.close()

    return ex_pi, mk, inv_ex_lambda, r_nk

def main_routine(x):

    for iteration in range(N_iter_start, N_iter_end, 1):
        pi, mu, var, r_nk = estimate(x, iteration)
        print(" ")

        f_output_011 = open(directory_output_main + "/" + directory_output_sub + "/output_vbgmm_011_" + str(iteration).zfill(4) + ".txt", "w")

        for i in range(len(r_nk)):
            f_output_011.write(str(i))
            f_output_011.write(" ")
            f_output_011.write(str(np.argmax(r_nk[i])))
            f_output_011.write("\n")
        f_output_011.close()

        if mode_write == "ON" and len(x[0]) == 2:
            num_class = pi.shape[0]
            plt.plot(x[:, 0], x[:, 1], '+', markersize=5, markeredgewidth=2)
            for k in range(num_class):
                if pi[k] > 10 ** (-3):
                    plt.plot(mu[k, 0], mu[k, 1], 'x',
                             markersize=10, markeredgewidth=3)
            weight = pi * 2
            for k in range(num_class):
                if pi[k] > 10 ** (-3):
                    rbuf = make_ring(var[k], ndiv=50)
                    plt.plot(rbuf[:, 0] + mu[k, 0], rbuf[:, 1] +
                             mu[k, 1], 'r', linewidth=3)
            plt.axis((-20, 20, -20, 20))

            plt.savefig(directory_output_main + "/" + directory_output_sub + '/output_vbgmm_020_' + str(iteration).zfill(4) + '.png')
            plt.clf()


if (__name__ == '__main__'):

    x = faithful_norm()

    time_start = time.time()

    main_routine(x)

    time_end = time.time()

    print(time_end - time_start)

#

