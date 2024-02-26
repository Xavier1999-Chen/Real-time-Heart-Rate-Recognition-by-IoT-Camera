#from obspy.signal.detrend import polynomial #, spline


from sklearn.decomposition import PCA

from scipy import linalg
from face2series import CAM2FACE

import numpy as np

import math
from scipy import sparse
from scipy import signal as frame


from BaseLoader import nn_preprocess
import torch
from neural_methods import trainer

# sns.set()
def ica(X, Nsources, Wprev=0):
    nRows = X.shape[0]
    nCols = X.shape[1]
    if nRows > nCols:
        print(
            "Warning - The number of rows is cannot be greater than the number of columns.")
        print("Please transpose input.")

    if Nsources > min(nRows, nCols):
        Nsources = min(nRows, nCols)
        print(
            'Warning - The number of soures cannot exceed number of observation channels.')
        print('The number of sources will be reduced to the number of observation channels ', Nsources)

    Winv, Zhat = jade(X, Nsources, Wprev)
    W = np.linalg.pinv(Winv)
    return W, Zhat


def jade(X, m, Wprev):
    n = X.shape[0]
    T = X.shape[1]
    nem = m
    seuil = 1 / math.sqrt(T) / 100
    if m < n:
        D, U = np.linalg.eig(np.matmul(X, np.mat(X).H) / T)
        Diag = D
        k = np.argsort(Diag)
        pu = Diag[k]
        ibl = np.sqrt(pu[n - m:n] - np.mean(pu[0:n - m]))
        bl = np.true_divide(np.ones(m, 1), ibl)
        W = np.matmul(np.diag(bl), np.transpose(U[0:n, k[n - m:n]]))
        IW = np.matmul(U[0:n, k[n - m:n]], np.diag(ibl))
    else:
        IW = linalg.sqrtm(np.matmul(X, X.H) / T)
        W = np.linalg.inv(IW)

    Y = np.mat(np.matmul(W, X))
    R = np.matmul(Y, Y.H) / T
    C = np.matmul(Y, Y.T) / T
    Q = np.zeros((m * m * m * m, 1))
    index = 0

    for lx in range(m):
        Y1 = Y[lx, :]
        for kx in range(m):
            Yk1 = np.multiply(Y1, np.conj(Y[kx, :]))
            for jx in range(m):
                Yjk1 = np.multiply(Yk1, np.conj(Y[jx, :]))
                for ix in range(m):
                    Q[index] = np.matmul(Yjk1 / math.sqrt(T), Y[ix, :].T / math.sqrt(
                        T)) - R[ix, jx] * R[lx, kx] - R[ix, kx] * R[lx, jx] - C[ix, lx] * np.conj(C[jx, kx])
                    index += 1
    # Compute and Reshape the significant Eigen
    D, U = np.linalg.eig(Q.reshape(m * m, m * m))
    Diag = abs(D)
    K = np.argsort(Diag)
    la = Diag[K]
    M = np.zeros((m, nem * m), dtype=complex)
    Z = np.zeros(m)
    h = m * m - 1
    for u in range(0, nem * m, m):
        Z = U[:, K[h]].reshape((m, m))
        M[:, u:u + m] = la[h] * Z
        h = h - 1
    # Approximate the Diagonalization of the Eigen Matrices:
    B = np.array([[1, 0, 0], [0, 1, 1], [0, 0 - 1j, 0 + 1j]])
    Bt = np.mat(B).H

    encore = 1
    if Wprev == 0:
        V = np.eye(m).astype(complex)
    else:
        V = np.linalg.inv(Wprev)
    # Main Loop:
    while encore:
        encore = 0
        for p in range(m - 1):
            for q in range(p + 1, m):
                Ip = np.arange(p, nem * m, m)
                Iq = np.arange(q, nem * m, m)
                g = np.mat([M[p, Ip] - M[q, Iq], M[p, Iq], M[q, Ip]])
                temp1 = np.matmul(g, g.H)
                temp2 = np.matmul(B, temp1)
                temp = np.matmul(temp2, Bt)
                D, vcp = np.linalg.eig(np.real(temp))
                K = np.argsort(D)
                la = D[K]
                angles = vcp[:, K[2]]
                if angles[0, 0] < 0:
                    angles = -angles
                c = np.sqrt(0.5 + angles[0, 0] / 2)
                s = 0.5 * (angles[1, 0] - 1j * angles[2, 0]) / c

                if abs(s) > seuil:
                    encore = 1
                    pair = [p, q]
                    G = np.mat([[c, -np.conj(s)], [s, c]])  # Givens Rotation
                    V[:, pair] = np.matmul(V[:, pair], G)
                    M[pair, :] = np.matmul(G.H, M[pair, :])
                    temp1 = c * M[:, Ip] + s * M[:, Iq]
                    temp2 = -np.conj(s) * M[:, Ip] + c * M[:, Iq]
                    temp = np.concatenate((temp1, temp2), axis=1)
                    M[:, Ip] = temp1
                    M[:, Iq] = temp2

    # Whiten the Matrix
    # Estimation of the Mixing Matrix and Signal Separation
    A = np.matmul(IW, V)
    S = np.matmul(np.mat(V).H, Y)
    return A, S
def detrend(input_signal, lambda_value):
        signal_length = input_signal.shape[0]
        # observation matrix
        H = np.identity(signal_length)
        ones = np.ones(signal_length)
        minus_twos = -2 * np.ones(signal_length)
        diags_data = np.array([ones, minus_twos, ones])
        diags_index = np.array([0, 1, 2])
        D = sparse.spdiags(diags_data, diags_index,
                    (signal_length - 2), signal_length).toarray()
        filtered_signal = np.dot(
            (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
        return filtered_signal
def detrend_polynomial(sig, order=2):
        time = np.arange(len(sig))
        coefficients = np.polyfit(time, sig, order)
        detrended_sig = sig - np.polyval(coefficients, time)
        return detrended_sig

class Series2rPPG():
    def __init__(self) -> None:
        # load hist series from CAM
        self.series_class = CAM2FACE()
        self.Ongoing = True

        ##########
        self.TSCAN_model=trainer.TscanTrainer.TscanTrainer()
        self.PhysFormer_model=trainer.PhysFormerTrainer.PhysFormerTrainer()
        ##########

    # Start Processes
    def PROCESS_start(self):
        self.series_class.PROCESS_start()

    def Signal_Preprocessing_single(self, sig):
        return detrend_polynomial(sig, order=2)

    def Signal_Preprocessing(self, rgbsig):
        data = np.array(rgbsig)
        data_r = detrend_polynomial(data[:, 0], order=2)
        data_g = detrend_polynomial(data[:, 1], order=2)
        data_b = detrend_polynomial(data[:, 2], order=2)

        return np.array([data_r, data_g, data_b]).T
    def LGI(self, signal):
        U, _, _ = np.linalg.svd(signal)
        #U.shape = (256, 256)
        S = U[:, 0] #S.shape = ()
        #S = np.expand_dims(S, 1)
        #S = np.array(S)
        St = S.T
        SST = np.matmul(S, St)
        #SST = np.outer(S, S)
        p = np.tile(np.identity(3), (S.shape[0], 1))
        P = p - SST
        Y = np.matmul(P, signal.T).T
        bvp = Y[:, 1]
        bvp = bvp.reshape(-1)
        return bvp
    

    def POS(self, signal):
        WinSec = 1.0
        N = signal.shape[0]
        H = np.zeros((1, N))
        fs = 15
        l = math.ceil(WinSec * fs)

        for n in range(N):
            m = n - l
            if m >= 0:
                Cn = np.true_divide(signal[m:n, :], np.mean(signal[m:n, :], axis=0))
                Cn = np.mat(Cn).H
                S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
                h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
                mean_h = np.mean(h)
                for temp in range(h.shape[1]):
                    h[0, temp] = h[0, temp] - mean_h
                H[0, m:n] = H[0, m:n] + (h[0])

        BVP = H
        BVP = detrend(np.mat(BVP).H, 100)
        BVP = np.asarray(np.transpose(BVP))[0]
        b, a = frame.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
        BVP = frame.filtfilt(b, a, BVP.astype(np.double))
        return BVP
    
    def ICA_POH(self, signal):

        LPF = 0.7
        HPF = 1.5
        RGB = signal
        FS = 15

        NyquistF = 1 / 2 * FS
        BGRNorm = np.zeros(RGB.shape)
        Lambda = 100
        for c in range(3):
            BGRDetrend = detrend(RGB[:, c], Lambda)
            BGRNorm[:, c] = (BGRDetrend - np.mean(BGRDetrend)) / np.std(BGRDetrend)
        _, S = ica(np.mat(BGRNorm).H, 3)

        # select BVP Source
        MaxPx = np.zeros((1, 3))
        for c in range(3):
            FF = np.fft.fft(S[c, :])
            F = np.arange(0, FF.shape[1]) / FF.shape[1] * FS * 60
            FF = FF[:, 1:]
            FF = FF[0]
            N = FF.shape[0]
            Px = np.abs(FF[:math.floor(N / 2)])
            Px = np.multiply(Px, Px)
            Fx = np.arange(0, N / 2) / (N / 2) * NyquistF
            Px = Px / np.sum(Px, axis=0)
            MaxPx[0, c] = np.max(Px)
        MaxComp = np.argmax(MaxPx)
        BVP_I = S[MaxComp, :]
        B, A = frame.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')
        BVP_F = frame.filtfilt(B, A, np.real(BVP_I).astype(np.double))

        BVP = BVP_F[0]
        return BVP

    def PBV(self, signal):
        sig_mean = np.mean(signal, axis=1)

        sig_norm_r = signal[:, 0]/sig_mean[0]
        sig_norm_g = signal[:, 1]/sig_mean[1]
        sig_norm_b = signal[:, 2]/sig_mean[2]

        pbv_n = np.array(
            [np.std(sig_norm_r), np.std(sig_norm_g), np.std(sig_norm_b)])
        pbv_d = np.sqrt(
            np.var(sig_norm_r) + np.var(sig_norm_g) + np.var(sig_norm_b))
        pbv = pbv_n/pbv_d

        C = np.array([sig_norm_r, sig_norm_g, sig_norm_b])
        print(C.shape)
        Ct = C.T
        Q = np.matmul(C, Ct)
        W = np.linalg.solve(Q, pbv)

        A = np.matmul(Ct, W)
        B = np.matmul(pbv.T, W)
        bvp = A/B
        return bvp

    def CHROM(self, signal):
        X = signal
        Xcomp = 3*X[:, 0] - 2*X[:, 1]
        Ycomp = (1.5*X[:, 0])+X[:, 1]-(1.5*X[:, 2])
        sX = np.std(Xcomp)
        sY = np.std(Ycomp)
        alpha = sX/sY
        bvp = Xcomp - alpha * Ycomp
        return bvp

    def PCA(self, signal):
        bvp = []
        for i in range(signal.shape[0]):
            X = signal[i]
            pca = PCA(n_components=3)
            pca.fit(X)
            bvp.append(pca.components_[0] * pca.explained_variance_[0])
            bvp.append(pca.components_[1] * pca.explained_variance_[1])
        bvp = np.array(bvp)
        return bvp

    def GREEN(self, signal):
        return signal[:, 1]

    def GREEN_RED(self, signal):
        return signal[:, 1]-signal[:, 0]
    
    
    def PhysFormer_predict(self, frames):
        process_P=nn_preprocess(frames,128,128,['DiffNormalized'],160)
        #print('process_P',process_P.shape)
        P_input = np.transpose(process_P, (0,4,1,2,3))
        P_input = torch.from_numpy(P_input).float()
        P_output=self.PhysFormer_model.test(P_input)
        P_output=P_output.numpy().flatten()
        return P_output

    def TSCAN_predict(self,frames):
        process_T=nn_preprocess(frames,72,72,['DiffNormalized','Standardized'],180)
        # print('process_T',process_T.shape)
        T_input= np.transpose(process_T,(0,1,4,2,3))
        T_input = torch.from_numpy(T_input).float()
        T_output=self.TSCAN_model.test(T_input)
        T_output=T_output.numpy().flatten()
        return T_output
   
    

    def cal_bpm(self, pre_bpm, spec, fps):
        return pre_bpm*0.95+np.argmax(spec[:int(len(spec)/2)])/len(spec)*fps*60*0.05

    # Deconstruction

    def __del__(self):
        self.Ongoing = False
        self.series_class.__del__()


if __name__ == "__main__":
    processor = Series2rPPG()
    processor.PROCESS_start()
