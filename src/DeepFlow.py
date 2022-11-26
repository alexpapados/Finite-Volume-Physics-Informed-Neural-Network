import numpy as np

class autodiff:
    def minmod(self, v):
        s = np.sum(np.sign(v)) / np.size(v)
        if abs(s) == 1:
            mm = s * np.min(np.abs(v[:]))
        else:
            mm = 0
        return mm
    def vanalbada(self, da,db,h):
        eps2=(0.3*h)**3
        va=0.5*(np.sign(da)*np.sign(db)+1)*((db**2+eps2)*da+(da**2+eps2)*db)/(da**2+db**2+2*eps2)
        return va

    #*********** 1-D Compressible Euler *****************#
    def HLLE_FLUX_1D(self, qL, qR, gamma):

        rL = qL[0]
        uL = qL[1] / rL
        EL = qL[2] / rL
        pL = (gamma - 1) * (qL[2] - rL * uL * uL / 2)

        aL = np.sqrt(np.abs(gamma * pL / rL))
        HL = (qL[2] + pL) / rL

        rR = qR[0]
        uR = qR[1] / rR
        ER = qR[2] / rR
        pR = (gamma - 1) * (qR[2] - rR * uR * uR / 2)

        aR = np.sqrt(np.abs(gamma * pR / rR))
        HR = (qR[2] + pR) / rR

        RT = np.sqrt(np.abs(rR / rL))
        u = (uL + RT * uR) / (1 + RT)
        H = (HL + RT * HR) / (1 + RT)
        a = np.sqrt(np.abs((gamma - 1) * (H - u * u / 2)))

        SLm = np.min([uL - aL, u - a])
        SRp = np.max([uR + aR, u + a])

        FL = np.zeros((3, 1))
        FR = np.zeros((3, 1))
        FL[0], FL[1], FL[2] = rL * uL, rL * uL ** 2 + pL, uL * (rL * EL + pL)
        FR[0], FR[1], FR[2] = rR * uR, rR * uR ** 2 + pR, uR * (rR * ER + pR)

        if SLm >= 0:
            HLLE = FL

        elif (SLm <= 0) and (SRp >= 0):
            HLLE = (SRp * FL - SLm * FR + SLm * SRp * (qR - qL)) / (SRp - SLm)

        elif SRp <= 0:
            HLLE = FR

        return HLLE

    def HLLC_FLUX_1D(self, qL, qR, gamma):

        rL = qL[0]
        uL = qL[1] / rL
        EL = qL[2] / rL
        pL = (gamma - 1) * (qL[2] - rL * uL * uL / 2)
        aL = np.sqrt(gamma * pL / rL)

        rR = qR[0]
        uR = qR[1] / rR
        ER = qR[2] / rR
        pR = (gamma - 1) * (qR[2] - rR * uR * uR / 2)
        aR = np.sqrt(gamma * pR / rR)

        FL = np.zeros((3, 1))
        FR = np.zeros((3, 1))
        FL[0], FL[1], FL[2] = rL * uL, rL * uL ** 2 + pL, uL * (rL * EL + pL)
        FR[0], FR[1], FR[2] = rR * uR, rR * uR ** 2 + pR, uR * (rR * ER + pR)

        PPV = np.max([0, 0.5 * (pL + pR) + 0.5 * (uL - uR) * (0.25 * (rL + rR) * (aL + aR))])
        pmin = np.min([pL, pR])
        pmax = np.max([pL, pR])
        Qmax = pmax / pmin
        Quser = 2.0

        if (Qmax <= Quser) and (pmin <= PPV) and (PPV <= pmax):
            pM = PPV
        else:
            if PPV < pmin:
                PQ = (pL / pR)**(gamma - 1.0) / (2.0 * gamma)
                uM = (PQ * uL / aL + uR / aR + 2 / (gamma - 1) * (PQ - 1.0)) / (PQ / aL + 1.0 / aR)
                PTL = 1 + (gamma - 1) / 2.0 * (uL - uM) / aL
                PTR = 1 + (gamma - 1) / 2.0 * (uM - uR) / aR
                pM = 0.5 * (pL * PTL**(2 * gamma / (gamma - 1)) + pR * PTR ** (2 * gamma / (gamma - 1)))
            else:
                GEL = np.sqrt((2 / (gamma + 1) / rL) / ((gamma - 1) / (gamma + 1) * pL + PPV))
                GER = np.sqrt((2 / (gamma + 1) / rR) / ((gamma - 1) / (gamma + 1) * pR + PPV))
                pM = (GEL * pL + GER * pR - (uR - uL)) / (GEL + GER)
        if pM > pL:
            zL = np.sqrt(1+(gamma+1) / (2 * gamma) * (pM / pL-1))
        else:
            zL = 1
        if pM > pR:
            zR = np.sqrt(1 + (gamma + 1) / (2 * gamma) * (pM / pR - 1))
        else:
           zR = 1

        SL = uL - aL * zL
        SR = uR + aR * zR
        SM = (pL - pR + rR * uR * (SR - uR) - rL * uL * (SL - uL)) / (rR * (SR - uR) - rL * (SL - uL))

        if 0 <= SL:
            HLLC = FL
        elif ((SL <= 0) and (0 <= SM)):
            hll_tmp_L = np.zeros((3,1))
            hll_tmp_L[0] = 1
            hll_tmp_L[1] = SM
            hll_tmp_L[2] = qL[2] / rL + (SM - uL) * (SM + pL / (rL * (SL - uL)))
            qsL = rL * (SL - uL) / (SL - SM) * hll_tmp_L
            HLLC = FL + SL * (qsL - qL)
        elif ((SM <= 0) and (0 <= SR)):
            hll_tmp_R = np.zeros((3, 1))
            hll_tmp_R[0] = 1
            hll_tmp_R[1] = SM
            hll_tmp_R[2] = qR[2]/rR + (SM-uR)*(SM+pR/(rR*(SR-uR)))
            qsR = rR * (SR - uR) / (SR - SM) * hll_tmp_R
            HLLC = FR + SR * (qsR - qR)
        elif 0 >= SR:
            HLLC = FR
        return HLLC

    def MUSCL_EULER_1D(self, q, gamma, dx, nx, flux_method):
        # Compute and limit slopes
        res = np.zeros((3, nx))
        dq = np.zeros((3, nx))
        flux = np.zeros((3, nx - 1))
        q_l = np.zeros((3, nx - 1))
        q_r = np.zeros((3, nx - 1))
        for i in range(0, 3):
            for j in range(1, nx - 1):
                dqR = (q[i, j + 1] - q[i, j])
                dqL = (q[i, j] - q[i, j - 1])
                dq[i, j] = self.minmod([dqR, dqL])

        for j in range(1, nx - 2):
            q_l[:, [j]] = q[:, [j]] + dq[:, [j]] / 2
            q_r[:, [j]] = q[:, [j + 1]] - dq[:, [j + 1]] / 2
        for j in range(1, nx - 2):
            if flux_method == 'HLLE':
                flux[:, [j]] = self.HLLE_FLUX_1D(q_l[:, [j]], q_r[:, [j]], gamma)
            if flux_method == 'HLLC':
                flux[:, [j]] = self.HLLC_FLUX_1D(q_l[:, [j]], q_r[:, [j]], gamma)

            res[:, [j]] = res[:, [j]] + flux[:, [j]] / dx
            res[:, [j + 1]] = res[:, [j + 1]] - flux[:, [j]] / dx

        q_r[:, [0]] = q[:, [1]] - dq[:, [1]] * dx / 2
        q_l[:, [0]] = q_r[:, [0]]
        if flux_method == 'HLLE':
            flux[:, [0]] = self.HLLE_FLUX_1D(q_l[:, [0]], q_r[:, [0]], gamma)
        if flux_method == 'HLLC':
            flux[:, [0]] = self.HLLC_FLUX_1D(q_l[:, [0]], q_r[:, [0]], gamma)
        res[:, 1] = res[:, 1] - flux[:, 0] / dx

        q_l[:, [nx - 2]] = q[:, [nx - 2]] + dq[:, [nx - 2]] * (dx / 2)
        q_r[:, [nx - 2]] = q_l[:, [nx - 2]]
        if flux_method == 'HLLE':
            flux[:, [nx - 2]] = self.HLLE_FLUX_1D(q_l[:, [nx - 2]], q_r[:, [nx - 2]], gamma)
        if flux_method == 'HLLC':
            flux[:, [nx - 2]] = self.HLLC_FLUX_1D(q_l[:, [nx - 2]], q_r[:, [nx - 2]], gamma)

        res[:, [nx - 2]] = res[:, [nx - 2]] + flux[:, [nx - 2]] / dx
        return res

    # *********** 2-D Compressible Euler *****************#

    def HLLE_FLUX_2D(self, qL,qR, gamma, normal):
        nx = normal[0]
        ny = normal[1]

        rL = qL[0]
        uL = qL[1]/rL
        vL = qL[2]/rL
        vnL = uL * nx + vL * ny
        pL = (gamma - 1) * (qL[3] - rL * (uL**2 + vL**2) / 2)
        aL = np.sqrt(np.abs(gamma * pL / rL))
        HL = (qL[3] + pL) / rL

        rR = qR[0]
        uR = qR[1] / rR
        vR = qR[2] / rR
        vnR = uR * nx + vR * ny
        pR = (gamma - 1) * (qR[3] - rR * (uR**2 + vR**2) / 2)
        aR = np.sqrt(np.abs(gamma * pR / rR))
        HR = (qR[3] + pR) / rR

        RT = np.sqrt(rR / rL)
        u = (uL + RT * uR) / (1 + RT)
        v = (vL + RT * vR) / (1 + RT)
        H = (HL + RT * HR) / (1 + RT)
        a = np.sqrt(np.abs((gamma - 1) * (H - (u ** 2 + v ** 2) / 2)))
        vn = u * nx + v * ny

        SLm = np.min([vnL - aL, vn - a, 0])
        SRp = np.max([vnR + aR, vn + a, 0])
        FL = np.zeros((4,1))
        FR = np.zeros((4, 1))

        FL[0] = rL * vnL
        FL[1] = rL * vnL * uL + pL * nx
        FL[2] = rL * vnL * vL + pL * ny
        FL[3] = rL * vnL * HL

        FR[0] = rR * vnR
        FR[1] = rR * vnR * uR + pR * nx
        FR[2] = rR * vnR * vR + pR * ny
        FR[3] = rR * vnR * HR

        HLLE = (SRp * FL - SLm * FR + SLm * SRp * (qR - qL)) / (SRp - SLm)


        return HLLE

    def MUSCL_Euler_2D(self, q, smax, gamma, dx, dy, N, M, limiter, fluxMethod):
        res = np.zeros((M, N, 4))
        dqdy = np.zeros((M, N, 4))
        dqdx = np.zeros((M, N, 4))

        for i in range(1,M-1):
            for j in range(1,N-1):
                for k in range(0,4):
                    dqw = 2 * (q[i,j,k] - q[i,j - 1,k]) / dx;
                    dqe = 2 * (q[i,j + 1,k] - q[i,j,k]) / dx;
                    dqc = (q[i,j + 1,k] - q[i,j - 1,k]) / (2 * dx);
                    dqdx[i][j][k] = self.minmod([dqw, dqe, dqc]);

                    dqs = 2 * (q[i,j,k] - q[i - 1,j,k]) / dy;
                    dqn = 2 * (q[i + 1,j,k] - q[i,j,k]) / dy;
                    dqc = (q[i + 1,j,k] - q[i - 1,j,k]) / (2 * dy);
                    dqdy[i][j][k] = self.minmod([dqs, dqn, dqc]);

        for i in range(1,M-1):
           for j in range(1,N-2):
               qxL = q[i,j, :] + dqdx[i,j,:]*dx/2
               qxL = np.reshape(qxL,(4,1))

               qxR = q[i,j+1, :] - dqdx[i, j+1, :] * dx / 2
               qxR = np.reshape(qxR, (4, 1))

               flux = self.HLLE_FLUX_2D(qxL, qxR, gamma, [1, 0])
               flux = np.reshape(flux,(1,4))

               res[i,j,:] = res[i,j,:] + flux/dx
               res[i, j+1, :] = res[i, j+1, :] -  flux/dx

        for i in range(1, M - 2):
            for j in range(1, N - 1):
                qyL = q[i, j, :] + dqdy[i, j, :] * dy / 2
                qyL = np.reshape(qyL, (4, 1))

                qyR = q[i+1, j, :] - dqdy[i+1, j, :] * dy / 2
                qyR = np.reshape(qyR, (4, 1))

                flux = self.HLLE_FLUX_2D(qyL, qyR, gamma, [0, 1])
                flux = np.reshape(flux, (1, 4))

                res[i, j, :] = res[i, j, :] + flux / dy
                res[i+1, j, :] = res[i+1, j, :] - flux / dy

        for j in range(1,N-1):
            qR = q[M-2,j,:] + dqdy[M-2,j,:]*dy/2
            qL = qR
            qL = np.reshape(qL, (4, 1))
            qR = np.reshape(qR, (4, 1))

            flux = self.HLLE_FLUX_2D(qL, qR, gamma, [0, 1])
            flux = np.reshape(flux, (1, 4))

            res[M-2,j,:] = res[M-2,j,:] + flux/dy

        for i in range(1, M - 1):
            qR = q[i, N-1, :] + dqdx[i, N-1, :] * dx / 2
            qL = qR
            qL = np.reshape(qL, (4, 1))
            qR = np.reshape(qR, (4, 1))

            flux = self.HLLE_FLUX_2D(qL, qR, gamma, [1, 0])
            flux = np.reshape(flux, (1, 4))

            res[i, N-2, :] = res[i, N-2, :] + flux / dx

        for j in range(1,N-1):
            qR = q[1,j,:] - dqdy[1,j,:]*dy/2
            qL = qR
            qL = np.reshape(qL, (4, 1))
            qR = np.reshape(qR, (4, 1))

            flux = self.HLLE_FLUX_2D(qL, qR, gamma, [0, -1])
            flux = np.reshape(flux, (1, 4))

            res[1,j,:] = res[1,j,:] + flux/dy

        for i in range(1, M - 1):
            qR = q[i, 1, :] - dqdx[i, 1, :] * dx / 2
            qL = qR
            qL = np.reshape(qL, (4, 1))
            qR = np.reshape(qR, (4, 1))

            flux = self.HLLE_FLUX_2D(qL, qR, gamma, [-1, 0])
            flux = np.reshape(flux, (1, 4))

            res[i, 1, :] = res[i, 1, :] + flux / dx

        return res

