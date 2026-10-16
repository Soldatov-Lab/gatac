// Batched scipy.optimize.curve_fit(relu, x, y) for TOBIAS ATACorrect windows.
// LMDIF / fdjac2 / qrfac / lmpar / qrsolv / enorm are the SciPy 1.17 C translation of
// MINPACK (BSD-3, (C) SciPy developers; MINPACK (C) University of Chicago), specialised to
// n = 2 parameters with the residual f_i = max(0, a*x_i + b) - y_i evaluated in-kernel.
// Compile with --fmad=false so every operation rounds exactly like the x86-64 build.
//
// The licence notices of the translated code follow.

/*
 * Copyright (C) 2024 SciPy developers
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * a. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * b. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * c. Names of the SciPy Developers may not be used to endorse or promote
 *    products derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 *
 * This file is a C translation of the Fortran code written by Jorge J. Moré,
 * Burton S. Garbow, and Kenneth E. Hillstrom, with the original description
 * below. Original Fortran docstrings are included at the top of each function.
 *
 */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Minpack includes software for solving nonlinear equations and nonlinear
 * least squares problems.  Five algorithmic paths each include a core
 * subroutine and an easy-to-use driver.  The algorithms proceed either from
 * an analytic specification of the Jacobian matrix or directly from the
 * problem functions.  The paths include facilities forsystems of equations
 * with a banded Jacobian matrix, for least squares problems with a large
 * amount of data, and for checking the consistency of the Jacobian matrix
 * with the functions.
 *
 *
 * Disclaimer:
 * Minpack Copyright Notice (1999) University of Chicago.  All rights reserved
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the
 * following conditions are met:
 *
 * 1. Redistributions of source code must retain the above
 * copyright notice, this list of conditions and the following
 * disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials
 * provided with the distribution.
 *
 * 3. The end-user documentation included with the
 * redistribution, if any, must include the following
 * acknowledgment:
 *
 *    "This product includes software developed by the
 *    University of Chicago, as Operator of Argonne National
 *    Laboratory.
 *
 * Alternately, this acknowledgment may appear in the software
 * itself, if and wherever such third-party acknowledgments
 * normally appear.
 *
 * 4. WARRANTY DISCLAIMER. THE SOFTWARE IS SUPPLIED "AS IS"
 * WITHOUT WARRANTY OF ANY KIND. THE COPYRIGHT HOLDER, THE
 * UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, AND
 * THEIR EMPLOYEES: (1) DISCLAIM ANY WARRANTIES, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE
 * OR NON-INFRINGEMENT, (2) DO NOT ASSUME ANY LEGAL LIABILITY
 * OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR
 * USEFULNESS OF THE SOFTWARE, (3) DO NOT REPRESENT THAT USE OF
 * THE SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS, (4)
 * DO NOT WARRANT THAT THE SOFTWARE WILL FUNCTION
 * UNINTERRUPTED, THAT IT IS ERROR-FREE OR THAT ANY ERRORS WILL
 * BE CORRECTED.
 *
 * 5. LIMITATION OF LIABILITY. IN NO EVENT WILL THE COPYRIGHT
 * HOLDER, THE UNITED STATES, THE UNITED STATES DEPARTMENT OF
 * ENERGY, OR THEIR EMPLOYEES: BE LIABLE FOR ANY INDIRECT,
 * INCIDENTAL, CONSEQUENTIAL, SPECIAL OR PUNITIVE DAMAGES OF
 * ANY KIND OR NATURE, INCLUDING BUT NOT LIMITED TO LOSS OF
 * PROFITS OR LOSS OF DATA, FOR ANY REASON WHATSOEVER, WHETHER
 * SUCH LIABILITY IS ASSERTED ON THE BASIS OF CONTRACT, TORT
 * (INCLUDING NEGLIGENCE OR STRICT LIABILITY), OR OTHERWISE,
 * EVEN IF ANY OF SAID PARTIES HAS BEEN WARNED OF THE
 * POSSIBILITY OF SUCH LOSS OR DAMAGES.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * The original Fortran code can be found at https://www.netlib.org/minpack/
 *
 * References:
 *
 * [1]: J. J. Moré, B. S. Garbow, and K. E. Hillstrom, User Guide for
 *      MINPACK-1, Argonne National Laboratory Report ANL-80-74, Argonne,
 *      Illinois, 1980.
 *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


#define NP 2
#define MAXM 128

struct Win { const double* x; const double* y; int m; };

__device__ static const double EPSMCH = 2.220446049250313e-16;   // dpmpar[0]
__device__ static const double DWARF = 2.2250738585072014e-308;  // dpmpar[1]

__device__ void relu_fcn(const Win& w, const double* p, double* fvec) {
    for (int i = 0; i < w.m; i++) {
        double v = p[0] * w.x[i];
        v = v + p[1];
        v = (v > 0.0 || v != v) ? v : 0.0;   // np.maximum(0.0, v)
        fvec[i] = v - w.y[i];
    }
}

__device__ double enorm(const int n, const double* x) {
    double agiant, s1, s2, s3, xabs, x1max, x3max;
    const double rgiant = 13043817825332782212.0;   // pow(2.0, 63.5)
    const double rdwarf = 0.5 / rgiant;
    s1 = 0.0; s2 = 0.0; s3 = 0.0; x1max = 0.0; x3max = 0.0;
    agiant = rgiant / n;
    for (int i = 0; i < n; i++) {
        xabs = fabs(x[i]);
        if ((xabs <= rdwarf) || (xabs >= agiant)) {
            if (xabs > rdwarf) {
                if (xabs > x1max) { double t = x1max / xabs; s1 = 1.0 + s1 * (t * t); x1max = xabs; }
                else { double t = xabs / x1max; s1 += t * t; }
            } else {
                if (xabs > x3max) { double t = x3max / xabs; s3 = 1.0 + s3 * (t * t); x3max = xabs; }
                else if (xabs != 0.0) { double t = xabs / x3max; s3 += t * t; }
            }
        } else {
            s2 += xabs * xabs;
        }
    }
    if (s1 != 0.0) return x1max * sqrt(s1 + (s2 / x1max) / x1max);
    if (s2 != 0.0) {
        if (s2 >= x3max) return sqrt(s2 * (1.0 + (x3max / s2) * (x3max * s3)));
        return sqrt(x3max * ((s2 / x3max) + (x3max * s3)));
    }
    return x3max * sqrt(s3);
}

__device__ void fdjac2(const Win& w, const int m, const int n, double* x, const double* fvec,
                       double* fjac, const int ldfjac, const double epsfcn, double* wa) {
    const double eps = sqrt(fmax(epsfcn, EPSMCH));
    for (int j = 0; j < n; j++) {
        double temp = x[j];
        double h = eps * fabs(temp);
        if (h == 0.0) h = eps;
        x[j] = temp + h;
        relu_fcn(w, x, wa);
        x[j] = temp;
        for (int i = 0; i < m; i++) fjac[i + ldfjac * j] = (wa[i] - fvec[i]) / h;
    }
}

__device__ void qrfac(const int m, const int n, double* a, const int lda, int* ipvt,
                      double* rdiag, double* acnorm, double* wa) {
    int i, j, k, kmax, minmn;
    double ajnorm, ssum, temp;
    for (j = 0; j < n; j++) {
        acnorm[j] = enorm(m, &a[j * lda]);
        rdiag[j] = acnorm[j];
        wa[j] = rdiag[j];
        ipvt[j] = j;
    }
    minmn = (m > n ? n : m);
    for (j = 0; j < minmn; j++) {
        kmax = j;
        for (k = j; k < n; k++) if (rdiag[k] > rdiag[kmax]) kmax = k;
        if (kmax != j) {
            for (i = 0; i < m; i++) { temp = a[i + lda * j]; a[i + lda * j] = a[i + lda * kmax]; a[i + lda * kmax] = temp; }
            rdiag[kmax] = rdiag[j];
            wa[kmax] = wa[j];
            k = ipvt[j]; ipvt[j] = ipvt[kmax]; ipvt[kmax] = k;
        }
        ajnorm = enorm(m - j, &a[lda * j + j]);
        if (ajnorm == 0.0) { rdiag[j] = -ajnorm; continue; }
        if (a[lda * j + j] < 0.0) ajnorm = -ajnorm;
        for (i = j; i < m; i++) a[i + lda * j] /= ajnorm;
        a[j + lda * j] += 1.0;
        if (j == n - 1) { rdiag[j] = -ajnorm; continue; }
        for (k = j + 1; k < n; k++) {
            ssum = 0.0;
            for (i = j; i < m; i++) ssum += a[i + lda * j] * a[i + lda * k];
            temp = ssum / a[j + lda * j];
            for (i = j; i < m; i++) a[i + lda * k] -= temp * a[i + lda * j];
            if (rdiag[k] != 0.0) {
                temp = a[j + lda * k] / rdiag[k];
                rdiag[k] *= sqrt(fmax(0.0, 1.0 - temp * temp));
                double t = rdiag[k] / wa[k];
                if (0.05 * (t * t) <= EPSMCH) {
                    rdiag[k] = enorm(m - j - 1, &a[(j + 1) + lda * k]);
                    wa[k] = rdiag[k];
                }
            }
        }
        rdiag[j] = -ajnorm;
    }
}

__device__ void qrsolv(const int n, double* r, const int ldr, const int* ipvt, const double* diag,
                       const double* qtb, double* x, double* sdiag, double* wa) {
    int i, j, k, l, nsing;
    double ccos, ccotan, qtbpj, ssin, ssum, ttan, temp;
    for (j = 0; j < n; j++) {
        for (i = j; i < n; i++) r[i + j * ldr] = r[j + i * ldr];
        x[j] = r[j + j * ldr];
        wa[j] = qtb[j];
    }
    for (j = 0; j < n; j++) {
        l = ipvt[j];
        if (diag[l] != 0.0) {
            for (k = j; k < n; k++) sdiag[k] = 0.0;
            sdiag[j] = diag[l];
            qtbpj = 0.0;
            for (k = j; k < n; k++) {
                if (sdiag[k] != 0.0) {
                    if (fabs(r[ldr * k + k]) < fabs(sdiag[k])) {
                        ccotan = r[ldr * k + k] / sdiag[k];
                        ssin = 0.5 / sqrt(0.25 + 0.25 * (ccotan * ccotan));
                        ccos = ssin * ccotan;
                    } else {
                        ttan = sdiag[k] / r[ldr * k + k];
                        ccos = 0.5 / sqrt(0.25 + 0.25 * (ttan * ttan));
                        ssin = ccos * ttan;
                    }
                    r[ldr * k + k] *= ccos;
                    r[ldr * k + k] += ssin * sdiag[k];
                    temp = ccos * wa[k] + ssin * qtbpj;
                    qtbpj = -ssin * wa[k] + ccos * qtbpj;
                    wa[k] = temp;
                    if (k != n - 1) {
                        for (i = k + 1; i < n; i++) {
                            temp = ccos * r[i + ldr * k] + ssin * sdiag[i];
                            sdiag[i] = -ssin * r[i + ldr * k] + ccos * sdiag[i];
                            r[i + ldr * k] = temp;
                        }
                    }
                }
            }
        }
        sdiag[j] = r[ldr * j + j];
        r[ldr * j + j] = x[j];
    }
    nsing = n;
    for (j = 0; j < n; j++) {
        if ((sdiag[j] == 0.0) && (nsing == n)) nsing = j;
        if (nsing < n) wa[j] = 0.;
    }
    if (nsing > 0) {
        for (k = 1; k <= nsing; k++) {
            j = nsing - k;
            ssum = 0.0;
            if (nsing >= j + 1) for (i = j + 1; i < nsing; ++i) ssum += r[i + j * ldr] * wa[i];
            wa[j] = (wa[j] - ssum) / sdiag[j];
        }
    }
    for (j = 0; j < n; j++) { l = ipvt[j]; x[l] = wa[j]; }
}

__device__ void lmpar(const int n, double* r, const int ldr, const int* ipvt, const double* diag,
                      const double* qtb, const double delta, double* par, double* x, double* sdiag,
                      double* wa1, double* wa2) {
    int i, iter, j, k, l, nsing;
    double dxnorm, fp, gnorm, parc, parl, paru, ssum, temp;
    nsing = n;
    for (j = 0; j < n; j++) {
        wa1[j] = qtb[j];
        if ((r[j + j * ldr] == 0.0) && (nsing == n)) nsing = j;
        if (nsing < n) wa1[j] = 0.0;
    }
    if (nsing > 0) {
        for (k = 1; k <= nsing; k++) {
            j = nsing - k;
            wa1[j] /= r[j + j * ldr];
            temp = wa1[j];
            if (j >= 1) for (i = 0; i < j; i++) wa1[i] -= r[i + j * ldr] * temp;
        }
    }
    for (j = 0; j < n; j++) { l = ipvt[j]; x[l] = wa1[j]; }
    iter = 0;
    for (j = 0; j < n; j++) wa2[j] = diag[j] * x[j];
    dxnorm = enorm(n, wa2);
    fp = dxnorm - delta;
    if (fp <= 0.1 * delta) { if (iter == 0) *par = 0.0; return; }
    parl = 0.0;
    if (nsing == n) {
        for (j = 0; j < n; j++) { l = ipvt[j]; wa1[j] = diag[l] * (wa2[l] / dxnorm); }
        for (j = 0; j < n; j++) {
            ssum = 0.0;
            if (j > 0) for (i = 0; i < j; i++) ssum += r[i + j * ldr] * wa1[i];
            wa1[j] = (wa1[j] - ssum) / r[j + j * ldr];
        }
        temp = enorm(n, wa1);
        parl = ((fp / delta) / temp) / temp;
    }
    for (j = 0; j < n; j++) {
        ssum = 0.0;
        for (i = 0; i <= j; i++) ssum += r[i + j * ldr] * qtb[i];
        l = ipvt[j];
        wa1[j] = ssum / diag[l];
    }
    gnorm = enorm(n, wa1);
    paru = gnorm / delta;
    if (paru == 0.0) paru = DWARF / fmin(delta, 0.1);
    *par = fmin(fmax(*par, parl), paru);
    if (*par == 0.0) *par = gnorm / dxnorm;
    while (1) {
        iter++;
        if (*par == 0.) *par = fmax(DWARF, paru * 0.001);
        temp = sqrt(*par);
        for (j = 0; j < n; j++) wa1[j] = temp * diag[j];
        qrsolv(n, r, ldr, ipvt, wa1, qtb, x, sdiag, wa2);
        for (j = 0; j < n; j++) wa2[j] = diag[j] * x[j];
        dxnorm = enorm(n, wa2);
        temp = fp;
        fp = dxnorm - delta;
        if ((fabs(fp) <= 0.1 * delta) || ((parl == 0.0) && (fp <= temp) && (temp < 0.0)) || (iter == 10)) {
            if (iter == 0) *par = 0.0;
            return;
        }
        for (j = 0; j < n; j++) { l = ipvt[j]; wa1[j] = diag[l] * (wa2[l] / dxnorm); }
        for (j = 0; j < n; j++) {
            wa1[j] /= sdiag[j];
            temp = wa1[j];
            if (j != n - 1) for (i = j + 1; i < n; i++) wa1[i] -= r[i + j * ldr] * temp;
        }
        temp = enorm(n, wa1);
        parc = ((fp / delta) / temp) / temp;
        if (fp > 0.) parl = fmax(parl, *par);
        if (fp < 0.) paru = fmin(paru, *par);
        *par = fmax(parl, *par + parc);
    }
}

// Returns info; x holds the solution, fjac/ipvt the final factorisation.
__device__ int lmdif(const Win& w, double* x, double* fvec, double* fjac, int* ipvt, double* diag,
                     double* qtf, double* wa1, double* wa2, double* wa3, double* wa4, int* nfev_out) {
    const int m = w.m, n = NP, ldfjac = m, maxfev = 200 * (NP + 1), mode = 1;
    const double ftol = 1.49012e-8, xtol = 1.49012e-8, gtol = 0.0, epsfcn = EPSMCH, factor = 100.0;
    int i, iter, j, l, info = 0, nfev = 0;
    double actred, delta = 0.0, dirder, fnorm, fnorm1, gnorm, par, pnorm;
    double prered, ratio, ssum, temp, temp1, temp2, xnorm = 0.0;

    relu_fcn(w, x, fvec);
    nfev = 1;
    fnorm = enorm(m, fvec);
    par = 0.0;
    iter = 1;
    while (1) {
        fdjac2(w, m, n, x, fvec, fjac, ldfjac, epsfcn, wa4);
        nfev += n;
        qrfac(m, n, fjac, ldfjac, ipvt, wa1, wa2, wa3);
        if (iter == 1) {
            if (mode != 2) for (j = 0; j < n; j++) { diag[j] = wa2[j]; if (wa2[j] == 0.0) diag[j] = 1.0; }
            for (j = 0; j < n; j++) wa3[j] = diag[j] * x[j];
            xnorm = enorm(n, wa3);
            delta = factor * xnorm;
            delta = (delta == 0.0 ? factor : delta);
        }
        for (i = 0; i < m; i++) wa4[i] = fvec[i];
        for (j = 0; j < n; j++) {
            if (fjac[j + ldfjac * j] != 0.0) {
                ssum = 0.0;
                for (i = j; i < m; i++) ssum += fjac[i + ldfjac * j] * wa4[i];
                temp = -ssum / fjac[j + ldfjac * j];
                for (i = j; i < m; i++) wa4[i] += fjac[i + ldfjac * j] * temp;
            }
            fjac[j + ldfjac * j] = wa1[j];
            qtf[j] = wa4[j];
        }
        gnorm = 0.0;
        if (fnorm != 0.0) {
            for (j = 0; j < n; j++) {
                l = ipvt[j];
                if (wa2[l] != 0.0) {
                    ssum = 0.0;
                    for (i = 0; i <= j; i++) ssum += fjac[i + ldfjac * j] * (qtf[i] / fnorm);
                    gnorm = fmax(gnorm, fabs(ssum / wa2[l]));
                }
            }
        }
        if (gnorm <= gtol) info = 4;
        if (info != 0) break;
        if (mode != 2) for (j = 0; j < n; j++) diag[j] = fmax(diag[j], wa2[j]);
        do {
            lmpar(n, fjac, ldfjac, ipvt, diag, qtf, delta, &par, wa1, wa2, wa3, wa4);
            for (j = 0; j < n; j++) { wa1[j] = -wa1[j]; wa2[j] = x[j] + wa1[j]; wa3[j] = diag[j] * wa1[j]; }
            pnorm = enorm(n, wa3);
            if (iter == 1) delta = fmin(delta, pnorm);
            relu_fcn(w, wa2, wa4);
            nfev += 1;
            fnorm1 = enorm(m, wa4);
            { double t = fnorm1 / fnorm; actred = (0.1 * fnorm1 < fnorm ? 1.0 - t * t : -1.0); }
            for (j = 0; j < n; j++) {
                wa3[j] = 0.0;
                l = ipvt[j];
                temp = wa1[l];
                for (i = 0; i <= j; i++) wa3[i] += fjac[i + ldfjac * j] * temp;
            }
            temp1 = enorm(n, wa3) / fnorm;
            temp2 = (sqrt(par) * pnorm) / fnorm;
            prered = temp1 * temp1 + temp2 * temp2 / 0.5;
            dirder = -(temp1 * temp1 + temp2 * temp2);
            ratio = (prered != 0.0 ? actred / prered : 0.0);
            if (ratio <= 0.25) {
                temp = (actred < 0.0 ? (0.5 * dirder) / (dirder + 0.5 * actred) : 0.5);
                if ((0.1 * fnorm1 >= fnorm) || (temp < 0.1)) temp = 0.1;
                delta = temp * fmin(delta, pnorm / 0.1);
                par /= temp;
            } else if ((par == 0.0) || (ratio >= 0.75)) {
                delta = pnorm / 0.5;
                par *= 0.5;
            }
            if (ratio >= 0.0001) {
                for (j = 0; j < n; ++j) { x[j] = wa2[j]; wa2[j] = diag[j] * x[j]; }
                for (i = 0; i < m; i++) fvec[i] = wa4[i];
                xnorm = enorm(n, wa2);
                fnorm = fnorm1;
                iter++;
            }
            if ((fabs(actred) <= ftol) && (prered <= ftol) && (0.5 * ratio <= 1.0)) info = 1;
            if (delta <= xtol * xnorm) info = 2;
            if ((fabs(actred) <= ftol) && (prered <= ftol) && (0.5 * ratio <= 1.0) && (info == 2)) info = 3;
            if (info != 0) break;
            if (nfev >= maxfev) info = 5;
            if ((fabs(actred) <= EPSMCH) && (prered <= EPSMCH) && (0.5 * ratio <= 1.0)) info = 6;
            if (delta <= EPSMCH * xnorm) info = 7;
            if (gnorm <= EPSMCH) info = 8;
            if (info != 0) break;
        } while (ratio < 0.0001);
        if (info != 0) break;
    }
    *nfev_out = nfev;
    return info;
}

// One thread per window. Writes the normalised bias prediction of TOBIAS's window loop.
// status: 0 = no signal (zeros), 1 = fit ok, 2 = RuntimeError fallback, 3 = OptimizeWarning fallback
extern "C" __global__ void relu_windows(const double* bias, const double* signal, const long long* wstart,
                                        const int n_win, const int m, double* pred, double* popt, int* status) {
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_win) return;
    const double* xw = bias + wstart[t];
    const double* yw = signal + wstart[t];
    double* out = pred + (long long)t * m;
    Win w; w.x = xw; w.y = yw; w.m = m;

    double smax = yw[0];
    for (int i = 1; i < m; i++) smax = fmax(smax, yw[i]);
    if (!(smax > 0.0)) {
        for (int i = 0; i < m; i++) out[i] = 0.0;
        status[t] = 0;
        return;
    }
    double x[NP] = {1.0, 1.0};
    double fvec[MAXM], wa4[MAXM], fjac[MAXM * NP];
    double diag[NP], qtf[NP], wa1[NP], wa2[NP], wa3[NP];
    int ipvt[NP], nfev;
    int info = lmdif(w, x, fvec, fjac, ipvt, diag, qtf, wa1, wa2, wa3, wa4, &nfev);

    int st = 1;
    if (info < 1 || info > 4) st = 2;
    else {
        // leastsq: r = triu(fjac^T[:n]); trtri(r) fails on an exact zero diagonal
        double r00 = fjac[0], r01 = fjac[m], r11 = fjac[m + 1];
        if (r00 == 0.0 || r11 == 0.0) st = 3;
        else {
            double i00 = 1.0 / r00, i11 = 1.0 / r11;
            double i01 = (i00 * r01) * (-i11);
            // invR[perm] = invR ; cov = invR @ invR.T ; NaN anywhere -> warning
            double R[2][2] = {{i00, i01}, {0.0, i11}}, P[2][2];
            for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) P[ipvt[a]][b] = R[a][b];
            for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) {
                double c = P[a][0] * P[b][0]; c = c + P[a][1] * P[b][1];
                if (c != c) st = 3;
            }
        }
    }
    popt[2 * t] = x[0];
    popt[2 * t + 1] = x[1];
    status[t] = st;
    double mx = -1.0 / 0.0;
    if (st == 1) {
        for (int i = 0; i < m; i++) {
            double v = x[0] * xw[i]; v = v + x[1];
            v = (v > 0.0 || v != v) ? v : 0.0;
            out[i] = v;
            mx = fmax(mx, v);
        }
    } else {
        // cut_positions = ~isclose(signal, 0); bias_predict = clip(bias - min(bias[cut]), 0)
        double bmin = 1.0 / 0.0;
        for (int i = 0; i < m; i++) if (!(fabs(yw[i]) <= 1e-8)) bmin = fmin(bmin, xw[i]);
        for (int i = 0; i < m; i++) {
            double v = xw[i] - bmin;
            if (v < 0) v = 0.0;
            out[i] = v;
            mx = fmax(mx, v);
        }
    }
    if (mx > 0) for (int i = 0; i < m; i++) out[i] = out[i] / mx;
}
