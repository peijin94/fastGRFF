#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>
#include "grff_constants.h"
#include "grff_tables.cuh"

// Plasma constants from GRFF/source/Plasma.h
static __device__ __forceinline__ double grff_me() { return 9.1093837015e-28; }
static __device__ __forceinline__ double grff_c() { return 2.99792458e10; }
static __device__ __forceinline__ double grff_e() { return (1.602176634e-19 * grff_c() / 10.0); }
static __device__ __forceinline__ double grff_kB() { return 1.380649e-16; }
static __device__ __forceinline__ double grff_hPl() { return 6.62607015e-27; }
static __device__ __forceinline__ double grff_hPlbar() { return (grff_hPl() / 2.0 / CUDART_PI); }
static __device__ __forceinline__ double grff_em_alpha() { return (1.0 / 137.035999084); }
static __device__ __forceinline__ double grff_AU() { return 1.495978707e13; }
static __device__ __forceinline__ double grff_sfu() { return 1e-19; }
static __device__ __forceinline__ double grff_ieH() { return 2.1798718e-11; }
static __device__ __forceinline__ double grff_ieHe12() { return 3.9393356e-11; }
static __device__ __forceinline__ double grff_ieHe2() { return 8.71830663945283224e-11; }

static __device__ __forceinline__ double sqr(double x) { return x * x; }
static __device__ __forceinline__ double dmin(double a, double b) { return (a < b) ? a : b; }
static __device__ __forceinline__ double dmax(double a, double b) { return (a > b) ? a : b; }
static __device__ __forceinline__ int imin(int a, int b) { return (a < b) ? a : b; }
static __device__ __forceinline__ int imax(int a, int b) { return (a > b) ? a : b; }
static __device__ __forceinline__ double dsign(double x) { return (x < 0.0) ? -1.0 : 1.0; }
static __device__ __forceinline__ int dfinit(double x) { return isfinite(x); }

static __device__ __forceinline__ double IntTabulated(const double *x, const double *y, int N)
{
    double s = 0.0;
    for (int i = 1; i < N; i++) {
        s += 0.5 * (y[i - 1] + y[i]) * (x[i] - x[i - 1]);
    }
    return s;
}

static __device__ __forceinline__ double InterpolateBilinear(const double *arr, double i1, double i2, int N1, int N2, double missing)
{
    if (i1 < 0 || i1 > (N1 - 1) || i2 < 0 || i2 > (N2 - 1)) return missing;

    int j = (int)i1;
    int k = (int)i2;
    double t = i1 - j;
    double u = i2 - k;

    double y1 = arr[N2 * j + k];
    double y2 = arr[N2 * (j + 1) + k];
    double y3 = arr[N2 * (j + 1) + k + 1];
    double y4 = arr[N2 * j + k + 1];

    return (1.0 - t) * (1.0 - u) * y1 + t * (1.0 - u) * y2 + t * u * y3 + (1.0 - t) * u * y4;
}

static __device__ __forceinline__ double InterpolBilinear(const double *arr, const double *x1arr, const double *x2arr,
                                                         double x1, double x2, int N1, int N2)
{
    int j, j1, k, k1, l;

    if (x1 < x1arr[0]) {
        j = 0;
        j1 = 1;
    } else if (x1 > x1arr[N1 - 1]) {
        j = N1 - 2;
        j1 = N1 - 1;
    } else {
        j = 0;
        j1 = N1 - 1;
        while ((j1 - j) > 1) {
            l = (j1 + j) >> 1;
            if (x1arr[l] > x1) j1 = l;
            else j = l;
        }
    }
    double dx1 = x1arr[j1] - x1arr[j];
    double t = (x1 - x1arr[j]) / dx1;

    if (x2 < x2arr[0]) {
        k = 0;
        k1 = 1;
    } else if (x2 > x2arr[N2 - 1]) {
        k = N2 - 2;
        k1 = N2 - 1;
    } else {
        k = 0;
        k1 = N2 - 1;
        while ((k1 - k) > 1) {
            l = (k1 + k) >> 1;
            if (x2arr[l] > x2) k1 = l;
            else k = l;
        }
    }
    double dx2 = x2arr[k1] - x2arr[k];
    double u = (x2 - x2arr[k]) / dx2;

    double y1 = arr[N2 * j + k];
    double y2 = arr[N2 * j1 + k];
    double y3 = arr[N2 * j1 + k1];
    double y4 = arr[N2 * j + k1];

    return (1.0 - t) * (1.0 - u) * y1 + t * (1.0 - u) * y2 + t * u * y3 + (1.0 - t) * u * y4;
}

static __device__ __forceinline__ double LogFactorial(int n)
{
    return lgamma((double)n + 1.0);
}

static __device__ __forceinline__ double lnC1(double T, double f)
{
    double log_u = log10(k_u * f / T);
    double log_g2 = log10(k_g2 / T);

    double idx_u = (log_u - s_u) / d;
    double idx_g2 = (log_g2 - s_g2) / d;

    double G = InterpolateBilinear((const double *)g_arr, idx_u, idx_g2, N_u, N_g2, CUDART_NAN);
    return G * CUDART_PI / sqrt(3.0);
}

static __device__ __forceinline__ double Zeta_Solar(double T, double f, int ABcode)
{
    const double *z;
    switch (ABcode) {
        case 0: z = (const double *)ZetaSolarCoronal_arr; break;
        case 1: z = (const double *)ZetaSolarPhotCaffau_arr; break;
        case 2: z = (const double *)ZetaSolarPhotScott_arr; break;
        default: z = (const double *)ZetaSolarCoronal_arr; break;
    }

    return InterpolBilinear(z, (const double *)lnf_arr, (const double *)lnT_arr, log(f), log(T), N_f, N_T);
}

static __device__ __forceinline__ double Zeta_arbitrary(double T, double f, int ABcode, int N_fZ, int N_TZ,
                                                       const double *lnfZ_arr, const double *lnTZ_arr, const double *Z_arr)
{
    const double *z = Z_arr + (size_t)ABcode * N_fZ * N_TZ;
    return InterpolBilinear(z, lnTZ_arr, lnfZ_arr, log(T), log(f), N_TZ, N_fZ);
}

static __device__ __forceinline__ void FindPlasmaDispersion(double f, double f_p, double f_B, double theta, int sigma,
                                                           double *N, double *FZh, double *L, double *T, double *st_out, double *ct_out)
{
    const double cst_min = 1e-5;
    double f_c = (sigma == -1) ? f_B / 2.0 + sqrt(sqr(f_p) + sqr(f_B) / 4.0) : f_p;

    if (f <= f_c) {
        *N = CUDART_NAN;
    } else {
        double ct = cos(theta);
        double st = sin(theta);
        if (fabs(ct) < cst_min) {
            ct = cst_min * dsign(ct);
            st = sqrt(1.0 - sqr(ct)) * dsign(st);
        }
        if (fabs(st) < cst_min) {
            st = cst_min * dsign(st);
            ct = sqrt(1.0 - sqr(st)) * dsign(ct);
        }

        double u = sqr(f_B / f);
        double v = sqr(f_p / f);

        double Delta = sqrt(sqr(u * sqr(st)) + 4.0 * u * sqr((1.0 - v) * ct));
        *N = sqrt(1.0 - 2.0 * v * (1.0 - v) / (2.0 * (1.0 - v) - u * sqr(st) + (double)sigma * Delta));

        if (FZh) {
            *FZh = u ? 2.0 * (u * sqr(st) + 2.0 * sqr(1.0 - v) - (double)sigma * sqr(u * sqr(st)) / Delta)
                        / sqr(2.0 * (1.0 - v) - u * sqr(st) + (double)sigma * Delta)
                     : 1.0;
        }

        if (L != 0 || T != 0) {
            double Tloc = 2.0 * sqrt(u) * (1.0 - v) * ct / (u * sqr(st) - (double)sigma * Delta);

            if (T) *T = Tloc;
            if (L) *L = (v * sqrt(u) * st + Tloc * u * v * st * ct) / (1.0 - u - v + u * v * sqr(ct));
        }

        if (st_out) *st_out = st;
        if (ct_out) *ct_out = ct;
    }
}

static __device__ __forceinline__ double SahaH(double n0, double T0)
{
    double x = 0.0;
    if (T0 > 0.0 && n0 > 0.0) {
        double xi = pow(2.0 * CUDART_PI * grff_me() * grff_kB() * T0 / sqr(grff_hPl()), 1.5) / n0 * exp(-grff_ieH() / grff_kB() / T0);
        x = xi ? 2.0 / (sqrt(1.0 + 4.0 / xi) + 1.0) : 0.0;
    }
    return x;
}

static __device__ __forceinline__ void SahaHe(double n_p, double T0, double *a12, double *a2)
{
    *a12 = 0;
    *a2 = 0;

    if (T0 > 0.0 && n_p > 0.0) {
        double A = 4.0 * pow(2.0 * CUDART_PI * grff_me() * grff_kB() * T0 / sqr(grff_hPl()), 1.5) / n_p;

        double xi12 = A * exp(-grff_ieHe12() / grff_kB() / T0);
        *a12 = xi12 / (1.0 + xi12);

        double xi2 = A * exp(-grff_ieHe2() / grff_kB() / T0);
        *a2 = xi2 / (1.0 + xi2);
    }
}

static __device__ __forceinline__ void FindIonizationsSolar(double n0, double T0, double *n_e, double *n_H, double *n_He)
{
    double n_Htotal = n0 * 0.922;
    double n_Hetotal = n0 * 0.078;

    double a = SahaH(n_Htotal, T0);
    double n_p = n_Htotal * a;
    *n_H = n_Htotal * (1.0 - a);

    double a12, a2;
    SahaHe(n_p, T0, &a12, &a2);
    *n_He = n_Hetotal * (1.0 - a12);
    *n_e = n_p + n_Hetotal * (a12 + a2) + n_Htotal * 1e-3;
}

static __device__ __forceinline__ void DEM_moments(const double *T_arr, const double *lnT_arr, const double *DEM_arr, int N,
                                                   double *n_avg, double *T_avg, double *tmp1, double *tmp2)
{
    for (int i = 0; i < N; i++) {
        tmp1[i] = DEM_arr[i] * T_arr[i];
        tmp2[i] = tmp1[i] * T_arr[i];
    }

    double n2_avg = IntTabulated(lnT_arr, tmp1, N);
    *n_avg = sqrt(n2_avg);
    *T_avg = (n2_avg > 0) ? IntTabulated(lnT_arr, tmp2, N) / n2_avg : 0.0;
}

static __device__ __forceinline__ void DDM_moments(const double *T_arr, const double *lnT_arr, const double *DDM_arr, int N,
                                                   double *n_avg, double *T_avg, double *tmp1, double *tmp2)
{
    for (int i = 0; i < N; i++) {
        tmp1[i] = DDM_arr[i] * T_arr[i];
        tmp2[i] = tmp1[i] * T_arr[i];
    }

    *n_avg = IntTabulated(lnT_arr, tmp1, N);
    *T_avg = (*n_avg > 0) ? IntTabulated(lnT_arr, tmp2, N) / (*n_avg) : 0.0;
}

static __device__ __forceinline__ void FindFF_single(double f, double theta, int sigma, double f_p, double f_B, double T0, double n_e, int ABcode,
                                                    int AZ_on, int NfZ, int NTZ, const double *lnfZ_arr, const double *lnTZ_arr, const double *Z_arr,
                                                    double *j, double *k)
{
    double N, FZh;

    FindPlasmaDispersion(f, f_p, f_B, theta, sigma, &N, &FZh, 0, 0, 0, 0);
    if (dfinit(N)) {
        if (n_e > 0) {
            double lnC = lnC1(T0, f);
            double zeta = (AZ_on) ? Zeta_arbitrary(T0, f, ABcode, NfZ, NTZ, lnfZ_arr, lnTZ_arr, Z_arr) : Zeta_Solar(T0, f, ABcode);

            double jff = 8 * pow(grff_e(), 6) * N / (3.0 * sqrt(2.0 * CUDART_PI) * sqrt(pow(grff_me(), 3)) * pow(grff_c(), 3)) *
                         sqr(n_e) * lnC / sqrt(grff_kB() * T0) * (1.0 + zeta);
            double kff = 8 * pow(grff_e(), 6) / (3.0 * sqrt(2.0 * CUDART_PI) * N * grff_c() * sqr(f) * sqrt(pow(grff_me(), 3))) *
                         sqr(n_e) * lnC / (sqrt(grff_kB() * T0) * grff_kB() * T0) * (1.0 + zeta);

            *j = jff * FZh;
            *k = kff * FZh;
        } else {
            *j = 0.0;
            *k = 0.0;
        }
    } else {
        *j = 0.0;
        *k = 1e100;
    }
}

static __device__ __forceinline__ void FindFF_DEM_XO(double f, double theta, double f_p, double f_B,
                                                    const double *T_arr, const double *lnT_arr, const double *DEM_arr, int NT, int ABcode,
                                                    int AZ_on, int NfZ, int NTZ, const double *lnfZ_arr, const double *lnTZ_arr, const double *Z_arr,
                                                    double *jX, double *kX, double *jO, double *kO,
                                                    double *tmp_j, double *tmp_k)
{
    double NX, FZhX, NO, FZhO;

    FindPlasmaDispersion(f, f_p, f_B, theta, -1, &NX, &FZhX, 0, 0, 0, 0);
    FindPlasmaDispersion(f, f_p, f_B, theta,  1, &NO, &FZhO, 0, 0, 0, 0);

    if (dfinit(NX) || dfinit(NO)) {
        for (int i = 0; i < NT; i++) {
            if (DEM_arr[i] > 0) {
                double zeta = (AZ_on) ? Zeta_arbitrary(T_arr[i], f, ABcode, NfZ, NTZ, lnfZ_arr, lnTZ_arr, Z_arr) :
                                        Zeta_Solar(T_arr[i], f, ABcode);
                tmp_j[i] = DEM_arr[i] * lnC1(T_arr[i], f) / sqrt(grff_kB() * T_arr[i]) * (1.0 + zeta) * T_arr[i];
            } else {
                tmp_j[i] = 0.0;
            }
            tmp_k[i] = tmp_j[i] / (grff_kB() * T_arr[i]);
        }

        double aj = IntTabulated(lnT_arr, tmp_j, NT);
        double ak = IntTabulated(lnT_arr, tmp_k, NT);

        if (dfinit(NX)) {
            *jX = 8 * pow(grff_e(), 6) * NX / (3.0 * sqrt(2.0 * CUDART_PI) * sqrt(pow(grff_me(), 3)) * pow(grff_c(), 3)) * aj * FZhX;
            *kX = 8 * pow(grff_e(), 6) / (3.0 * sqrt(2.0 * CUDART_PI) * NX * grff_c() * sqr(f) * sqrt(pow(grff_me(), 3))) * ak * FZhX;
        } else {
            *jX = 0.0;
            *kX = 1e100;
        }

        if (dfinit(NO)) {
            *jO = 8 * pow(grff_e(), 6) * NO / (3.0 * sqrt(2.0 * CUDART_PI) * sqrt(pow(grff_me(), 3)) * pow(grff_c(), 3)) * aj * FZhO;
            *kO = 8 * pow(grff_e(), 6) / (3.0 * sqrt(2.0 * CUDART_PI) * NO * grff_c() * sqr(f) * sqrt(pow(grff_me(), 3))) * ak * FZhO;
        } else {
            *jO = 0.0;
            *kO = 1e100;
        }
    }
}

static __device__ __forceinline__ void FindGR_single(double f, double theta, int sigma, int s, double f_p, double f_B, double n_e, double T0, double LB,
                                                    double *tau, double *I0)
{
    double N, L, T, st, ct;

    FindPlasmaDispersion(f, f_p, f_B, theta, sigma, &N, 0, &L, &T, &st, &ct);
    if (dfinit(N)) {
        if (f_p <= 0 || T0 <= 0) {
            *tau = 0.0;
            *I0 = 0.0;
        } else {
            double lnQ = log(grff_kB() * T0 / grff_me() / grff_c() / grff_c() * sqr(s * N * st) / 2.0) * (s - 1);
            *tau = exp(lnQ - LogFactorial(s)) * CUDART_PI * grff_e() * grff_e() * n_e / (f * grff_me() * grff_c()) * s * s / N * LB *
                   sqr(T * ct + L * st + 1.0) / (1.0 + sqr(T));
            *I0 = grff_kB() * T0 * sqr(f * N / grff_c());
        }
    } else {
        *tau = 1e100;
        *I0 = 0.0;
    }
}

static __device__ __forceinline__ void FindGR_DDM_XO(double f, double theta, int s, double f_p, double f_B,
                                                    const double *T_arr, const double *lnT_arr, const double *DDM_arr, int NT, double LB,
                                                    double *tauX, double *I0X, double *tauO, double *I0O,
                                                    double *tmp_tau, double *tmp_J)
{
    double NX, LX, TX, stX, ctX, NO, LO, TO, stO, ctO;

    FindPlasmaDispersion(f, f_p, f_B, theta, -1, &NX, 0, &LX, &TX, &stX, &ctX);
    FindPlasmaDispersion(f, f_p, f_B, theta,  1, &NO, 0, &LO, &TO, &stO, &ctO);

    if (dfinit(NX) || dfinit(NO)) {
        for (int i = 0; i < NT; i++) {
            tmp_tau[i] = (DDM_arr[i] > 0) ?
                         DDM_arr[i] * exp(log(grff_kB() * T_arr[i] / grff_me() / grff_c() / grff_c()) * (s - 1)) * T_arr[i] : 0.0;
            tmp_J[i] = tmp_tau[i] * (grff_kB() * T_arr[i] / grff_me() / grff_c() / grff_c());
        }

        double I_tau = IntTabulated(lnT_arr, tmp_tau, NT);
        double I_J = IntTabulated(lnT_arr, tmp_J, NT);

        if (dfinit(NX)) {
            if (f_p <= 0) {
                *tauX = 0.0;
                *I0X = 0.0;
            } else {
                double lnQ = log(sqr(s * NX * stX) / 2.0) * (s - 1);
                *tauX = exp(lnQ - LogFactorial(s)) * I_tau * CUDART_PI * grff_e() * grff_e() / (f * grff_me() * grff_c()) * s * s / NX * LB *
                        sqr(TX * ctX + LX * stX + 1.0) / (1.0 + sqr(TX));
                *I0X = grff_me() * I_J / I_tau * sqr(f * NX);
            }
        } else {
            *tauX = 1e100;
            *I0X = 0.0;
        }

        if (dfinit(NO)) {
            if (f_p <= 0) {
                *tauO = 0.0;
                *I0O = 0.0;
            } else {
                double lnQ = log(sqr(s * NO * stO) / 2.0) * (s - 1);
                *tauO = exp(lnQ - LogFactorial(s)) * I_tau * CUDART_PI * grff_e() * grff_e() / (f * grff_me() * grff_c()) * s * s / NO * LB *
                        sqr(TO * ctO + LO * stO + 1.0) / (1.0 + sqr(TO));
                *I0O = grff_me() * I_J / I_tau * sqr(f * NO);
            }
        } else {
            *tauO = 1e100;
            *I0O = 0.0;
        }
    }
}

static __device__ __forceinline__ void FindNeutralsEffect(double f, double theta, int sigma, double f_p, double f_B, double T0, double n_e, double n_H, double n_He,
                                                         double *j, double *k)
{
    double N, FZh;

    FindPlasmaDispersion(f, f_p, f_B, theta, sigma, &N, &FZh, 0, 0, 0, 0);
    if (dfinit(N)) {
        double jH = 0.0, kH = 0.0;

        if (n_e > 0 && n_H > 0 && T0 > 2500 && T0 < 50000) {
            double kT = sqrt(grff_kB() * T0 / grff_ieH());
            double xi = 4.862 * kT * (1.0 - 0.2096 * kT + 0.0170 * kT * kT - 0.00968 * kT * kT * kT);
            kH = 1.2737207e-11 * n_e * n_H * sqrt(T0) / (sqr(f) * N) * exp(-xi);

            jH = kH * sqr(N * f / grff_c()) * grff_kB() * T0;
        }

        double jHe = 0.0, kHe = 0.0;
        if (n_e > 0 && n_He > 0 && T0 > 2500 && T0 < 25000) {
            double kT = sqrt(grff_kB() * T0 / grff_ieH());
            kHe = 5.9375453e-13 * n_e * n_He * sqrt(T0) / (sqr(f) * N) * (1.868 + 7.415 * kT - 22.56 * kT * kT + 15.59 * kT * kT * kT);

            jHe = kHe * sqr(N * f / grff_c()) * grff_kB() * T0;
        }

        *j = (jH + jHe) * FZh;
        *k = (kH + kHe) * FZh;
    } else {
        *j = 0.0;
        *k = 1e100;
    }
}
