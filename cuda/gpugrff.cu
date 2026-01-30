#include <cuda_runtime.h>
#include <math_constants.h>

#include "grff_constants.h"
#include "grff_device.cuh"

struct Voxel
{
    double B, theta, psi, Bx, By, Bz;
    double B1, B2, Bz1, Bz2;
    double B_a[2], B_b[2], Bx_a[2], Bx_b[2], By_a[2], By_b[2], Bz_a[2], Bz_b[2];
    double dB_dz[2], dtheta_dz[2];
    double n_e, T0;
    double n_H, n_He;
    double zstart, zstart1, dz;
    double f_p;
    int DEM_on, DDM_on, FF_on, GR_on, HHe_on, force_isothermal, s_max, s_min, j_ofs, ABcode, dfcode;
    double S, refTerm;
};

struct Level
{
    int s;
    double zstart, zstart1;
    double z0;
};

static __device__ double ProcessVoxels(int Nz0, const double *Parms, int NT, const double *T_arr, const double *lnT_arr,
                                      const double *DEM_arr, const double *DDM_arr,
                                      int smin_global, int smax_global, Voxel *V, double *tmp1, double *tmp2)
{
    for (int j = 0; j < Nz0; j++) {
        const double *p = Parms + j * InSize;

        V[j].dz = dmax(p[0], 0.0);
        V[j].T0 = dmax(p[1], 0.0);
        V[j].n_e = dmax(p[2], 0.0);
        V[j].B = dmax(p[3], 0.0);
        V[j].theta = p[4] * CUDART_PI / 180.0;
        V[j].psi = p[5] * CUDART_PI / 180.0;

        int em_flag = (int)p[6];
        V[j].GR_on = ((em_flag & 1) == 0);
        V[j].FF_on = ((em_flag & 2) == 0);
        V[j].HHe_on = ((em_flag & 4) == 0);
        V[j].force_isothermal = ((em_flag & 8) != 0);

        V[j].s_max = (smax_global > 0) ? smax_global : (int)p[7];
        V[j].s_min = (smin_global > 0) ? imax(smin_global, 2) : 2;
        V[j].n_H = dmax(p[8], 0.0);
        V[j].n_He = dmax(p[9], 0.0);

        V[j].DEM_on = (p[10] == 0 && NT > 1);
        V[j].DDM_on = (p[11] == 0 && NT > 1);

        V[j].ABcode = (int)p[12];
        if (V[j].ABcode < 0 || V[j].ABcode > 2) V[j].ABcode = 0;

        V[j].dfcode = (int)p[13];
        if (V[j].dfcode < 0 || V[j].dfcode > 2) V[j].dfcode = 0;

        V[j].S = p[14];
        V[j].j_ofs = j;

        V[j].Bx = V[j].B * sin(V[j].theta) * cos(V[j].psi);
        V[j].By = V[j].B * sin(V[j].theta) * sin(V[j].psi);
        V[j].Bz = V[j].B * cos(V[j].theta);

        if (V[j].DDM_on) {
            DDM_moments(T_arr, lnT_arr, DDM_arr + NT * j, NT, &(V[j].n_e), &(V[j].T0), tmp1, tmp2);
        } else if (V[j].DEM_on) {
            DEM_moments(T_arr, lnT_arr, DEM_arr + NT * j, NT, &(V[j].n_e), &(V[j].T0), tmp1, tmp2);
        }

        V[j].f_p = grff_e() * sqrt(V[j].n_e / grff_me() / CUDART_PI);
    }

    V[0].zstart = V[0].zstart1 = 0.0;
    for (int j = 1; j < Nz0; j++) {
        V[j].zstart = V[j - 1].zstart + V[j - 1].dz;
        V[j].zstart1 = V[j - 1].zstart1 + ((V[j - 1].n_e > 0) ? V[j - 1].dz : 0);
    }

    double LLOS = 0.0;
    for (int j = 0; j < Nz0; j++) LLOS += V[j].dz;

    return LLOS;
}

static __device__ void CompressVoxels(Voxel *V, int Nz0, int *Nz)
{
    int jmin;
    for (jmin = 0; jmin < Nz0; jmin++) if (V[jmin].n_e > 0) break;

    int jmax;
    for (jmax = Nz0 - 1; jmax >= 0; jmax--) if (V[jmax].n_e > 0) break;

    *Nz = 0;
    for (int j = jmin; j <= jmax; j++) if (V[j].dz > 0) {
        if (*Nz != j) V[*Nz] = V[j];
        (*Nz)++;
    }
}

static __device__ void ProcessVoxelGradients(Voxel *V, int Nz, int ref_on)
{
    for (int j = 0; j < Nz; j++) {
        if (j == 0 && j == (Nz - 1)) {
            V[j].B1 = V[j].B2 = V[j].B;
            V[j].Bz1 = V[j].Bz2 = V[j].Bz;
            V[j].B_a[0] = V[j].B_a[1] = V[j].Bx_a[0] = V[j].Bx_a[1] = V[j].By_a[0] = V[j].By_a[1] = V[j].Bz_a[0] = V[j].Bz_a[1] = 0.0;
            V[j].B_b[0] = V[j].B_b[1] = V[j].B;
            V[j].Bx_b[0] = V[j].Bx_b[1] = V[j].Bx;
            V[j].By_b[0] = V[j].By_b[1] = V[j].By;
            V[j].Bz_b[0] = V[j].Bz_b[1] = V[j].Bz;
            V[j].dB_dz[0] = V[j].dB_dz[1] = V[j].dtheta_dz[0] = V[j].dtheta_dz[1] = 0.0;
        } else {
            for (int k = 0; k < 2; k++) {
                int j1, j2;
                double z1, z2;

                if (j == 0) {
                    j1 = j;
                    j2 = j + 1;
                    z1 = V[j1].dz / 2.0;
                    z2 = V[j1].dz + V[j2].dz / 2.0;
                } else if (j == (Nz - 1)) {
                    j1 = j - 1;
                    j2 = j;
                    z1 = -V[j1].dz / 2.0;
                    z2 = V[j2].dz / 2.0;
                } else {
                    if (k == 0) {
                        j1 = j - 1;
                        j2 = j;
                        z1 = -V[j1].dz / 2.0;
                        z2 = V[j2].dz / 2.0;
                    } else {
                        j1 = j;
                        j2 = j + 1;
                        z1 = V[j1].dz / 2.0;
                        z2 = V[j1].dz + V[j2].dz / 2.0;
                    }
                }

                V[j].B_a[k] = V[j].dB_dz[k] = (V[j1].B - V[j2].B) / (z1 - z2);
                V[j].B_b[k] = (V[j2].B * z1 - V[j1].B * z2) / (z1 - z2);
                V[j].Bx_a[k] = (V[j1].Bx - V[j2].Bx) / (z1 - z2);
                V[j].Bx_b[k] = (V[j2].Bx * z1 - V[j1].Bx * z2) / (z1 - z2);
                V[j].By_a[k] = (V[j1].By - V[j2].By) / (z1 - z2);
                V[j].By_b[k] = (V[j2].By * z1 - V[j1].By * z2) / (z1 - z2);
                V[j].Bz_a[k] = (V[j1].Bz - V[j2].Bz) / (z1 - z2);
                V[j].Bz_b[k] = (V[j2].Bz * z1 - V[j1].Bz * z2) / (z1 - z2);
                V[j].dtheta_dz[k] = (V[j1].theta - V[j2].theta) / (z1 - z2);
            }

            V[j].B1 = V[j].B_b[0];
            V[j].B2 = V[j].B_a[1] * V[j].dz + V[j].B_b[1];
            V[j].Bz1 = V[j].Bz_b[0];
            V[j].Bz2 = V[j].Bz_a[1] * V[j].dz + V[j].Bz_b[1];
        }
    }

    if (ref_on) {
        if (Nz == 1) V[0].refTerm = 0.0;
        else {
            V[0].refTerm = (V[1].S - V[0].S) / (V[0].dz + V[1].dz) * 2.0 / V[0].S;
            V[Nz - 1].refTerm = (V[Nz - 1].S - V[Nz - 2].S) / (V[Nz - 2].dz + V[Nz - 1].dz) * 2.0 / V[Nz - 1].S;

            for (int i = 1; i < (Nz - 1); i++) {
                V[i].refTerm = (V[i + 1].S - V[i - 1].S) / (V[i].dz + (V[i - 1].dz + V[i + 1].dz) / 2.0) / V[i].S;
            }
        }
    } else {
        for (int i = 0; i < Nz; i++) V[i].refTerm = 0.0;
    }
}

static __device__ void AddLevel(Level *l, int s, double zstart, double zstart1, double z0, int *Nlev, int max_levels)
{
    int old = 0;
    for (int i = 0; i < *Nlev; i++) {
        if (l[i].s == s && l[i].z0 == z0) {
            old = 1;
            break;
        }
    }

    if (!old) {
        if (*Nlev < max_levels) {
            l[*Nlev].s = s;
            l[*Nlev].zstart = zstart;
            l[*Nlev].zstart1 = zstart1;
            l[*Nlev].z0 = z0;
            (*Nlev)++;
        }
    }
}

static __device__ void SortLevels(Level *l, int Nlev)
{
    if (Nlev > 1) {
        for (int i = 0; i < (Nlev - 1); i++) {
            for (int j = i + 1; j < Nlev; j++) {
                if (l[i].z0 > l[j].z0) {
                    Level a = l[i];
                    l[i] = l[j];
                    l[j] = a;
                }
            }
        }
    }
}

static __device__ int MW_Transfer(const int *Lparms, const double *Rparms, const double *Parms, const double *T_arr,
                                  const double *DEM_arr, const double *DDM_arr, double *RL)
{
    int Nz0 = Lparms[0];
    int Nf = Lparms[1];
    int NT = Lparms[2];

    // NOTE: AZ_on/custom zeta not supported in this path (GET_MW only).
    int smin_global = 0;
    int smax_global = 0;

    double *lnT_arr = nullptr;
    if (NT > 1) {
        lnT_arr = (double *)malloc(sizeof(double) * NT);
        if (!lnT_arr) return -2;
        for (int i = 0; i < NT; i++) lnT_arr[i] = log(T_arr[i]);
    }

    double *tmp1 = (double *)malloc(sizeof(double) * NT);
    double *tmp2 = (double *)malloc(sizeof(double) * NT);
    if (NT > 1 && (!tmp1 || !tmp2)) {
        if (lnT_arr) free(lnT_arr);
        if (tmp1) free(tmp1);
        if (tmp2) free(tmp2);
        return -3;
    }

    Voxel *V = (Voxel *)malloc(sizeof(Voxel) * Nz0);
    if (!V) {
        if (lnT_arr) free(lnT_arr);
        if (tmp1) free(tmp1);
        if (tmp2) free(tmp2);
        return -4;
    }

    double LLOS = ProcessVoxels(Nz0, Parms, NT, T_arr, lnT_arr, DEM_arr, DDM_arr, smin_global, smax_global, V, tmp1, tmp2);
    (void)LLOS;

    int Nz = 0;
    CompressVoxels(V, Nz0, &Nz);

    int ref_on = (Nz > 0);
    for (int i = 0; i < Nz; i++) if (V[i].S <= 0) ref_on = 0;

    ProcessVoxelGradients(V, Nz, ref_on);

    double Sang1 = Rparms[0] / (sqr(grff_AU()) * grff_sfu());
    double Sang2 = Sang1;
    if (ref_on) {
        Sang1 = V[0].S / (sqr(grff_AU()) * grff_sfu());
        Sang2 = V[Nz - 1].S / (sqr(grff_AU()) * grff_sfu());
    }

    int max_smax = 2;
    for (int i = 0; i < Nz; i++) max_smax = imax(max_smax, V[i].s_max);
    int max_levels = imax(4, 2 * max_smax);

    Level *levels = (Level *)malloc(sizeof(Level) * max_levels);
    if (!levels) {
        free(V);
        if (lnT_arr) free(lnT_arr);
        if (tmp1) free(tmp1);
        if (tmp2) free(tmp2);
        return -5;
    }

    for (int i = 0; i < Nf; i++) {
        double f = 0.0;
        if (Rparms[1] > 0) {
            double dnu = pow(10.0, Rparms[2]);
            f = (i == 0) ? Rparms[1] : Rparms[1] * pow(dnu, (double)i);
        } else {
            f = RL[i * OutSize] * 1e9;
        }

        double Lw = RL[i * OutSize + 1] / Sang1;
        double Rw = RL[i * OutSize + 2] / Sang1;
        double Ls = RL[i * OutSize + 3] / Sang1;
        double Rs = RL[i * OutSize + 4] / Sang1;
        double Le = RL[i * OutSize + 5] / Sang1;
        double Re = RL[i * OutSize + 6] / Sang1;

        double B_res = f * 2.0 * CUDART_PI * grff_me() * grff_c() / grff_e();

        for (int j = 0; j < Nz; j++) {
            int Nlev = 0;

            for (int lr = 0; lr < 2; lr++) {
                int QTfound = (lr == 0) ? V[j].Bz1 * V[j].Bz < 0 : V[j].Bz * V[j].Bz2 < 0;
                if (QTfound) {
                    double z0 = -V[j].Bz_b[lr] / V[j].Bz_a[lr];
                    if (z0 != (V[j].dz / 2.0)) AddLevel(levels, 0, V[j].zstart, V[j].zstart1, z0, &Nlev, max_levels);
                }
            }

            if (V[j].GR_on) for (int lr = 0; lr < 2; lr++) {
                double B1 = (lr == 0) ? V[j].B1 : V[j].B2;
                double B2 = V[j].B;

                if (B1 > 0 && B2 > 0 && B1 != B2) {
                    int smin = (int)ceil(B_res / dmax(B1, B2));
                    int smax = (int)floor(B_res / dmin(B1, B2));
                    smin = imax(smin, V[j].s_min);
                    smax = imin(smax, V[j].s_max);

                    for (int s = smin; s <= smax; s++) {
                        double z0 = (B_res / s - V[j].B_b[lr]) / V[j].B_a[lr];
                        if (z0 != (V[j].dz / 2.0)) AddLevel(levels, s, V[j].zstart, V[j].zstart1, z0, &Nlev, max_levels);
                    }
                }
            }

            SortLevels(levels, Nlev);

            for (int k = 0; k <= Nlev; k++) {
                double z1 = (k == 0) ? 0.0 : levels[k - 1].z0;
                double z2 = (k == Nlev) ? V[j].dz : levels[k].z0;
                double dz = z2 - z1;

                if (dz > 0) {
                    double zc = (z1 + z2) / 2.0;
                    int lr = (zc < (V[j].dz / 2.0)) ? 0 : 1;
                    double Bx = V[j].Bx_a[lr] * zc + V[j].Bx_b[lr];
                    double By = V[j].By_a[lr] * zc + V[j].By_b[lr];
                    double Bz = V[j].Bz_a[lr] * zc + V[j].Bz_b[lr];

                    double B = sqrt(sqr(Bx) + sqr(By) + sqr(Bz));
                    double theta = (B > 0) ? acos(Bz / B) : 0.0;
                    double f_B = grff_e() * B / grff_me() / grff_c() / (2.0 * CUDART_PI);

                    double jXff = 0, kXff = 0, jOff = 0, kOff = 0;
                    if (V[j].FF_on) {
                        if (V[j].DEM_on && !V[j].force_isothermal) {
                            FindFF_DEM_XO(f, theta, V[j].f_p, f_B, T_arr, lnT_arr, DEM_arr + NT * V[j].j_ofs, NT, V[j].ABcode,
                                         0, 0, 0, nullptr, nullptr, nullptr,
                                         &jXff, &kXff, &jOff, &kOff, tmp1, tmp2);
                        } else {
                            FindFF_single(f, theta, -1, V[j].f_p, f_B, V[j].T0, V[j].n_e, V[j].ABcode,
                                          0, 0, 0, nullptr, nullptr, nullptr, &jXff, &kXff);
                            FindFF_single(f, theta,  1, V[j].f_p, f_B, V[j].T0, V[j].n_e, V[j].ABcode,
                                          0, 0, 0, nullptr, nullptr, nullptr, &jOff, &kOff);
                        }
                    }

                    double jXen = 0, kXen = 0, jOen = 0, kOen = 0;
                    if (V[j].HHe_on) {
                        FindNeutralsEffect(f, theta, -1, V[j].f_p, f_B, V[j].T0, V[j].n_e, V[j].n_H, V[j].n_He, &jXen, &kXen);
                        FindNeutralsEffect(f, theta,  1, V[j].f_p, f_B, V[j].T0, V[j].n_e, V[j].n_H, V[j].n_He, &jOen, &kOen);
                    }

                    double jX = jXff + jXen;
                    double kX = kXff + kXen + V[j].refTerm;
                    double jO = jOff + jOen;
                    double kO = kOff + kOen + V[j].refTerm;

                    double tauX = -kX * dz;
                    double eX = (tauX < 700) ? exp(tauX) : 0.0;
                    double dIX = (kX == 0.0 || tauX > 700) ? 0.0 : jX / kX * ((1.0 - eX) ? 1.0 - eX : -tauX);
                    double tauO = -kO * dz;
                    double eO = (tauO < 700) ? exp(tauO) : 0.0;
                    double dIO = (kO == 0.0 || tauO > 700) ? 0.0 : jO / kO * ((1.0 - eO) ? 1.0 - eO : -tauO);

                    if (theta > (CUDART_PI / 2.0)) {
                        Lw = dIX + Lw * eX;
                        Ls = dIX + Ls * eX;
                        Le = dIX + Le * eX;
                        Rw = dIO + Rw * eO;
                        Rs = dIO + Rs * eO;
                        Re = dIO + Re * eO;
                    } else {
                        Lw = dIO + Lw * eO;
                        Ls = dIO + Ls * eO;
                        Le = dIO + Le * eO;
                        Rw = dIX + Rw * eX;
                        Rs = dIX + Rs * eX;
                        Re = dIX + Re * eX;
                    }
                }

                if (k != Nlev) {
                    int lr = (levels[k].z0 < (V[j].dz / 2.0)) ? 0 : 1;
                    double Bx = V[j].Bx_a[lr] * levels[k].z0 + V[j].Bx_b[lr];
                    double By = V[j].By_a[lr] * levels[k].z0 + V[j].By_b[lr];
                    double Bz = V[j].Bz_a[lr] * levels[k].z0 + V[j].Bz_b[lr];
                    double dB_dz = fabs(V[j].dB_dz[lr]);
                    double dtheta_dz = fabs(V[j].dtheta_dz[lr]);

                    if (levels[k].s < 2) {
                        double a = Lw; Lw = Rw; Rw = a;

                        double B = sqrt(sqr(Bx) + sqr(By) + sqr(Bz));
                        double QT = pow(grff_e(), 5) / (32.0 * CUDART_PI * CUDART_PI * pow(grff_me(), 4) * pow(grff_c(), 4)) *
                                    V[j].n_e * sqr(B) * B / sqr(sqr(f)) / dtheta_dz;
                        QT = exp(-QT);
                        a = Le * QT + Re * (1.0 - QT);
                        Re = Re * QT + Le * (1.0 - QT);
                        Le = a;
                    } else {
                        double B = B_res / levels[k].s;
                        double f_B = f / levels[k].s;
                        double theta = acos(Bz / sqrt(sqr(Bx) + sqr(By) + sqr(Bz)));
                        double LB = B / dB_dz;

                        double tauX = 0.0, tauO = 0.0, I0X = 0.0, I0O = 0.0;
                        if (V[j].DDM_on && !V[j].force_isothermal) {
                            FindGR_DDM_XO(f, theta, levels[k].s, V[j].f_p, f_B, T_arr, lnT_arr, DDM_arr + NT * V[j].j_ofs, NT, LB,
                                          &tauX, &I0X, &tauO, &I0O, tmp1, tmp2);
                        } else {
                            FindGR_single(f, theta, -1, levels[k].s, V[j].f_p, f_B, V[j].n_e, V[j].T0, LB, &tauX, &I0X);
                            FindGR_single(f, theta,  1, levels[k].s, V[j].f_p, f_B, V[j].n_e, V[j].T0, LB, &tauO, &I0O);
                        }

                        double eX = exp(-tauX);
                        double dIX = I0X * ((1.0 - eX) ? 1.0 - eX : tauX);
                        double eO = exp(-tauO);
                        double dIO = I0O * ((1.0 - eO) ? 1.0 - eO : tauO);

                        if (theta > (CUDART_PI / 2.0)) {
                            Lw = dIX + Lw * eX;
                            Ls = dIX + Ls * eX;
                            Le = dIX + Le * eX;
                            Rw = dIO + Rw * eO;
                            Rs = dIO + Rs * eO;
                            Re = dIO + Re * eO;
                        } else {
                            Lw = dIO + Lw * eO;
                            Ls = dIO + Ls * eO;
                            Le = dIO + Le * eO;
                            Rw = dIX + Rw * eX;
                            Rs = dIX + Rs * eX;
                            Re = dIX + Re * eX;
                        }
                    }
                }
            }
        }

        RL[i * OutSize + 0] = f / 1e9;
        RL[i * OutSize + 1] = Lw * Sang2;
        RL[i * OutSize + 2] = Rw * Sang2;
        RL[i * OutSize + 3] = Ls * Sang2;
        RL[i * OutSize + 4] = Rs * Sang2;
        RL[i * OutSize + 5] = Le * Sang2;
        RL[i * OutSize + 6] = Re * Sang2;
    }

    free(levels);
    free(V);
    if (lnT_arr) free(lnT_arr);
    if (tmp1) free(tmp1);
    if (tmp2) free(tmp2);

    return 0;
}

extern "C" __global__ void get_mw_kernel(const int *Lparms,
                                         const double *Rparms,
                                         const double *Parms,
                                         const double *T_arr,
                                         const double *DEM_arr,
                                         const double *DDM_arr,
                                         double *RL,
                                         int *status)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int res = MW_Transfer(Lparms, Rparms, Parms, T_arr, DEM_arr, DDM_arr, RL);
        if (status) status[0] = res;
    }
}

extern "C" __global__ void get_mw_slice_kernel(const int *Lparms_M,
                                               const double *Rparms_M,
                                               const double *Parms_M,
                                               const double *T_arr,
                                               const double *DEM_arr_M,
                                               const double *DDM_arr_M,
                                               double *RL_M,
                                               int *status)
{
    const int pix = blockIdx.x;
    const int Npix = Lparms_M ? Lparms_M[0] : 0;
    if (pix >= Npix) return;

    if (threadIdx.x == 0) {
        const int *Lparms = Lparms_M + 1;
        const int Nz = Lparms[0];
        const int Nf = Lparms[1];
        const int NT = Lparms[2];

        const double *Rparms = Rparms_M + pix * RpSize;
        const double *Parms = Parms_M + pix * Nz * InSize;
        const double *DEM_arr = DEM_arr_M + pix * Nz * NT;
        const double *DDM_arr = DDM_arr_M + pix * Nz * NT;
        double *RL = RL_M + pix * Nf * OutSize;

        int res = MW_Transfer(Lparms, Rparms, Parms, T_arr, DEM_arr, DDM_arr, RL);
        if (status) status[pix] = res;
    }
}
