// Flash's long-range kernel (_lr_kernel) in CUDA. Electrostatics (Ewald/PME real
// space or damped, tabulated phi) and the MBD-free dispersion of SO3LR (TS-like C6 with
// damping, optional PME residual), per atom over all partners within the reach, both ends of
// every pair (gather only: deterministic). One warp per atom (atoms in cell order); the cell
// rows walked as flash walks them; hits compacted into a per-warp queue so all 32 lanes
// evaluate real pairs.
#include <cstdint>

#include "ffi_common.cuh"

constexpr int LW = 8;          // warps per block

struct LrParams {
  const double *px, *py, *pz, *box;          // cell order (index k: atom order[k])
  const int *z, *order, *cell_start, *cell_of;
  const float *q, *a0r, *c6r, *alphas, *c6tab, *phi_v, *phi_d, *lrs, *c6lr_v, *c6lr_d;   // q, z, a0r, c6r: cell order
  const float *e_in, *f_in;       // f_in: [6][n] (fx, fy, fz, wx, wy, wz)
  const int* row0;                // this device's rows: row0 <= atom < row0 + nl
  float *e_out, *gq, *ga0, *gc6, *f_out;
  int nl;
  int n, NX, NY, NZ, KC, ELEC, DISP, DPME, FAR, EL_SWITCH, DISP_SWITCH;
};

__device__ __forceinline__ void switching(float x, float on, float off, float& f, float& df) {
  float c = (x - on) / (off - on), pp = 1.f - c;
  float sa = pp > 0.f ? expf(-1.f / pp) : 0.f, sb = c > 0.f ? expf(-1.f / c) : 0.f;
  float da = (pp > 0.f ? -sa / (pp * pp) : 0.f) / (off - on);
  float db = (c > 0.f ? sb / (c * c) : 0.f) / (off - on);
  float den = sa + sb;
  bool ok = den > 0.f;
  float ds = ok ? den : 1.f;
  f = ok ? sa / ds : 0.f;
  df = ok ? (da * sb - sa * db) / (ds * ds) : 0.f;
}

__device__ __forceinline__ void hermite(const float* tv, const float* td, int ti, float tu, float phs, float& v,
                                        float& dv) {
  float f0 = tv[ti], f1 = tv[ti + 1], d0 = td[ti], d1 = td[ti + 1];
  float u2 = tu * tu, u3 = u2 * tu;
  v = (2.f * u3 - 3.f * u2 + 1.f) * f0 + (3.f * u2 - 2.f * u3) * f1 + phs * ((u3 - 2.f * u2 + tu) * d0 + (u3 - u2) * d1);
  dv = (6.f * u2 - 6.f * tu) * (f0 - f1) / phs + (3.f * u2 - 4.f * tu + 1.f) * d0 + (3.f * u2 - 2.f * tu) * d1;
}

struct Acc {
  float e, gq, gal, gc6, fx, fy, fz, wx, wy, wz;
};

struct AtomA {
  float qa, al_a, c6_a;
};

// one pair a <- b at displacement (dx, dy, dz) = r_b - r_a; b's charge, alpha and C6
__device__ __forceinline__ void pair_terms(const LrParams& p, const AtomA& A, float qb, float al_b, float c6_b,
                                           float dx, float dy, float dz, Acc& acc) {
  const float rc = p.lrs[0], phs = p.lrs[1], phn = p.lrs[2], ke = p.lrs[3], dstart = p.lrs[4], gsc = p.lrs[5],
              fs = p.lrs[6], hartree = p.lrs[7], bohr = p.lrs[8], rtab = p.lrs[10];
  const float r = sqrtf(dx * dx + dy * dy + dz * dz);
  float de = 0.f;
  const float rt = p.FAR ? fminf(r, rtab) : r;
  const float t = rt / phs;
  const float fi = fminf(fmaxf(floorf(t), 0.f), phn - 1.f);
  const int ti = (int)fi;
  const float tu = t - fi;
  if (p.ELEC) {
    float phi, dphi;
    hermite(p.phi_v, p.phi_d, ti, tu, phs, phi, dphi);
    if (p.FAR && r > rtab) {
      float inv = 1.f / r, fc = -1.f / (rc * rc), f = 1.f, df = 0.f;
      if (p.EL_SWITCH) switching(r, 0.45f * rc, rc, f, df);
      phi = inv - 1.f / rc - (1.f - f) * fc * (r - rc);
      dphi = -inv * inv - (1.f - f) * fc + df * fc * (r - rc);
    }
    float kq = ke * qb;
    acc.e += 0.5f * kq * A.qa * phi;
    acc.gq += kq * phi;
    de += kq * A.qa * dphi;
  }
  if (p.DISP) {
    const float b0 = -0.00433008f, b1 = 0.24428889f, b2 = 0.04125273f, b3 = -0.00078893f, p0 = 5.08f;
    float al_a = A.al_a, c6_a = A.c6_a;
    float den = al_a * al_a * c6_b + al_b * al_b * c6_a;
    float c6p = 2.f * c6_a * c6_b * al_a * al_b / den;
    float dc6_dal = (2.f * c6_a * c6_b * al_b - c6p * 2.f * al_a * c6_b) / den;
    float dc6_dc6 = (2.f * c6_b * al_a * al_b - c6p * al_b * al_b) / den;
    float alp = 0.5f * (al_a + al_b);
    float a17 = expf(logf(alp) / 7.f);
    float rv = fs * a17;
    float sg = ((b3 * rv + b2) * rv + b1) * rv + b0;
    float gam = 0.5f / (sg * sg);
    float dgam = -(3.f * b3 * rv * rv + 2.f * b2 * rv + b1) * rv / (7.f * alp) / (sg * sg * sg);
    float pp = gsc * p0 * a17;
    float dp = pp / (7.f * alp);
    float R = r / bohr, R2 = R * R, R6 = R2 * R2 * R2;
    float p2 = pp * pp, p6 = p2 * p2 * p2;
    float i6 = 1.f / (R6 + p6), i8 = 1.f / (R6 * R2 + p6 * p2), i10 = 1.f / (R6 * R2 * R2 + p6 * p2 * p2);
    float ig = 1.f / gam;
    float c8 = 5.f * ig * c6p, c10 = 245.f / 8.f * ig * ig * c6p;
    float v = -c6p * i6 - c8 * i8 - c10 * i10;
    float dvr = (6.f * R2 * R2 * R * c6p * i6 * i6 + 8.f * R6 * R * c8 * i8 * i8 + 10.f * R6 * R2 * R * c10 * i10 * i10) / bohr;
    float dv_c6 = -i6 - 5.f * ig * i8 - 245.f / 8.f * ig * ig * i10;
    float dv_gam = 5.f * c6p * ig * ig * i8 + 2.f * 245.f / 8.f * c6p * ig * ig * ig * i10;
    float dv_p = 6.f * p2 * p2 * pp * c6p * i6 * i6 + 8.f * p6 * pp * c8 * i8 * i8 + 10.f * p6 * p2 * pp * c10 * i10 * i10;
    float w = 1.f, dw = 0.f;
    if (p.DISP_SWITCH) switching(r, dstart, rc, w, dw);
    float hw = hartree * w;
    float g6 = hw * dv_c6 * dc6_dc6;
    if (p.DPME) {
      float c6g = sqrtf(c6_a * c6_b);
      float lr6, dlr6;
      hermite(p.c6lr_v, p.c6lr_d, ti, tu, phs, lr6, dlr6);
      v += c6g * lr6;
      dvr += c6g * dlr6;
      g6 += hw * lr6 * c6g / (2.f * c6_a);
    }
    acc.e += 0.5f * hw * v;
    acc.gal += hw * (dv_c6 * dc6_dal + 0.5f * (dv_gam * dgam + dv_p * dp));
    acc.gc6 += g6;
    de += hartree * (w * dvr + dw * v);
  }
  const float fr = de / r;
  acc.fx += fr * dx; acc.fy += fr * dy; acc.fz += fr * dz;
  acc.wx += fr * dx * dx; acc.wy += fr * dy * dy; acc.wz += fr * dz * dz;   // per-axis virial
}

__device__ __forceinline__ void pair(const LrParams& p, const AtomA& A, int j, float dx, float dy, float dz, Acc& acc) {
  const int zb = p.z[j] - 1;
  pair_terms(p, A, p.q[j], p.alphas[zb] * p.a0r[j], p.c6tab[zb] * p.c6r[j], dx, dy, dz, acc);
}

__global__ void __launch_bounds__(LW * 32) lr_kernel(LrParams p) {
  __shared__ int qj[LW][64];
  __shared__ float qx[LW][64], qy[LW][64], qz[LW][64];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int wi = blockIdx.x * LW + warp;
  if (wi >= p.n) return;
  const int a = p.order[wi];                       // atoms in cell order; their data at wi
  if (a < p.row0[0] || a >= p.row0[0] + p.nl) return;   // another device's row
  const double Lx = p.box[0], Ly = p.box[1], Lz = p.box[2];
  const double iLx = 1.0 / Lx, iLy = 1.0 / Ly, iLz = 1.0 / Lz;
  const double ax = p.px[wi], ay = p.py[wi], az = p.pz[wi];
  const float reach = p.lrs[9], reach2 = reach * reach;
  const int NX = p.NX, NY = p.NY, NZ = p.NZ, KC = p.KC;
  const double sxd = Lx / NX, syd = Ly / NY, szd = Lz / NZ;
  const float sx = (float)sxd, sy = (float)syd, sz = (float)szd;
  const int c = p.cell_of[a];
  const int cx = c % NX, cy = (c / NX) % NY, cz = c / (NX * NY);
  const float ox = (float)fmin(fmax(ax - Lx * floor(ax / Lx) - cx * sxd, 0.0), sxd);
  const float oy = (float)fmin(fmax(ay - Ly * floor(ay / Ly) - cy * syd, 0.0), syd);
  const float oz = (float)fmin(fmax(az - Lz * floor(az / Lz) - cz * szd, 0.0), szd);
  AtomA A;
  const int za = p.z[wi] - 1;
  A.qa = p.q[wi];
  const float al_ta = p.alphas[za], c6_ta = p.c6tab[za];
  A.al_a = al_ta * p.a0r[wi];
  A.c6_a = c6_ta * p.c6r[wi];
  Acc acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  int qn = 0;
  const int W2 = 2 * KC + 1;
  for (int k = 0; k < W2 * W2; ++k) {
    const int dzc = k / W2 - KC, dyc = k % W2 - KC;
    const float gz_ = dzc > 0 ? dzc * sz - oz : (dzc < 0 ? oz + (-dzc - 1) * sz : 0.f);
    const float gy_ = dyc > 0 ? dyc * sy - oy : (dyc < 0 ? oy + (-dyc - 1) * sy : 0.f);
    const float g2 = gy_ * gy_ + gz_ * gz_;
    if (!(g2 < reach2)) continue;
    const float xr = sqrtf(fmaxf(reach2 - g2, 0.f));
    const int lo = max(cx + (int)floorf((ox - xr) / sx), cx - KC);
    const int hi = min(cx + (int)floorf((ox + xr) / sx), cx + KC);
    const int zc = (cz + dzc + NZ) % NZ, yc = (cy + dyc + NY) % NY;
    const int row = (zc * NY + yc) * NX;
    for (int part = 0; part < 2; ++part) {
      int c0, c1;
      if (part == 0) { c0 = max(lo, 0); c1 = min(hi, NX - 1); }
      else if (lo < 0) { c0 = lo + NX; c1 = NX - 1; }
      else if (hi >= NX) { c0 = 0; c1 = hi - NX; }
      else { c0 = 1; c1 = 0; }
      if (hi < lo) c1 = c0 - 1;
      if (c1 < c0) continue;
      const int s = p.cell_start[row + c0], e = p.cell_start[row + c1 + 1];
      for (int k0 = s; k0 < e; k0 += 32) {
        const int kk = k0 + lane;
        bool hit = false;
        int j = 0;
        float dx = 0.f, dy = 0.f, dz = 0.f;
        if (kk < e) {
          j = kk;                                   // cell order: contiguous loads
          double dxd = p.px[j] - ax, dyd = p.py[j] - ay, dzd = p.pz[j] - az;
          dxd -= Lx * floor(dxd * iLx + 0.5);
          dyd -= Ly * floor(dyd * iLy + 0.5);
          dzd -= Lz * floor(dzd * iLz + 0.5);
          dx = (float)dxd; dy = (float)dyd; dz = (float)dzd;
          float r2 = dx * dx + dy * dy + dz * dz;
          hit = r2 < reach2 && j != wi && r2 > 0.f;
        }
        const unsigned m = __ballot_sync(0xffffffff, hit);
        if (hit) {
          const int pos = qn + __popc(m & ((1u << lane) - 1));
          qj[warp][pos] = j; qx[warp][pos] = dx; qy[warp][pos] = dy; qz[warp][pos] = dz;
        }
        qn += __popc(m);
        __syncwarp();
        if (qn >= 32) {
#ifdef NOPAIR
          acc.e += qx[warp][lane];
#else
          pair(p, A, qj[warp][lane], qx[warp][lane], qy[warp][lane], qz[warp][lane], acc);
#endif
          __syncwarp();
          if (lane < qn - 32) {
            qj[warp][lane] = qj[warp][lane + 32]; qx[warp][lane] = qx[warp][lane + 32];
            qy[warp][lane] = qy[warp][lane + 32]; qz[warp][lane] = qz[warp][lane + 32];
          }
          qn -= 32;
          __syncwarp();
        }
      }
    }
  }
#ifdef NOPAIR
  if (lane < qn) acc.e += qx[warp][lane];
#else
  if (lane < qn) pair(p, A, qj[warp][lane], qx[warp][lane], qy[warp][lane], qz[warp][lane], acc);
#endif
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    acc.e += __shfl_xor_sync(0xffffffff, acc.e, o);
    acc.gq += __shfl_xor_sync(0xffffffff, acc.gq, o);
    acc.gal += __shfl_xor_sync(0xffffffff, acc.gal, o);
    acc.gc6 += __shfl_xor_sync(0xffffffff, acc.gc6, o);
    acc.fx += __shfl_xor_sync(0xffffffff, acc.fx, o);
    acc.fy += __shfl_xor_sync(0xffffffff, acc.fy, o);
    acc.fz += __shfl_xor_sync(0xffffffff, acc.fz, o);
    acc.wx += __shfl_xor_sync(0xffffffff, acc.wx, o);
    acc.wy += __shfl_xor_sync(0xffffffff, acc.wy, o);
    acc.wz += __shfl_xor_sync(0xffffffff, acc.wz, o);
  }
  if (lane == 0) {
    const int n = p.n;
    p.e_out[a] = p.e_in[a] + acc.e;
    p.gq[a] = acc.gq;
    p.ga0[a] = acc.gal * al_ta;
    p.gc6[a] = acc.gc6 * c6_ta;
    p.f_out[a] = p.f_in[a] + acc.fx;
    p.f_out[n + a] = p.f_in[n + a] + acc.fy;
    p.f_out[2 * n + a] = p.f_in[2 * n + a] + acc.fz;
    p.f_out[3 * n + a] = p.f_in[3 * n + a] + acc.wx;
    p.f_out[4 * n + a] = p.f_in[4 * n + a] + acc.wy;
    p.f_out[5 * n + a] = p.f_in[5 * n + a] + acc.wz;
  }
}



static ffi::Error LrImpl(cudaStream_t stream, F64 px, F64 py, F64 pz, F64 box, S32 z, S32 order, S32 cell_start,
                         S32 cell_of, F32 q, F32 a0r, F32 c6r, F32 alphas, F32 c6tab, F32 phi_v, F32 phi_d, F32 lrs,
                         F32 c6lr_v, F32 c6lr_d, F32 e_in, F32 f_in, S32 row0, RF32 e_out, RF32 gq, RF32 ga0,
                         RF32 gc6, RF32 f_out, int64_t NX, int64_t NY, int64_t NZ, int64_t KC, int64_t flags,
                         int64_t NL) {
  LrParams p;
  p.px = px.typed_data(); p.py = py.typed_data(); p.pz = pz.typed_data(); p.box = box.typed_data();
  p.z = z.typed_data(); p.order = order.typed_data(); p.cell_start = cell_start.typed_data();
  p.cell_of = cell_of.typed_data();
  p.q = q.typed_data(); p.a0r = a0r.typed_data(); p.c6r = c6r.typed_data(); p.alphas = alphas.typed_data();
  p.c6tab = c6tab.typed_data(); p.phi_v = phi_v.typed_data(); p.phi_d = phi_d.typed_data();
  p.lrs = lrs.typed_data(); p.c6lr_v = c6lr_v.typed_data(); p.c6lr_d = c6lr_d.typed_data();
  p.e_in = e_in.typed_data(); p.f_in = f_in.typed_data();
  p.e_out = e_out->typed_data(); p.gq = gq->typed_data(); p.ga0 = ga0->typed_data(); p.gc6 = gc6->typed_data();
  p.f_out = f_out->typed_data();
  p.n = (int)px.dimensions()[0];
  p.NX = (int)NX; p.NY = (int)NY; p.NZ = (int)NZ; p.KC = (int)KC; p.nl = (int)NL; p.row0 = row0.typed_data();
  p.ELEC = flags & 1; p.DISP = (flags >> 1) & 1; p.DPME = (flags >> 2) & 1; p.FAR = (flags >> 3) & 1;
  p.EL_SWITCH = (flags >> 4) & 1; p.DISP_SWITCH = (flags >> 5) & 1;
  lr_kernel<<<(p.n + LW - 1) / LW, LW * 32, 0, stream>>>(p);
  return launch_status();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(LongRange, LrImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<F64>().Arg<F64>().Arg<F64>().Arg<F64>()                         // px py pz box
        .Arg<S32>().Arg<S32>().Arg<S32>().Arg<S32>()                         // z order cell_start cell_of
        .Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>()              // q a0r c6r alphas c6tab
        .Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>()              // phi_v phi_d lrs c6lr_v c6lr_d
        .Arg<F32>().Arg<F32>().Arg<S32>()                                               // e_in f_in [6, n]
        .Ret<F32>().Ret<F32>().Ret<F32>().Ret<F32>().Ret<F32>()              // e gq ga0 gc6 f [6, n]
        .Attr<int64_t>("NX").Attr<int64_t>("NY").Attr<int64_t>("NZ").Attr<int64_t>("KC").Attr<int64_t>("flags")
        .Attr<int64_t>("NL"));
