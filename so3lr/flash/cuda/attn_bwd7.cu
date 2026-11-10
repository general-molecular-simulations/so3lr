// One layer's attention backward of SO3LR (both channels), pipelined like the forward. Producer warps stage per 32-pair tile: silu'(z) of the spherical filters'
// hidden layers, d(radial coefficients)/drt, and the degree channel's Y . dE/dev both ways.
// Consumer warps (one per channel and head) read the forward's w, form both directions'
// scores and their gradients (written per directed slot for the gather), dE/dw, its radial
// part (dw/drt = dU @ V on tensor cores) and dE/dhz (dE/dw @ wb^T on tensor cores), per-head
// partials into fixed shared-memory slots (deterministic). The consumers' epilogue turns them
// into each pair's ev gradient and force, equal and opposite on its two atoms.
#include <cstdint>

#include "attn_common.cuh"
#include "ffi_common.cuh"

constexpr int NSC = 5 * NH;               // slot scalars: gs, gr, ce s' / avg_nc (heads), gs, gr (degrees)
constexpr int NFW = 6;                    // per slot: the force (3) and the per-axis virial (3)
constexpr int FVS = 8;                    // the slot record's floats: NFW, padded to two float4
#ifndef TP
#define TP 32
#endif
constexpr int MT = TP / 16;
#ifndef NSTAGES
#define NSTAGES 2
#endif
#ifdef CHSPLIT
// one channel per CTA (blockIdx.y): 4 producer + HC consumer warps; each channel's CTA writes
// its own copy of the per-slot forces and ev gradients, which the gather sums
constexpr int NCH = 1, NCW = HC;
#else
constexpr int NCH = 2, NCW = 2 * HC;
#endif
#ifndef CTAS                  // CTAs per SM (the registers' bound): as many as the shared memory holds, up to two
#define CTAS (NCH == 1 && HC == 4 ? 2 : 1)
#endif
constexpr int NPW = 4, NTH = (NPW + NCW) * 32, NST = NSTAGES;
constexpr int BSW = FC + 4;   // wb rows read transposed (gz = gw @ wb^T: B[kk][n] = wb[n][kk])

struct B7Params {
  const float *q1, *k1, *v1, *q2, *k2, *ev, *gdx, *gdev;
  const int4* PI;
  const float4* PG;
  const int* off;
  const float *u1d, *u1v, *u2d, *u2v;
  const float *wa1, *ba1, *wb1, *wa2, *ba2, *wb2, *vr1, *vr2, *cdeg, *scal, *par;
  const float* Wst;
  const float* Sf;               // the forward's scores per directed slot [slots][2 NH]
  float *SC, *FV, *GEV;
  int n, nslot, pmax;            // nslot = nl K: the range's directed slots (a pair's reverse slot >= nslot
};                               // is a shadow: its partner is outside the range; no scalars, gradient or force)

struct B7Stage {
  float DZ[NCH][TP][SH + 8];     // silu'(z) (row stride 40: conflict-free float2 reads)
  float DU[NCH][TP][R + 4];      // d(radial coefficients)/d(rt / spacing ...): dU/drt
  float ga2[TP][NH], gar2[TP][NH]; // degree channel: sum_m Y_m dE/dev_a[m], par sum_m Y_m dE/dev_b[m]
  float ga1[TP][HC], gar1[TP][HC]; // heads channel: dE/dx_a . v_b and dE/dx_b . v_a per head of the CTA
  int4 pi[TP];
  float4 g0[TP], g1[TP];         // (ux, uy, uz, r), (ce, dce, drt, tc)
  float sf[TP][2][2 * NH];       // the forward's scores of both directed slots (prefetched)
};

struct B7Epi {
  float GCD[NCW][TP][NH];        // each head's part of dE/dc_d per pair
  float GRR[TP][NCW], GCE[TP][NCW];
  float SCA[TP][NSC], SCB[TP][NSC];  // each pair's slot scalars, both directed slots
};


#ifdef B_NOQK
#define LQB(ptr) make_float2(1.f, 1.f)
#else
#define LQB(ptr) (*reinterpret_cast<const float2*>(ptr))
#endif
#ifdef B_NOMMA
#define B_MMA 0
#else
#define B_MMA 1
#endif
#ifdef B_NOEPI
#define B_EPI 0
#else
#define B_EPI 1
#endif

// A pair's ev gradient and force from its chain totals (dE/dc_d per degree entry, dE/drt,
// dE/dce), 8 threads per pair (part), into both directed slots (the reverse one unless a
// shadow). The angular force takes the degree channel's scores of both slots from the stage
// (sf: the forward's 2 NH scores of the pair's slot, then its reverse slot's).
__device__ __forceinline__ void pair_finish(const B7Params& p, float* FV, float* GEV, bool angular, int4 pi, float4 g0,
                                            float4 g1, const float* gcd, float grr, float gce, const float* sf, int part,
                                            bool valid) {
  float gx = 0.f, gy = 0.f, gzv = 0.f;
  int yi[MC], ei[MC];                    // this thread's components: their harmonics and entries
#pragma unroll
  for (int k = 0; k < MC; ++k) { yi[k] = yidx(part * MC + k); ei[k] = entry_of(part * MC + k); }
  float dyx[MC], dyy[MC], dyz[MC];
#pragma unroll
  for (int k = 0; k < MC; ++k) { dyx[k] = 0.f; dyy[k] = 0.f; dyz[k] = 0.f; }
#pragma unroll
  for (int j = 0; j < 24; ++j) {
    float ax, ay, az;
    dylm(j, g0.x, g0.y, g0.z, ax, ay, az);
#pragma unroll
    for (int k = 0; k < MC; ++k) {
      const bool hit = yi[k] == j;
      dyx[k] = hit ? ax : dyx[k]; dyy[k] = hit ? ay : dyy[k]; dyz[k] = hit ? az : dyz[k];
    }
  }
  const bool rev = valid && pi.w < p.nslot;
#pragma unroll
  for (int k = 0; k < MC; ++k) {
    const int m = part * MC + k, en = ei[k];
    const float df = valid ? p.ev[(size_t)pi.y * M + m] - p.ev[(size_t)pi.x * M + m] : 0.f;
    float gce_d = 0.f;
#pragma unroll
    for (int d = 0; d < NH; ++d) gce_d = (d == en) ? gcd[d] : gce_d;
    const float sg = 2.f * gce_d * p.cdeg[en] * df;
    if (valid) {
      GEV[(size_t)pi.z * M + m] = -sg;
      if (rev) GEV[(size_t)pi.w * M + m] = sg;
    }
    const bool ang = valid && angular;
    const float al2 = ang ? sf[NH + en] : 0.f, alr2 = ang ? sf[3 * NH + en] : 0.f;
    const float G = ang ? al2 * p.gdev[(size_t)pi.x * M + m] + p.par[en] * alr2 * p.gdev[(size_t)pi.y * M + m] : 0.f;
    gx += G * dyx[k]; gy += G * dyy[k]; gzv += G * dyz[k];
  }
#pragma unroll
  for (int o = 1; o <= 4; o <<= 1) {
    gx += __shfl_xor_sync(0xffffffff, gx, o);
    gy += __shfl_xor_sync(0xffffffff, gy, o);
    gzv += __shfl_xor_sync(0xffffffff, gzv, o);
  }
  if (part == 0 && valid) {
    const float rr = g0.w, ux = g0.x, uy = g0.y, uz = g0.z;
    const float fr = grr * g1.z + gce * g1.y;
    const float proj = gx * ux + gy * uy + gzv * uz, inv_r = 1.f / rr;
    const float fx = fr * ux + (gx - proj * ux) * inv_r;
    const float fy = fr * uy + (gy - proj * uy) * inv_r;
    const float fz = fr * uz + (gzv - proj * uz) * inv_r;
    // per slot: the force (fx, fy, fz) and the per-axis virial (the force component times the
    // displacement component, rr u_a; the angular part cancels only in their sum), padded to FVS;
    // the reverse slot: the opposite force, the same virial
    const float wx = fx * rr * ux, wy = fy * rr * uy, wz = fz * rr * uz;
    auto put = [&](int slot, float sgn) {
      *reinterpret_cast<float4*>(FV + (size_t)slot * FVS) = make_float4(sgn * fx, sgn * fy, sgn * fz, wx);
      *reinterpret_cast<float4*>(FV + (size_t)slot * FVS + 4) = make_float4(wy, wz, 0.f, 0.f);
    };
    put(pi.z, 1.f);
    if (rev) put(pi.w, -1.f);
  }
}

// A tile's epilogue, 8 threads per pair (et: this thread's index among nth): dE/dc_d summed over
// the heads, each directed slot's ev gradient (24), slot scalars (20, staged in E) and force.
__device__ __forceinline__ void bwd_epilogue(const B7Params& p, const B7Stage& S, const B7Epi& E, int et, int nth, int ch) {
  // one channel per CTA: its copy of the forces and ev gradients; the angular force is the
  // degree channel's
  float* FV = p.FV + (NCH == 1 ? (size_t)ch * p.nslot * FVS : 0);
  float* GEV = p.GEV + (NCH == 1 ? (size_t)ch * p.nslot * M : 0);
  const bool angular = NCH == 2 || ch == 1;
#pragma unroll 1
  for (int r0 = 0; r0 < (B_EPI ? TP : 0); r0 += nth / 8) {
    const int row = r0 + (et >> 3), part = et & 7;
    if (row >= TP) break;
    float gcd[NH];
#pragma unroll
    for (int d = 0; d < NH; ++d) {
      gcd[d] = 0.f;
#pragma unroll
      for (int w = 0; w < NCW; ++w) gcd[d] += E.GCD[w][row][d];
    }
    float grr = 0.f, gce = 0.f;
#pragma unroll
    for (int w = 0; w < NCW; ++w) { grr += E.GRR[row][w]; gce += E.GCE[row][w]; }
    const int4 pi = S.pi[row];
    const float4 g0 = S.g0[row], g1 = S.g1[row];
    const bool valid = pi.z >= 0;
    if (valid) {
      // this CTA's heads' slot scalars: in each of the 5 groups of NH, its HC as 16-byte stores
      // (one channel per CTA: the heads channel's groups 0-2, the degree channel's 3-4)
      constexpr int HQ = HC / 4;
#pragma unroll
      for (int v = part; v < 2 * 5 * HQ; v += 8) {
        const int side = v / (5 * HQ), c = (v / HQ) % 5, q = v % HQ;
        if (NCH == 1 && (ch == 0 ? c >= 3 : c < 3)) continue;
        if (side && pi.w >= p.nslot) continue;                       // shadow slot
        const int o = c * NH + 4 * q;
        const float4 x = *reinterpret_cast<const float4*>(side ? &E.SCB[row][o] : &E.SCA[row][o]);
        *reinterpret_cast<float4*>(p.SC + (size_t)(side ? pi.w : pi.z) * NSC + o) = x;
      }
    }
    pair_finish(p, FV, GEV, angular, pi, g0, g1, gcd, grr, gce, &S.sf[row][0][0], part, valid);
  }
}

constexpr int BVW = R + 4;    // V transposed, fp32: [FC n][R + 4] per channel (ldmatrix)
constexpr int BV_FLOATS = NCH * FC * BVW;
constexpr int B7_SMEM = (BV_FLOATS + NCH * SH * BSW) * sizeof(float) + NST * sizeof(B7Stage) + sizeof(B7Epi);
#ifdef SMEM
// build.py sized the tiles by its count of the shared memory (with WA's): the kernel's own
static_assert(B7_SMEM + sizeof(FilterW1) == SMEM, "build.bwd_smem differs from the kernel's");
#endif
__global__ void __launch_bounds__(NTH, CTAS) bwd7(B7Params p) {
  extern __shared__ float smf[];
  float* Bv = smf;                                                     // [NCH][R][BVW]: V rows (uint2 hi/lo, or fp32)
  float* Bw = smf + BV_FLOATS;                                         // [NCH][SH][BSW]: wb rows
  B7Stage* st = reinterpret_cast<B7Stage*>(Bw + NCH * SH * BSW);       // [NST]
  B7Epi* Es = reinterpret_cast<B7Epi*>(st + NST);           // one: the consumers run each tile's epilogue
  __shared__ FilterW1 WA;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int ch0 = NCH == 1 ? (int)blockIdx.y : 0;                      // the CTA's channel (CHSPLIT)
  for (int i = tid; i < NCH * KD * FC; i += NTH) {
    const int cl = i / (KD * FC), ch = ch0 + cl, k = (i / FC) % KD, col = i % FC;
    if (k < SH) Bw[(cl * SH + k) * BSW + col] = (ch ? p.wb2 : p.wb1)[k * F + col];
    else {
      const float v = (ch ? p.vr2 : p.vr1)[(k - SH) * F + col];
      Bv[(cl * FC + col) * BVW + (k - SH)] = v;                         // transposed
    }
  }
  for (int i = tid; i < 2 * (NH + 1) * SH; i += NTH) {
    int ch = i / ((NH + 1) * SH), r = (i / SH) % (NH + 1), k = i % SH;
    WA[ch][r][k] = r < NH ? (ch ? p.wa2 : p.wa1)[r * SH + k] : (ch ? p.ba2 : p.ba1)[k];
  }
  __syncthreads();
  const int P = min(p.off[p.n], p.pmax);      // the records' room: more pairs only while a row overflows
  const int tiles = (P + TP - 1) / TP;
  const float inv_nc = p.scal[0], spacing = p.scal[1], n_int = p.scal[2];

  if (warp < NPW) {
    // =============================== producers ======================================
    // CHSPLIT: all 8 threads of a pair on the CTA's channel (4 hidden units, R/8 coefficients each)
    const int pp = tid >> 3, sub = tid & 7, ch = NCH == 1 ? ch0 : (tid >> 2) & 1, qt = NCH == 1 ? sub : tid & 3;
    const int cl = NCH == 1 ? 0 : ch;
    const float* uv = ch ? p.u2v : p.u1v;
    const float* ud = ch ? p.u2d : p.u1d;
    for (int i = 0, t = tile_of(0); t < tiles; ++i, t = tile_of(i)) {
      const int s = i % NST;
      if (i >= NST) bar_sync(1 + NST + s, NTH);
      B7Stage& S = st[s];
#pragma unroll 1
      for (int rd = 0; rd < MT; ++rd) {
        const int row = rd * 16 + pp, q = t * TP + row;
        const bool valid = q < P;
        const int4 pi = valid ? p.PI[q] : make_int4(0, 0, -1, -1);
        const float4 g0 = valid ? p.PG[2 * q] : make_float4(1.f, 0.f, 0.f, 1.f);
        const float4 g1 = valid ? p.PG[2 * q + 1] : make_float4(0.f, 0.f, 1.f, 0.f);
#pragma unroll
        for (int v = sub; v < 2 * (2 * NH / 4); v += 8) {               // both slots' 2 NH scores, 16 B each
          const int side = v / (2 * NH / 4), c = v % (2 * NH / 4);
          const int slot = side ? pi.w : pi.z;
          *reinterpret_cast<float4*>(&S.sf[row][side][4 * c]) =
              slot >= 0 ? *reinterpret_cast<const float4*>(p.Sf + (size_t)slot * 2 * NH + 4 * c) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#ifdef B_NOPROD
        if (sub == 0) { S.pi[row] = pi; S.g0[row] = g0; S.g1[row] = g1; }
        continue;
#endif
        // MC ev components each: |dev|^2 per degree entry, Y . dE/dev both ways
        float cc[NH], ga[NH], gr[NH];
#pragma unroll
        for (int d = 0; d < NH; ++d) { cc[d] = 0.f; ga[d] = 0.f; gr[d] = 0.f; }
        int yi[MC], ei[MC];                     // this thread's components: their harmonics and entries
#pragma unroll
        for (int k = 0; k < MC; ++k) { yi[k] = yidx(sub * MC + k); ei[k] = entry_of(sub * MC + k); }
        float yk[MC];                           // their harmonics, from one pass over all 24
#pragma unroll
        for (int k = 0; k < MC; ++k) yk[k] = 0.f;
        {
#pragma unroll
          for (int j = 0; j < 24; ++j) {
            const float yj = ylm(j, g0.x, g0.y, g0.z);
#pragma unroll
            for (int k = 0; k < MC; ++k) yk[k] = (yi[k] == j) ? yj : yk[k];
          }
        }
#pragma unroll
        for (int k = 0; k < MC; ++k) {
          const int m = sub * MC + k, en = ei[k];
          const float df = valid ? p.ev[(size_t)pi.y * M + m] - p.ev[(size_t)pi.x * M + m] : 0.f;
          const float y = yk[k];
          const bool need = valid && (NCH == 2 || ch == 1);
          const float gda = need ? p.gdev[(size_t)pi.x * M + m] : 0.f, gdb = need ? p.gdev[(size_t)pi.y * M + m] : 0.f;
          const float gA = gda * y;
          const float gB = gdb * y * p.par[en];
#pragma unroll
          for (int d = 0; d < NH; ++d) {
            cc[d] += (en == d) ? df * df : 0.f;
            ga[d] += (en == d) ? gA : 0.f;
            gr[d] += (en == d) ? gB : 0.f;
          }
        }
#pragma unroll
        for (int d = 0; d < NH; ++d) {
#pragma unroll
          for (int o = 1; o <= 4; o <<= 1) {
            cc[d] += __shfl_xor_sync(0xffffffff, cc[d], o);
            ga[d] += __shfl_xor_sync(0xffffffff, ga[d], o);
            gr[d] += __shfl_xor_sync(0xffffffff, gr[d], o);
          }
          cc[d] *= p.cdeg[d];
        }
        if (NCH == 2 || ch == 0) {
          // heads channel: dE/dx_a . v_b and dE/dx_b . v_a per head, FPT features per thread
          constexpr int FPT = FC / 8, TPH = 32 / FPT;             // threads per head: 2 or 1
          float da = 0.f, db = 0.f;
          if (valid) {
            const float4* gxa = reinterpret_cast<const float4*>(p.gdx + (size_t)pi.x * F + sub * FPT);
            const float4* gxb = reinterpret_cast<const float4*>(p.gdx + (size_t)pi.y * F + sub * FPT);
            const float4* va = reinterpret_cast<const float4*>(p.v1 + (size_t)pi.x * F + sub * FPT);
            const float4* vb = reinterpret_cast<const float4*>(p.v1 + (size_t)pi.y * F + sub * FPT);
#pragma unroll
            for (int j = 0; j < FPT / 4; ++j) {
              const float4 x = gxa[j], y = vb[j], u = gxb[j], w = va[j];
              da += x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
              db += u.x * w.x + u.y * w.y + u.z * w.z + u.w * w.w;
            }
          }
          if (TPH == 2) {
            da += __shfl_xor_sync(0xffffffff, da, 1);
            db += __shfl_xor_sync(0xffffffff, db, 1);
          }
          if (sub % TPH == 0) { S.ga1[row][sub / TPH] = da; S.gar1[row][sub / TPH] = db; }
        }
        if (sub == 0) {
          S.pi[row] = pi; S.g0[row] = g0; S.g1[row] = g1;
#pragma unroll
          for (int d = 0; d < NH; ++d) { S.ga2[row][d] = ga[d]; S.gar2[row][d] = gr[d]; }
        }
        // (channel, part): silu' for SH / NQ hidden units, dU/drt for R / NQ coefficients
        constexpr int NQ = NCH == 1 ? 8 : 4, HU = SH / NQ, RC = R / NQ;
#pragma unroll
        for (int j = 0; j < HU; ++j) {
          const int k = qt * HU + j;
          float zz = WA[ch][NH][k];
#pragma unroll
          for (int d = 0; d < NH; ++d) zz += cc[d] * WA[ch][d][k];
          const float sg = 1.f / (1.f + __expf(-zz));
          S.DZ[cl][row][k] = sg * (1.f + zz * (1.f - sg));
        }
        const float tc = g1.w;
        const float fi = fminf(fmaxf(floorf(tc), 0.f), n_int - 1.f);
        const float uu = tc - fi, u2 = uu * uu;
        const float c0 = (6.f * u2 - 6.f * uu) / spacing, c1 = 3.f * u2 - 4.f * uu + 1.f, c2 = 3.f * u2 - 2.f * uu;
        const size_t o0 = (size_t)fi * R + qt * RC, o1 = o0 + R;
#pragma unroll
        for (int j = 0; j < RC / 2; ++j) {                 // float2 loads: RC = 6 or 12 coefficients
          const float2 f0 = __ldg(reinterpret_cast<const float2*>(uv + o0) + j);
          const float2 f1 = __ldg(reinterpret_cast<const float2*>(uv + o1) + j);
          const float2 d0 = __ldg(reinterpret_cast<const float2*>(ud + o0) + j);
          const float2 d1 = __ldg(reinterpret_cast<const float2*>(ud + o1) + j);
          float* dst = &S.DU[cl][row][qt * RC + 2 * j];
          dst[0] = c0 * (f0.x - f1.x) + c1 * d0.x + c2 * d1.x;
          dst[1] = c0 * (f0.y - f1.y) + c1 * d0.y + c2 * d1.y;
        }
      }
      bar_arrive(1 + s, NTH);
    }
  } else {
    // =============================== consumers ======================================
    const int cw = warp - NPW, ch = NCH == 1 ? ch0 : cw / HC, h = cw % HC, ctid = tid - NPW * 32;
    const int cl = NCH == 1 ? 0 : ch;
    const int g = lane >> 2, tig = lane & 3;
    const float* Bvc = Bv + cl * FC * BVW;
    const float* Bwc = Bw + cl * SH * BSW;
    const float* q = ch ? p.q2 : p.q1;
    const float* kk = ch ? p.k2 : p.k1;
    const int src0 = (lane & ~3) | (tig >> 1), src1 = (lane & ~3) | (2 + (tig >> 1));
    for (int i = 0, t = tile_of(0); t < tiles; ++i, t = tile_of(i)) {
      const int s = i % NST;
      bar_sync(1 + s, NTH);
      const B7Stage& S = st[s];
      B7Epi& E = Es[0];
#pragma unroll 1
      for (int mt = 0; mt < MT; ++mt) {
        const int r0 = mt * 16 + g, r1 = r0 + 8;
        const int4 pi0 = S.pi[r0], pi1 = S.pi[r1];
        const float ce0 = S.g1[r0].x, ce1 = S.g1[r1].x;
        const size_t qq0 = (size_t)t * TP + r0, qq1 = (size_t)t * TP + r1;
        // q, k of both atoms for pass 2, issued first: in flight during pass 1 and dU @ V
        float2 QK[4][2][4];
#pragma unroll
        for (int ntl = 0; ntl < 4; ++ntl) {
          const int col = h * 32 + ntl * 8 + 2 * tig;
#pragma unroll
          for (int rw = 0; rw < 2; ++rw) {
            const int4 pi = rw ? pi1 : pi0;
            QK[ntl][rw][0] = LQB(q + (size_t)pi.x * F + col);
            QK[ntl][rw][1] = LQB(kk + (size_t)pi.y * F + col);
            QK[ntl][rw][2] = LQB(q + (size_t)pi.y * F + col);
            QK[ntl][rw][3] = LQB(kk + (size_t)pi.x * F + col);
          }
        }
        // the forward's scores (S = ce s / avg_nc per directed slot) give s and s' without w
        float s0 = 0.f, s1 = 0.f, sr0 = 0.f, sr1 = 0.f;
        if (pi0.z >= 0 && ce0 != 0.f) {
          const float r = 1.f / (ce0 * inv_nc);
          s0 = S.sf[r0][0][ch * NH + h] * r; sr0 = S.sf[r0][1][ch * NH + h] * r;
        }
        if (pi1.z >= 0 && ce1 != 0.f) {
          const float r = 1.f / (ce1 * inv_nc);
          s1 = S.sf[r1][0][ch * NH + h] * r; sr1 = S.sf[r1][1][ch * NH + h] * r;
        }
        float ga0, ga1, gb0, gb1;
        if (ch == 1) {
          ga0 = S.ga2[r0][h]; ga1 = S.ga2[r1][h]; gb0 = S.gar2[r0][h]; gb1 = S.gar2[r1][h];
        } else {
          ga0 = S.ga1[r0][h]; ga1 = S.ga1[r1][h]; gb0 = S.gar1[r0][h]; gb1 = S.gar1[r1][h];
        }
        const float gs0 = ga0 * ce0 * inv_nc, gs1 = ga1 * ce1 * inv_nc;
        const float gr0 = gb0 * ce0 * inv_nc, gr1 = gb1 * ce1 * inv_nc;
        if (tig == 0) {
#pragma unroll
          for (int rw = 0; rw < 2; ++rw) {
            const int4 pi = rw ? pi1 : pi0;
            const int row = rw ? r1 : r0;
            const float gs = rw ? gs1 : gs0, gr = rw ? gr1 : gr0, ce = rw ? ce1 : ce0;
            const float sv = rw ? s1 : s0, svr = rw ? sr1 : sr0, ga = rw ? ga1 : ga0, gb = rw ? gb1 : gb0;
            if (pi.z >= 0) {
              float* A_ = &E.SCA[row][0];          // written out per slot (80 bytes) in the epilogue
              float* B_ = &E.SCB[row][0];
              if (ch == 0) {
                A_[h] = gs; A_[NH + h] = gr; A_[2 * NH + h] = ce * svr * inv_nc;
                B_[h] = gr; B_[NH + h] = gs; B_[2 * NH + h] = ce * sv * inv_nc;
              } else {
                A_[3 * NH + h] = gs; A_[4 * NH + h] = gr;
                B_[3 * NH + h] = gr; B_[4 * NH + h] = gs;
              }
            }
            E.GCE[row][cw] = pi.z >= 0 ? (ga * sv + gb * svr) * inv_nc : 0.f;
          }
        }
        // dw/drt for this head's 32 columns: DU[16 x R] @ V[R x 32] (TF32 x3), then sum gw * dw
        float dcf[4][4];
#pragma unroll
        for (int ntl = 0; ntl < 4; ++ntl)
#pragma unroll
          for (int j = 0; j < 4; ++j) dcf[ntl][j] = 0.f;
#pragma unroll 2
        for (int kt = 0; kt < (B_MMA ? R / 8 : 0); ++kt) {
          uint32_t ahi[4], alo[4];
          {
            uint32_t araw[4];
            ldm_a(araw, &S.DU[cl][mt * 16][kt * 8], R + 4, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const float v = __uint_as_float(araw[j]);
              ahi[j] = tf32(v); alo[j] = tlo(v - __uint_as_float(ahi[j]));
            }
          }
          uint32_t braw[2][2][2];
#pragma unroll
          for (int half = 0; half < 2; ++half) ldm_b2(braw[half], Bvc + (h * 32 + half * 16) * BVW + kt * 8, BVW, lane);
#pragma unroll
          for (int ntl = 0; ntl < 4; ++ntl) {
            uint2 b0, b1;
            {
              const float v0 = __uint_as_float(braw[ntl >> 1][ntl & 1][0]), v1 = __uint_as_float(braw[ntl >> 1][ntl & 1][1]);
              b0 = make_uint2(tf32(v0), 0u); b1 = make_uint2(tf32(v1), 0u);
              b0.y = tlo(v0 - __uint_as_float(b0.x)); b1.y = tlo(v1 - __uint_as_float(b1.x));
            }
            mma(dcf[ntl], alo, b0.x, b1.x);
            mma(dcf[ntl], ahi, b0.y, b1.y);
            mma(dcf[ntl], ahi, b0.x, b1.x);
          }
        }
        // pass 2: dE/dw (accumulator layout)
        float GWr[4][2][2];
#pragma unroll
        for (int ntl = 0; ntl < 4; ++ntl) {
          const int col = h * 32 + ntl * 8 + 2 * tig;
#pragma unroll
          for (int rw = 0; rw < 2; ++rw) {
            const int4 pi = rw ? pi1 : pi0;
            const float gs = rw ? gs1 : gs0, gr = rw ? gr1 : gr0;
            const float2 qa = QK[ntl][rw][0], kb = QK[ntl][rw][1], qb = QK[ntl][rw][2], ka = QK[ntl][rw][3];
            GWr[ntl][rw][0] = gs * qa.x * kb.x + gr * qb.x * ka.x;
            GWr[ntl][rw][1] = gs * qa.y * kb.y + gr * qb.y * ka.y;
          }
        }

        float gq0 = 0.f, gq1 = 0.f;
#pragma unroll
        for (int ntl = 0; ntl < 4; ++ntl) {
          gq0 += GWr[ntl][0][0] * dcf[ntl][0] + GWr[ntl][0][1] * dcf[ntl][1];
          gq1 += GWr[ntl][1][0] * dcf[ntl][2] + GWr[ntl][1][1] * dcf[ntl][3];
        }
        gq0 += __shfl_xor_sync(0xffffffff, gq0, 1); gq0 += __shfl_xor_sync(0xffffffff, gq0, 2);
        gq1 += __shfl_xor_sync(0xffffffff, gq1, 1); gq1 += __shfl_xor_sync(0xffffffff, gq1, 2);
        if (tig == 0) {
          E.GRR[r0][cw] = pi0.z >= 0 ? gq0 : 0.f;
          E.GRR[r1][cw] = pi1.z >= 0 ? gq1 : 0.f;
        }
        // dE/dhz (this head): dE/dw[16 x 32] @ wb[:, head cols]^T, A fragments by shuffles, the
        // hidden units 32 at a time; then this head's part of dE/dc_d = sum_k dE/dhz_k silu'(z_k) wa[d][k]
        float gc0[NH], gc1[NH];
#pragma unroll
        for (int d = 0; d < NH; ++d) { gc0[d] = 0.f; gc1[d] = 0.f; }
#pragma unroll 1
        for (int nh0 = 0; nh0 < SH; nh0 += 32) {
        float gz[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
          for (int j = 0; j < 4; ++j) gz[nt][j] = 0.f;
#pragma unroll
        for (int kt = 0; kt < (B_MMA ? 4 : 0); ++kt) {
          const float x00 = __shfl_sync(0xffffffff, GWr[kt][0][0], src0), x01 = __shfl_sync(0xffffffff, GWr[kt][0][1], src0);
          const float x10 = __shfl_sync(0xffffffff, GWr[kt][1][0], src0), x11 = __shfl_sync(0xffffffff, GWr[kt][1][1], src0);
          const float y00 = __shfl_sync(0xffffffff, GWr[kt][0][0], src1), y01 = __shfl_sync(0xffffffff, GWr[kt][0][1], src1);
          const float y10 = __shfl_sync(0xffffffff, GWr[kt][1][0], src1), y11 = __shfl_sync(0xffffffff, GWr[kt][1][1], src1);
          const bool odd = tig & 1;
          const float av[4] = {odd ? x01 : x00, odd ? x11 : x10, odd ? y01 : y00, odd ? y11 : y10};
          uint32_t ahi[4], alo[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) { ahi[j] = tf32(av[j]); alo[j] = tlo(av[j] - __uint_as_float(ahi[j])); }
          uint32_t wraw[2][2][2];
#pragma unroll
          for (int half = 0; half < 2; ++half) ldm_b2(wraw[half], Bwc + (nh0 + half * 16) * BSW + h * 32 + kt * 8, BSW, lane);
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            // hidden unit n = nh0 + nt * 8 + g: B[kk][n] = wb[n][h * 32 + kk]
            const float v0 = __uint_as_float(wraw[nt >> 1][nt & 1][0]), v1 = __uint_as_float(wraw[nt >> 1][nt & 1][1]);
            const uint32_t b0h = tf32(v0), b1h = tf32(v1);
            const uint32_t b0l = tlo(v0 - __uint_as_float(b0h)), b1l = tlo(v1 - __uint_as_float(b1h));
            mma(gz[nt], alo, b0h, b1h);
            mma(gz[nt], ahi, b0l, b1l);
            mma(gz[nt], ahi, b0h, b1h);
          }
        }
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int k = nh0 + nt * 8 + 2 * tig;
          const float2 z0 = *reinterpret_cast<const float2*>(&S.DZ[cl][r0][k]);
          const float2 z1 = *reinterpret_cast<const float2*>(&S.DZ[cl][r1][k]);
          const float v00 = gz[nt][0] * z0.x, v01 = gz[nt][1] * z0.y, v10 = gz[nt][2] * z1.x, v11 = gz[nt][3] * z1.y;
#pragma unroll
          for (int d = 0; d < NH; ++d) {
            gc0[d] += v00 * WA[ch][d][k] + v01 * WA[ch][d][k + 1];
            gc1[d] += v10 * WA[ch][d][k] + v11 * WA[ch][d][k + 1];
          }
        }
        }
#pragma unroll
        for (int d = 0; d < NH; ++d) {
          gc0[d] += __shfl_xor_sync(0xffffffff, gc0[d], 1); gc0[d] += __shfl_xor_sync(0xffffffff, gc0[d], 2);
          gc1[d] += __shfl_xor_sync(0xffffffff, gc1[d], 1); gc1[d] += __shfl_xor_sync(0xffffffff, gc1[d], 2);
        }
        if (tig == 0) {
#pragma unroll
          for (int d = 0; d < NH; d += 4) {
            *reinterpret_cast<float4*>(&E.GCD[cw][r0][d]) = make_float4(gc0[d], gc0[d + 1], gc0[d + 2], gc0[d + 3]);
            *reinterpret_cast<float4*>(&E.GCD[cw][r1][d]) = make_float4(gc1[d], gc1[d + 1], gc1[d + 2], gc1[d + 3]);
          }
        }
      }
      bar_sync(6, NCW * 32);                 // all heads' partials of this tile in place
      bwd_epilogue(p, S, E, ctid, NCW * 32, ch0);
      bar_sync(6, NCW * 32);                                  // epilogue buffers free for the next tile
      if (tile_of(i + NST) < tiles) bar_arrive(1 + NST + s, NTH);
    }
  }
}

struct GatherParams {
  const float *__restrict__ q1, *__restrict__ k1, *__restrict__ q2, *__restrict__ k2, *__restrict__ gdx,
      *__restrict__ Wst, *__restrict__ SC, *__restrict__ FV, *__restrict__ GEV, *__restrict__ gev_in, *__restrict__ f_in;
  const int *__restrict__ nbr, *__restrict__ cnt, *__restrict__ PIDX, *__restrict__ row0;
  float *__restrict__ gq1, *__restrict__ gk1, *__restrict__ gv1, *__restrict__ gq2, *__restrict__ gk2,
      *__restrict__ gev_out, *__restrict__ f_out;   // f: [6][n] (fx, fy, fz, wx, wy, wz)
  int n, K, nl;
  int local;                     // the projections' gradients for the range's rows only ([nl, F], from row0)
};

constexpr int GW_ = 8;
#ifndef GSPLIT
#define GSPLIT 2
#endif
constexpr int HSPLIT = NH / 4;    // warps per slot, each HPW heads (the wider model: two)
constexpr int HPW = NH / HSPLIT;
constexpr int WPA = GSPLIT * HSPLIT;   // warps per atom
static_assert(GW_ % WPA == 0, "whole atoms per block");
// WPA warps per atom: GSPLIT over the neighbours (each every GSPLIT-th), HSPLIT over the heads;
// the neighbour parts' sums combined in a fixed order in shared memory (deterministic)
__global__ void __launch_bounds__(GW_ * 32) bwd_gather(GatherParams p) {
  constexpr int MS = (M + 31) / 32;                        // 32-lane chunks of the ev gradient
  __shared__ float red[GW_][5][HPW][32];
  __shared__ float reds[GW_][MS + 1][32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int part = warp % GSPLIT, hw = (warp / GSPLIT) % HSPLIT;
  const int r0 = p.row0[0];
  const int a = r0 + (blockIdx.x * GW_ + warp) / WPA;         // the range's rows, written from row0[1] on
  const bool live = a >= p.row0[1] && a < r0 + p.nl && a < p.n;
  const int n_a = live ? min(p.cnt[a], p.K) : 0;
  if (p.local && part == 0 && a < p.row0[1]) {
    // the range-local projections' gradients of a row before row0[1] (a chunk shifted back:
    // the previous chunk's rows) zeros, not left unwritten
#pragma unroll
    for (int j = 0; j < HPW; ++j) {
      const size_t o = (size_t)(a - r0) * F + (hw * HPW + j) * 32 + lane;
      p.gq1[o] = 0.f; p.gk1[o] = 0.f; p.gv1[o] = 0.f; p.gq2[o] = 0.f; p.gk2[o] = 0.f;
    }
  }
  float acc[5][HPW], sv[MS], fv = 0.f;
#pragma unroll
  for (int i = 0; i < 5; ++i)
#pragma unroll
    for (int j = 0; j < HPW; ++j) acc[i][j] = 0.f;
#pragma unroll
  for (int ms = 0; ms < MS; ++ms) sv[ms] = 0.f;
  const size_t nrow = (size_t)a * p.K, srow = (size_t)(a - r0) * p.K;   // the neighbour row, the range's slots
#pragma unroll 1
  for (int c = part; c < n_a; c += GSPLIT) {
    const size_t slot = srow + c;
    const int b = p.nbr[nrow + c], pq = p.PIDX[slot];
    const float* sc = p.SC + slot * NSC;
    const float* w1 = p.Wst + (size_t)pq * 2 * F;
    const float* w2 = w1 + F;
#pragma unroll
    for (int j = 0; j < HPW; ++j) {
      const int h = hw * HPW + j, f = h * 32 + lane;
      const size_t ob = (size_t)b * F + f;
      const float W1 = w1[f], W2 = w2[f];
      acc[0][j] += sc[h] * W1 * p.k1[ob];
      acc[1][j] += sc[NH + h] * W1 * p.q1[ob];
      acc[2][j] += sc[2 * NH + h] * p.gdx[ob];
      acc[3][j] += sc[3 * NH + h] * W2 * p.k2[ob];
      acc[4][j] += sc[4 * NH + h] * W2 * p.q2[ob];
    }
    if (hw == 0) {
      // this slot's ev gradient and force, in the same iteration (one channel per CTA in the
      // backward: both channels' copies)
#pragma unroll
      for (int ms = 0; ms < MS; ++ms) {
        const int m = ms * 32 + lane;
        sv[ms] += m < M ? p.GEV[slot * M + m] : 0.f;
      }
      fv += lane < NFW ? p.FV[slot * FVS + lane] : 0.f;
      if (NCH == 1) {
        const size_t s2 = (size_t)p.nl * p.K + slot;                // the second channel's copy
#pragma unroll
        for (int ms = 0; ms < MS; ++ms) {
          const int m = ms * 32 + lane;
          sv[ms] += m < M ? p.GEV[s2 * M + m] : 0.f;
        }
        fv += lane < NFW ? p.FV[s2 * FVS + lane] : 0.f;
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 5; ++i)
#pragma unroll
    for (int j = 0; j < HPW; ++j) red[warp][i][j][lane] = acc[i][j];
#pragma unroll
  for (int ms = 0; ms < MS; ++ms) reds[warp][ms][lane] = sv[ms];
  reds[warp][MS][lane] = fv;
  __syncthreads();
  if (part != 0 || !live) return;
#pragma unroll
  for (int k = 1; k < GSPLIT; ++k) {
#pragma unroll
    for (int i = 0; i < 5; ++i)
#pragma unroll
      for (int j = 0; j < HPW; ++j) acc[i][j] += red[warp + k][i][j][lane];
#pragma unroll
    for (int ms = 0; ms < MS; ++ms) sv[ms] += reds[warp + k][ms][lane];
    fv += reds[warp + k][MS][lane];
  }
#pragma unroll
  for (int j = 0; j < HPW; ++j) {
    const size_t o = (size_t)(p.local ? a - r0 : a) * F + (hw * HPW + j) * 32 + lane;
    p.gq1[o] = acc[0][j]; p.gk1[o] = acc[1][j]; p.gv1[o] = acc[2][j]; p.gq2[o] = acc[3][j]; p.gk2[o] = acc[4][j];
  }
  if (hw != 0) return;
#pragma unroll
  for (int ms = 0; ms < MS; ++ms) {
    const int m = ms * 32 + lane;
    if (m < M) p.gev_out[(size_t)a * M + m] = p.gev_in[(size_t)a * M + m] + sv[ms];
  }
  if (lane < NFW) p.f_out[(size_t)lane * p.n + a] = p.f_in[(size_t)lane * p.n + a] + fv;
}

static ffi::Error B7Impl(cudaStream_t stream, F32 q1, F32 k1, F32 v1, F32 q2, F32 k2, F32 ev, F32 gdx, F32 gdev,
                         S32 PI, F32 PG, S32 off, F32 u1v, F32 u1d, F32 u2v, F32 u2d, F32 wa1, F32 ba1, F32 wb1,
                         F32 wa2, F32 ba2, F32 wb2, F32 vr1, F32 vr2, F32 cdeg, F32 scal, F32 par, F32 Wst,
                         F32 Sf, RF32 SC, RF32 FV, RF32 GEV, int64_t P) {
  B7Params p;
  p.q1 = q1.typed_data(); p.k1 = k1.typed_data(); p.v1 = v1.typed_data(); p.q2 = q2.typed_data();
  p.k2 = k2.typed_data(); p.ev = ev.typed_data(); p.gdx = gdx.typed_data(); p.gdev = gdev.typed_data();
  p.PI = reinterpret_cast<const int4*>(PI.typed_data()); p.PG = reinterpret_cast<const float4*>(PG.typed_data());
  p.off = off.typed_data();
  p.u1v = u1v.typed_data(); p.u1d = u1d.typed_data(); p.u2v = u2v.typed_data(); p.u2d = u2d.typed_data();
  p.wa1 = wa1.typed_data(); p.ba1 = ba1.typed_data(); p.wb1 = wb1.typed_data();
  p.wa2 = wa2.typed_data(); p.ba2 = ba2.typed_data(); p.wb2 = wb2.typed_data();
  p.vr1 = vr1.typed_data(); p.vr2 = vr2.typed_data(); p.cdeg = cdeg.typed_data(); p.scal = scal.typed_data();
  p.par = par.typed_data(); p.Wst = Wst.typed_data(); p.Sf = Sf.typed_data();
  p.SC = SC->typed_data(); p.FV = FV->typed_data(); p.GEV = GEV->typed_data();
  p.n = (int)off.dimensions()[0] - 1;
  p.nslot = (int)(SC->dimensions()[0] / NSC); p.pmax = (int)P;
  set_max_smem<bwd7>(B7_SMEM);
  bwd7<<<dim3(num_sms(), NCH == 1 ? 2 : 1), NTH, B7_SMEM, stream>>>(p);
  return launch_status();
}

// local: the projections' gradients are the range's rows only ([nl, F] outputs, row r0 first)
static ffi::Error GatherImpl(cudaStream_t stream, F32 q1, F32 k1, F32 q2, F32 k2, F32 gdx, F32 Wst, F32 SC, F32 FV,
                             F32 GEV, F32 gev_in, F32 f_in, S32 nbr, S32 cnt, S32 PIDX, S32 row0, RF32 gq1, RF32 gk1,
                             RF32 gv1, RF32 gq2, RF32 gk2, RF32 gev_out, RF32 f_out, int64_t NL, int64_t local) {
  GatherParams p;
  p.q1 = q1.typed_data(); p.k1 = k1.typed_data(); p.q2 = q2.typed_data(); p.k2 = k2.typed_data();
  p.gdx = gdx.typed_data(); p.Wst = Wst.typed_data(); p.SC = SC.typed_data(); p.FV = FV.typed_data();
  p.GEV = GEV.typed_data(); p.gev_in = gev_in.typed_data(); p.f_in = f_in.typed_data();
  p.nbr = nbr.typed_data(); p.cnt = cnt.typed_data(); p.PIDX = PIDX.typed_data();
  p.gq1 = gq1->typed_data(); p.gk1 = gk1->typed_data(); p.gv1 = gv1->typed_data(); p.gq2 = gq2->typed_data();
  p.gk2 = gk2->typed_data(); p.gev_out = gev_out->typed_data(); p.f_out = f_out->typed_data();
  auto d = nbr.dimensions();
  p.n = (int)d[0]; p.K = (int)d[1]; p.nl = (int)NL; p.row0 = row0.typed_data(); p.local = (int)local;
  bwd_gather<<<(p.nl * WPA + GW_ - 1) / GW_, GW_ * 32, 0, stream>>>(p);
  return launch_status();
}


XLA_FFI_DEFINE_HANDLER_SYMBOL(BwdPair7, B7Impl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>()  // q1 k1 v1 q2 k2 ev gdx gdev
        .Arg<S32>().Arg<F32>().Arg<S32>()                                                         // PI PG off
        .Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>()                                              // u1v u1d u2v u2d
        .Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>()                        // wa1 ba1 wb1 wa2 ba2 wb2
        .Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>()                        // vr1 vr2 cdeg scal par Wst
        .Arg<F32>()                                                                               // the forward's scores
        .Ret<F32>().Ret<F32>().Ret<F32>().Attr<int64_t>("P"));                                  // SC FV GEV

XLA_FFI_DEFINE_HANDLER_SYMBOL(BwdGather, GatherImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>()            // q1 k1 q2 k2 gdx
        .Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>()  // Wst SC FV GEV gev_in f_in
        .Arg<S32>().Arg<S32>().Arg<S32>().Arg<S32>()                         // nbr cnt PIDX row0
        .Ret<F32>().Ret<F32>().Ret<F32>().Ret<F32>().Ret<F32>().Ret<F32>().Ret<F32>().Attr<int64_t>("NL")
        .Attr<int64_t>("local"));
