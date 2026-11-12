// One layer's attention forward of SO3LR (so3lr-tauri-s/m/l, so3lr), pipelined, in two
// deterministic kernels.
//  att_pair: each unordered pair {a, b} once (edges packed 16 per tile across atoms): the
//    geometry, the spherical filters' hidden layers, the rank-R radial coefficients and the
//    filter [hz | coeffs] @ [wb ; V_R] (TF32 x3 tensor cores) once for both directions, and
//    both directed scores ce * (q . w . k) / avg_nc per head and degree entry, written to the
//    directed edges' slots S[row, slot, 8] (rows of the neighbour table sorted).
//  att_msg: per atom, gather only: x messages sum_b S v1_b, ev messages sum_b S Y(u_ab).
#include <cstdint>

#include "attn_common.cuh"
#include "ffi_common.cuh"

constexpr float FLOOR_W = 0.05f;          // the pair distance floor's width, as flash's _floor

// ===================================================================================== //
// pair prep (once per force call): pair q -> atoms a < b, their directed slots, geometry
// ===================================================================================== //
struct PrepParams {
  const double *px, *py, *pz, *box;
  const int *nbr, *cnt, *z, *off, *lo, *hi, *tile_atom, *row0;   // off, lo, hi, tile_atom: per row of the range
  const float *rmin, *scal;
  int4* PI;        // a, b, slot(a <- b), slot(b <- a) (a shadow slot >= nl K when b is outside the range)
  float4* PG;      // [2 per pair]: (ux, uy, uz, r), (ce, dce, drt, rt / spacing)
  int* PIDX;       // directed slot -> pair
  int n, K, zmax, nl, pmax;      // pmax: the records' room
  int KS;                        // the rows' periodic images (edge_atom), 0: the minimum image
};

// The pairs of a range of rows (a device's, or a chunk of them): for each of its rows a
// (row0 <= a < row0 + nl) the neighbours b outside the range, and the larger ones inside (a
// pair of two atoms of the range once). In the sorted row the kept columns are [0, lo)
// (b < row0) and [hi, cnt) (b > a). The directed slots and the per-row arrays are the
// range's own: slot (a - row0) K + column, so a range's buffers hold its rows only. A
// partner outside the range gets no slot: its reverse score goes to the shadow slot nl K + q.
// With images (KS) an entry is a partner and its image (edge_atom), compared as packed: an
// atom's own entry would be a C + C0 (C = (2 KS + 1)^3, C0 = (C - 1) / 2: no shift), the
// reverse of b's image s is a's image -s (code C - 1 - code), and a pair of an atom with its
// own image once (code above C0).
__global__ void pair_prep(PrepParams p) {
  const int P = min(p.off[p.nl], p.pmax);
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= P) return;
  const float spacing = p.scal[1], rc = p.scal[3];
  const int r0 = p.row0[0];
  int al = p.tile_atom[q >> 4];                      // the row within the range
  while (p.off[al + 1] <= q) ++al;
  const int a = r0 + al;
  const int j = q - p.off[al], la = p.lo[al], ha = p.hi[al];
  const int c = j < la ? j : j + (ha - la);
  const int v = p.nbr[(size_t)a * p.K + c];
  const int b = edge_atom(v, p.KS);
  const int W = 2 * p.KS + 1, C = W * W * W;
  const int back = a * C + (C - 1 - (v - b * C));       // b's entry of a (KS 0: C 1, back a)
  const bool b_local = b >= r0 && b < r0 + p.nl;
  const int cm = b_local ? min(p.cnt[b], p.K) : 0;
  int lo = 0, hi = cm;
  const int* row = p.nbr + (size_t)b * p.K;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (row[mid] < back) lo = mid + 1; else hi = mid;
  }
  // b's slot of a: none when b is outside the range, or (a row overflowed; the engine widens
  // the rows and calls again) when b's truncated row lacks a: the reverse score goes to the shadow
  const bool found = lo < cm && row[lo] == back;
  const int sba = found ? (b - r0) * p.K + lo : p.nl * p.K + q;
  double dxd, dyd, dzd;
  edge_disp(p.px, p.py, p.pz, a, v, p.KS, p.box[0], p.box[1], p.box[2], dxd, dyd, dzd);
  float dx = (float)dxd, dy = (float)dyd, dz = (float)dzd;
  float r2 = dx * dx + dy * dy + dz * dz;
  bool ok = r2 > 0.f;
  float rr = ok ? sqrtf(r2) : 2.f * rc;
  float inv = 1.f / rr;
  float x = rr / rc;
  bool inside = rr < rc;
  float ce = inside ? 1.f - 6.f * x * x * x * x * x + 15.f * x * x * x * x - 10.f * x * x * x : 0.f;
  float dce = inside ? (-30.f * x * x * x * x + 60.f * x * x * x - 30.f * x * x) / rc : 0.f;
  int za = p.z[a], zb = p.z[b];
  int l0 = min(za, zb) - 1, h0 = max(za, zb) - 1;
  float d_off = p.rmin[l0 * p.zmax + h0] + FLOOR_W;
  float tt = __expf(-2.f * fmaxf((d_off - rr) / FLOOR_W, 0.f));
  float th = (1.f - tt) / (1.f + tt);
  bool above = rr >= d_off;
  float rt = above ? rr : d_off - FLOOR_W * th;
  float drt = above ? 1.f : 1.f - th * th;
  p.PI[q] = make_int4(a, b, al * p.K + c, sba);
  p.PIDX[al * p.K + c] = q;
  if (found) p.PIDX[(b - r0) * p.K + lo] = q;
  p.PG[2 * q] = make_float4(ok ? dx * inv : 1.f, ok ? dy * inv : 0.f, ok ? dz * inv : 0.f, rr);
  p.PG[2 * q + 1] = make_float4(ce, dce, drt, rt / spacing);
}

// ===================================================================================== //
// att_pair v6: producer warps fill a double-buffered stage per 16-pair tile ([hz | radial
// coefficients] for both channels), consumer warps (one per channel and head) run the
// filter matmul and the scores; named barriers hand stages over
// ===================================================================================== //
#ifndef TP
#define TP 64
#endif
constexpr int MT = TP / 16;                        // mma row tiles per tile
#ifndef NSTAGES
#define NSTAGES 2
#endif
// one channel per CTA (blockIdx.y): 4 producer + HC consumer warps; two CTAs per SM with 4 heads
// where the shared memory holds them (CTAS: the registers' bound)
#ifndef CTAS
#define CTAS (HC == 4 ? 2 : 1)
#endif
constexpr int NCW = HC;                              // consumer warps (one per head)
constexpr int NPW = 4, NTH = (NPW + NCW) * 32, NST = NSTAGES;
constexpr int BS_FLOATS = FC * (KD + 4);           // the weights, transposed: n rows, k columns

struct PairParams {
  const float *q1, *k1, *q2, *k2, *ev;
  const int4* PI;
  const float4* PG;
  const int* off;
  const float *u1v, *u1d, *u2v, *u2d;
  const float *wa1, *ba1, *wb1, *bb1, *wa2, *ba2, *wb2, *bb2, *vr1, *vr2, *cdeg, *scal;
  float* S;
  float* Wst;
  int n, pmax;
};

struct Stage {
  float A[TP][KD + 4];
  int4 pi[TP];
  float ce[TP];
  float Sv[TP][2][HC];   // the tile's scores per pair, both directed slots (consumers)
};


#define WST_STORE(ptr, v) __stcs(ptr, v)                    // streaming: w is read back only in the backward
#ifdef NOQK
#define LQ(ptr) make_float2(1.f, 1.f)
#else
#define LQ(ptr) (*reinterpret_cast<const float2*>(ptr))
#endif
// KEEPW: write the filters per pair for the backward (else the scores only)
template <bool KEEPW>
__global__ void __launch_bounds__(NTH, CTAS) att_pair(PairParams p) {
  extern __shared__ float smf[];
  float* Bs = smf;                                                   // [FC][KD + 4] fp32, transposed (n rows, k columns)
  Stage* st = reinterpret_cast<Stage*>(smf + BS_FLOATS);             // [NST]
  __shared__ FilterW1 WA;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int ch0 = (int)blockIdx.y;                                   // the CTA's channel
  for (int i = tid; i < KD * FC; i += NTH) {
    const int ch = ch0, k = i / FC, col = i % FC;
    const float v = k < SH ? (ch ? p.wb2 : p.wb1)[k * F + col]
                           : (ch ? p.vr2 : p.vr1)[(k - SH) * F + col];
    Bs[col * (KD + 4) + k] = v;                           // transposed: n rows, k columns
  }
  for (int i = tid; i < 2 * (NH + 1) * SH; i += NTH) {
    int ch = i / ((NH + 1) * SH), r = (i / SH) % (NH + 1), k = i % SH;
    WA[ch][r][k] = r < NH ? (ch ? p.wa2 : p.wa1)[r * SH + k] : (ch ? p.ba2 : p.ba1)[k];
  }
  __syncthreads();
  const int P = min(p.off[p.n], p.pmax);
  const int tiles = (P + TP - 1) / TP;
  const float inv_nc = p.scal[0], spacing = p.scal[1], n_int = p.scal[2];

  if (warp < NPW) {
    // ------------------------------- producers ------------------------------------
    // MT rounds of 16 pairs; in each, 8 threads per pair (3 ev components each, then
    // (channel, quarter): 8 hidden units and R/4 radial coefficients)
    // CHSPLIT: all 8 threads of a pair on the CTA's channel (4 hidden units, R/8 coefficients each)
    const int pp = tid >> 3, sub = tid & 7, ch = ch0, qt = sub;
    constexpr int NQ = 8, HU = SH / NQ, RC = R / NQ;
    const float* uv = ch ? p.u2v : p.u1v;
    const float* ud = ch ? p.u2d : p.u1d;
    for (int i = 0, t = tile_of(0); t < tiles; ++i, t = tile_of(i)) {
      const int s = i % NST;
      if (i >= NST) bar_sync(1 + NST + s, NTH);          // stage s consumed (tile i - NST)
      Stage& S = st[s];
#pragma unroll 1
      for (int rd = 0; rd < MT; ++rd) {
        const int row = rd * 16 + pp, q = t * TP + row;
        const bool valid = q < P;
#ifdef NOPROD
        if (sub == 0) { S.pi[row] = valid ? p.PI[q] : make_int4(0, 0, -1, -1); S.ce[row] = 1.f; }
        continue;
#endif
        const int4 pi = valid ? p.PI[q] : make_int4(0, 0, -1, -1);
        const float4 g2 = valid ? p.PG[2 * q + 1] : make_float4(0.f, 0.f, 1.f, 0.f);
        float ea[MC], eb[MC];
#pragma unroll
        for (int k = 0; k < MC; ++k) {
          ea[k] = p.ev[(size_t)pi.x * M + sub * MC + k];
          eb[k] = p.ev[(size_t)pi.y * M + sub * MC + k];
        }
        const float tc = g2.w;
        const float fi = fminf(fmaxf(floorf(tc), 0.f), n_int - 1.f);
        const size_t o0 = (size_t)fi * R + qt * RC, o1 = o0 + R;
        float2 f0[RC / 2], f1[RC / 2], d0[RC / 2], d1[RC / 2];       // float2 loads: RC = 6 or 12 coefficients
#pragma unroll
        for (int j = 0; j < RC / 2; ++j) {
          f0[j] = __ldg(reinterpret_cast<const float2*>(uv + o0) + j);
          f1[j] = __ldg(reinterpret_cast<const float2*>(uv + o1) + j);
          d0[j] = __ldg(reinterpret_cast<const float2*>(ud + o0) + j);
          d1[j] = __ldg(reinterpret_cast<const float2*>(ud + o1) + j);
        }
        float cc[NH];
#pragma unroll
        for (int d = 0; d < NH; ++d) cc[d] = 0.f;
#pragma unroll
        for (int k = 0; k < MC; ++k) {
          const int m = sub * MC + k;
          const float df = valid ? eb[k] - ea[k] : 0.f;
          const int en = entry_of(m);
#pragma unroll
          for (int d = 0; d < NH; ++d) cc[d] += (en == d) ? df * df : 0.f;
        }
#pragma unroll
        for (int d = 0; d < NH; ++d) {
          cc[d] += __shfl_xor_sync(0xffffffff, cc[d], 1);
          cc[d] += __shfl_xor_sync(0xffffffff, cc[d], 2);
          cc[d] += __shfl_xor_sync(0xffffffff, cc[d], 4);
          cc[d] *= p.cdeg[d];
        }
#pragma unroll
        for (int j = 0; j < HU; ++j) {
          const int k = qt * HU + j;
          float zz = WA[ch][NH][k];
#pragma unroll
          for (int d = 0; d < NH; ++d) zz += cc[d] * WA[ch][d][k];
          S.A[row][k] = __fdividef(zz, 1.f + __expf(-zz));
        }
        const float uu = tc - fi, u2 = uu * uu, u3 = u2 * uu;
        const float h00 = 2.f * u3 - 3.f * u2 + 1.f, h01 = 3.f * u2 - 2.f * u3;
        const float h10 = spacing * (u3 - 2.f * u2 + uu), h11 = spacing * (u3 - u2);
#pragma unroll
        for (int j = 0; j < RC / 2; ++j) {
          float* dst = &S.A[row][SH + qt * RC + 2 * j];
          dst[0] = h00 * f0[j].x + h01 * f1[j].x + h10 * d0[j].x + h11 * d1[j].x;
          dst[1] = h00 * f0[j].y + h01 * f1[j].y + h10 * d0[j].y + h11 * d1[j].y;
        }
        if (sub == 0) {
          S.pi[row] = pi;
          S.ce[row] = g2.x;
        }
      }
      bar_arrive(1 + s, NTH);                            // stage s full
    }
  } else {
    // ------------------------------- consumers ------------------------------------
    // warp (channel, head): its 32 filter columns for the tile's TP pairs, each weight
    // fragment loaded (and split) once for the MT row tiles
    const unsigned cwu = warp - NPW;                     // consumer warp: masks and shifts, not signed division
    const int h = cwu % HC, ch = ch0;
    const int g = lane >> 2, tig = lane & 3;
    const float* Bc = Bs;
    const float* bb = ch ? p.bb2 : p.bb1;
    const float* q = ch ? p.q2 : p.q1;
    const float* kk = ch ? p.k2 : p.k1;
    float2 bias[4];                                      // this warp's 32 bias columns, once
#pragma unroll
    for (int ntl = 0; ntl < 4; ++ntl) bias[ntl] = *reinterpret_cast<const float2*>(bb + h * 32 + ntl * 8 + 2 * tig);
    for (int i = 0, t = tile_of(0); t < tiles; ++i, t = tile_of(i)) {
      const int s = i % NST;
      bar_sync(1 + s, NTH);                              // stage s full
      Stage& S = st[s];
      float cf[MT][4][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int ntl = 0; ntl < 4; ++ntl)
#pragma unroll
          for (int j = 0; j < 4; ++j) cf[mt][ntl][j] = 0.f;
#pragma unroll 1
#ifdef NOMMA
      for (int kt = 0; kt < 0; ++kt) {
#else
      for (int kt = 0; kt < KD / 8; ++kt) {
#endif
        uint32_t bh[4][2], bl[4][2];
        {
          uint32_t braw[2][2][2];
#pragma unroll
          for (int half = 0; half < 2; ++half)                        // n-tiles (2 half, 2 half + 1)
            ldm_b2(braw[half], Bc + (h * 32 + half * 16) * (KD + 4) + kt * 8, KD + 4, lane);
#pragma unroll
          for (int ntl = 0; ntl < 4; ++ntl)
#pragma unroll
            for (int j = 0; j < 2; ++j) {
              const float v = __uint_as_float(braw[ntl >> 1][ntl & 1][j]);
              bh[ntl][j] = tf32(v);
              bl[ntl][j] = tlo(v - __uint_as_float(bh[ntl][j]));
            }
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const float* Ar = &S.A[mt * 16][0];
          uint32_t ahi[4], alo[4];
          uint32_t araw[4];
          ldm_a(araw, Ar + kt * 8, KD + 4, lane);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float v = __uint_as_float(araw[j]);
            ahi[j] = tf32(v);
            alo[j] = tlo(v - __uint_as_float(ahi[j]));
          }
#pragma unroll
          for (int ntl = 0; ntl < 4; ++ntl) {
            mma(cf[mt][ntl], alo, bh[ntl][0], bh[ntl][1]);
            mma(cf[mt][ntl], ahi, bl[ntl][0], bl[ntl][1]);
            mma(cf[mt][ntl], ahi, bh[ntl][0], bh[ntl][1]);
          }
        }
      }
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int r0 = mt * 16 + g, r1 = r0 + 8;
        const int4 pi0 = S.pi[r0], pi1 = S.pi[r1];
        const float ce0 = S.ce[r0], ce1 = S.ce[r1];
        float sab0 = 0.f, sab1 = 0.f, sba0 = 0.f, sba1 = 0.f;
#pragma unroll
        for (int ntl = 0; ntl < 4; ++ntl) {
          const int col = h * 32 + ntl * 8 + 2 * tig;
          const float2 bq = bias[ntl];
          const float w00 = cf[mt][ntl][0] + bq.x, w01 = cf[mt][ntl][1] + bq.y;
          const float w10 = cf[mt][ntl][2] + bq.x, w11 = cf[mt][ntl][3] + bq.y;
          const float2 qa0 = LQ(q + (size_t)pi0.x * F + col);
          const float2 kb0 = LQ(kk + (size_t)pi0.y * F + col);
          const float2 qb0 = LQ(q + (size_t)pi0.y * F + col);
          const float2 ka0 = LQ(kk + (size_t)pi0.x * F + col);
          const float2 qa1 = LQ(q + (size_t)pi1.x * F + col);
          const float2 kb1 = LQ(kk + (size_t)pi1.y * F + col);
          const float2 qb1 = LQ(q + (size_t)pi1.y * F + col);
          const float2 ka1 = LQ(kk + (size_t)pi1.x * F + col);
          sab0 += qa0.x * w00 * kb0.x + qa0.y * w01 * kb0.y;
          sba0 += qb0.x * w00 * ka0.x + qb0.y * w01 * ka0.y;
          sab1 += qa1.x * w10 * kb1.x + qa1.y * w11 * kb1.y;
          sba1 += qb1.x * w10 * ka1.x + qb1.y * w11 * ka1.y;
#ifndef NOWST
          if constexpr (KEEPW) {
            if (pi0.z >= 0)
              WST_STORE(reinterpret_cast<float2*>(p.Wst + ((size_t)(t * TP + r0) * 2 + ch) * F + col), make_float2(w00, w01));
            if (pi1.z >= 0)
              WST_STORE(reinterpret_cast<float2*>(p.Wst + ((size_t)(t * TP + r1) * 2 + ch) * F + col), make_float2(w10, w11));
          }
#endif
        }
#pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
          sab0 += __shfl_xor_sync(0xffffffff, sab0, o);
          sba0 += __shfl_xor_sync(0xffffffff, sba0, o);
          sab1 += __shfl_xor_sync(0xffffffff, sab1, o);
          sba1 += __shfl_xor_sync(0xffffffff, sba1, o);
        }
        const int slot = tig == 0 ? pi0.z : tig == 1 ? pi0.w : tig == 2 ? pi1.z : pi1.w;
        const float sv = tig == 0 ? sab0 : tig == 1 ? sba0 : tig == 2 ? sab1 : sba1;
        const float cr = tig < 2 ? ce0 : ce1;
        S.Sv[tig < 2 ? r0 : r1][tig & 1][h] = cr != 0.f ? cr * sv * inv_nc : 0.f;
      }
      // this CTA's heads' scores of the tile in place: per directed slot and channel its HC
      // as 16-byte stores at [slot][channel NH + half HC]
      bar_sync(1 + 2 * NST, NCW * 32);
      constexpr int HQ = HC / 4;
      for (int v = threadIdx.x - NPW * 32; v < TP * 2 * HQ; v += NCW * 32) {
        const int q = v % HQ, u = v / HQ;
        const int row = u / 2, side = u & 1;
        const int4 pi = S.pi[row];
        const int slot = side ? pi.w : pi.z;
        if (slot >= 0)
          *reinterpret_cast<float4*>(p.S + (size_t)slot * 2 * NH + ch0 * NH + 4 * q) =
              *reinterpret_cast<const float4*>(&S.Sv[row][side][4 * q]);
      }
      if (tile_of(i + NST) < tiles) bar_arrive(1 + NST + s, NTH);   // stage s free
    }
  }
}

struct MsgParams {
  const float *v1, *S;
  const double *px, *py, *pz, *box;
  const int *nbr, *cnt;
  const float *x_in, *ev_in, *scal;
  const int* row0;
  float *x_out, *ev_out;
  int n, K, nl, KS;
};

// each equivariant component of a half (0: components 0 .. M/2, 1: the rest): its degree
// entry's score times its harmonic; the half a template so entries and harmonics are compile-time
constexpr int MH = M / 2;
template <int HALF> __device__ __forceinline__ void deg_half_t(const float s2[NH], float x, float y, float z, float v[MH]) {
#pragma unroll
  for (int i = 0; i < MH; ++i) v[i] = s2[entry_of(HALF * MH + i)] * ylm(yidx(HALF * MH + i), x, y, z);
}
__device__ __forceinline__ void deg_half(int half, const float s2[NH], float x, float y, float z, float v[MH]) {
  if (half == 0) deg_half_t<0>(s2, x, y, z, v);
  else deg_half_t<1>(s2, x, y, z, v);
}

constexpr int MW = 8;
__global__ void __launch_bounds__(MW * 32) att_msg(MsgParams p) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, e = lane & 15, half = lane >> 4;
  const int r0 = p.row0[0];
  const int a = r0 + blockIdx.x * MW + warp;
  // the range's rows from row0[1] on: a chunk shifted back to end at the last row leaves
  // the rows it shares with the one before it as that one wrote them
  if (a < p.row0[1] || a >= r0 + p.nl || a >= p.n) return;
  const double Lx = p.box[0], Ly = p.box[1], Lz = p.box[2];
  const double iLx = 1.0 / Lx, iLy = 1.0 / Ly, iLz = 1.0 / Lz;
  const double ax = p.px[a], ay = p.py[a], az = p.pz[a];
  const int n_a = min(p.cnt[a], p.K);
  const float* Sa = p.S + (size_t)(a - r0) * p.K * 2 * NH;         // the range's slots
  const int* row = p.nbr + (size_t)a * p.K;
  float acc[NH];
#pragma unroll
  for (int h = 0; h < NH; ++h) acc[h] = 0.f;
  float dev[MH];
#pragma unroll
  for (int i = 0; i < MH; ++i) dev[i] = 0.f;
  // the row's neighbours and their heads' scores, 32 slots a lane each, handed out by shuffles
  for (int c0 = 0; c0 < n_a; c0 += 32) {
    const bool have = c0 + lane < n_a;
    const int nbc = have ? row[c0 + lane] : 0;
    float scc[NH];
#pragma unroll
    for (int h = 0; h < NH; ++h) scc[h] = have ? Sa[(size_t)(c0 + lane) * 2 * NH + h] : 0.f;
    const int cn = min(32, n_a - c0);
#pragma unroll 4
    for (int j = 0; j < cn; ++j) {
      const int b = edge_atom(__shfl_sync(0xffffffff, nbc, j), p.KS);
      const float* vr = p.v1 + (size_t)b * F + lane;
#pragma unroll
      for (int h = 0; h < NH; ++h) acc[h] += __shfl_sync(0xffffffff, scc[h], j) * vr[h * 32];
    }
  }
  for (int t0 = 0; t0 < n_a; t0 += 16) {
    int c = t0 + e;
    bool valid = c < n_a;
    double dxd, dyd, dzd;
    // the minimum image inline (not edge_disp): this loop keeps a's position and 1 / L in registers
    if (p.KS) {
      edge_disp(p.px, p.py, p.pz, a, valid ? row[c] : a * (2 * p.KS + 1) * (2 * p.KS + 1) * (2 * p.KS + 1),
                p.KS, Lx, Ly, Lz, dxd, dyd, dzd);
    } else {
      int b = valid ? row[c] : a;
      dxd = p.px[b] - ax; dyd = p.py[b] - ay; dzd = p.pz[b] - az;
      dxd -= Lx * floor(dxd * iLx + 0.5);
      dyd -= Ly * floor(dyd * iLy + 0.5);
      dzd -= Lz * floor(dzd * iLz + 0.5);
    }
    float dx = (float)dxd, dy = (float)dyd, dz = (float)dzd;
    float r2 = dx * dx + dy * dy + dz * dz;
    bool ok = valid && r2 > 0.f;
    float inv = ok ? rsqrtf(r2) : 0.f;
    float s2[NH];
#pragma unroll
    for (int d = 0; d < NH; ++d) s2[d] = valid ? Sa[(size_t)c * 2 * NH + NH + d] : 0.f;
    float v[MH];                                 // each component: its entry's score times its harmonic
    deg_half(half, s2, ok ? dx * inv : 1.f, ok ? dy * inv : 0.f, ok ? dz * inv : 0.f, v);
#pragma unroll
    for (int i = 0; i < MH; ++i) {
      v[i] += __shfl_xor_sync(0xffffffff, v[i], 1);
      v[i] += __shfl_xor_sync(0xffffffff, v[i], 2);
      v[i] += __shfl_xor_sync(0xffffffff, v[i], 4);
      v[i] += __shfl_xor_sync(0xffffffff, v[i], 8);
      dev[i] += v[i];
    }
  }
#pragma unroll
  for (int h = 0; h < NH; ++h) {
    size_t o = (size_t)a * F + h * 32 + lane;
    p.x_out[o] = p.x_in[o] + acc[h];
  }
  if (e == 0) {
#pragma unroll
    for (int i = 0; i < MH; ++i) {
      size_t o = (size_t)a * M + half * MH + i;
      p.ev_out[o] = p.ev_in[o] + dev[i];
    }
  }
}

static ffi::Error PrepImpl(cudaStream_t stream, F64 px, F64 py, F64 pz, F64 box, S32 nbr, S32 cnt, S32 z, S32 off,
                           S32 lo, S32 hi, S32 tile_atom, S32 row0, F32 rmin, F32 scal, S32 pidx0, RS32 PI,
                           RF32 PG, RS32 PIDX, int64_t zmax, int64_t P,
                           int64_t NL, int64_t KS) {
  PrepParams p;
  p.px = px.typed_data(); p.py = py.typed_data(); p.pz = pz.typed_data(); p.box = box.typed_data();
  p.nbr = nbr.typed_data(); p.cnt = cnt.typed_data(); p.z = z.typed_data(); p.off = off.typed_data();
  p.tile_atom = tile_atom.typed_data(); p.rmin = rmin.typed_data();
  p.scal = scal.typed_data();
  p.PI = reinterpret_cast<int4*>(PI->typed_data()); p.PG = reinterpret_cast<float4*>(PG->typed_data());
  p.PIDX = PIDX->typed_data();
  auto d = nbr.dimensions();
  p.n = (int)d[0]; p.K = (int)d[1]; p.zmax = (int)zmax; p.nl = (int)NL; p.pmax = (int)P; p.KS = (int)KS;
  p.lo = lo.typed_data(); p.hi = hi.typed_data(); p.row0 = row0.typed_data();
  pair_prep<<<((int)P + 255) / 256, 256, 0, stream>>>(p);
  return launch_status();
}

constexpr int PAIR_SMEM = BS_FLOATS * sizeof(float) + NST * sizeof(Stage);
#ifdef SMEM
// build.py sized the tiles by its count of the shared memory (with WA's): the kernel's own
static_assert(PAIR_SMEM + sizeof(FilterW1) == SMEM, "build.fwd_smem differs from the kernel's");
#endif

static ffi::Error PairImpl(cudaStream_t stream, F32 q1, F32 k1, F32 q2, F32 k2, F32 ev, S32 PI, F32 PG, S32 off,
                           F32 u1v, F32 u1d, F32 u2v, F32 u2d, F32 wa1, F32 ba1, F32 wb1, F32 bb1, F32 wa2, F32 ba2,
                           F32 wb2, F32 bb2, F32 vr1, F32 vr2, F32 cdeg, F32 scal, RF32 S,
                           RF32 Wst, int64_t P, int64_t keep_w) {
  PairParams p;
  p.q1 = q1.typed_data(); p.k1 = k1.typed_data(); p.q2 = q2.typed_data(); p.k2 = k2.typed_data();
  p.ev = ev.typed_data(); p.PI = reinterpret_cast<const int4*>(PI.typed_data());
  p.PG = reinterpret_cast<const float4*>(PG.typed_data()); p.off = off.typed_data();
  p.u1v = u1v.typed_data(); p.u1d = u1d.typed_data(); p.u2v = u2v.typed_data(); p.u2d = u2d.typed_data();
  p.wa1 = wa1.typed_data(); p.ba1 = ba1.typed_data(); p.wb1 = wb1.typed_data(); p.bb1 = bb1.typed_data();
  p.wa2 = wa2.typed_data(); p.ba2 = ba2.typed_data(); p.wb2 = wb2.typed_data(); p.bb2 = bb2.typed_data();
  p.vr1 = vr1.typed_data(); p.vr2 = vr2.typed_data(); p.cdeg = cdeg.typed_data(); p.scal = scal.typed_data();
  p.S = S->typed_data(); p.Wst = Wst->typed_data();
  p.n = (int)off.dimensions()[0] - 1; p.pmax = (int)P;
  if (keep_w) {
    set_max_smem<att_pair<true>>(PAIR_SMEM);
    att_pair<true><<<dim3(num_sms(), 2), NTH, PAIR_SMEM, stream>>>(p);
  } else {
    set_max_smem<att_pair<false>>(PAIR_SMEM);
    att_pair<false><<<dim3(num_sms(), 2), NTH, PAIR_SMEM, stream>>>(p);
  }
  return launch_status();
}

static ffi::Error MsgImpl(cudaStream_t stream, F32 v1, F32 S, F64 px, F64 py, F64 pz, F64 box, S32 nbr, S32 cnt,
                          F32 x_in, F32 ev_in, F32 scal, S32 row0, RF32 x_out,
                          RF32 ev_out, int64_t NL, int64_t KS) {
  MsgParams p;
  p.v1 = v1.typed_data(); p.S = S.typed_data(); p.px = px.typed_data(); p.py = py.typed_data();
  p.pz = pz.typed_data(); p.box = box.typed_data(); p.nbr = nbr.typed_data(); p.cnt = cnt.typed_data();
  p.x_in = x_in.typed_data(); p.ev_in = ev_in.typed_data(); p.scal = scal.typed_data();
  p.x_out = x_out->typed_data(); p.ev_out = ev_out->typed_data();
  auto d = nbr.dimensions();
  p.n = (int)d[0]; p.K = (int)d[1]; p.nl = (int)NL; p.row0 = row0.typed_data(); p.KS = (int)KS;
  att_msg<<<(p.nl + MW - 1) / MW, MW * 32, 0, stream>>>(p);
  return launch_status();
}


XLA_FFI_DEFINE_HANDLER_SYMBOL(PairPrep, PrepImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<F64>().Arg<F64>().Arg<F64>().Arg<F64>()                       // px py pz box
        .Arg<S32>().Arg<S32>().Arg<S32>().Arg<S32>().Arg<S32>().Arg<S32>()  // nbr cnt z off lo hi
        .Arg<S32>().Arg<S32>()                                             // tile_atom row0
        .Arg<F32>().Arg<F32>().Arg<S32>()                                  // rmin scal, PIDX zeroed (aliased)
        .Ret<S32>().Ret<F32>().Ret<S32>().Attr<int64_t>("zmax").Attr<int64_t>("P").Attr<int64_t>("NL")
        .Attr<int64_t>("KS"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(AttPair, PairImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>()            // q1 k1 q2 k2 ev
        .Arg<S32>().Arg<F32>().Arg<S32>()                                  // PI PG off
        .Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>()                       // u1v u1d u2v u2d
        .Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>()                       // wa1 ba1 wb1 bb1
        .Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>()                       // wa2 ba2 wb2 bb2
        .Arg<F32>().Arg<F32>().Arg<F32>().Arg<F32>()                       // vr1 vr2 cdeg scal
        .Ret<F32>().Ret<F32>().Attr<int64_t>("P").Attr<int64_t>("keep_w"));  // S, w per pair (if keep_w)

XLA_FFI_DEFINE_HANDLER_SYMBOL(AttMsg, MsgImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<F32>().Arg<F32>().Arg<F64>().Arg<F64>().Arg<F64>().Arg<F64>()   // v1 S px py pz box
        .Arg<S32>().Arg<S32>().Arg<F32>().Arg<F32>().Arg<F32>().Arg<S32>()    // nbr cnt x_in ev_in scal row0
        .Ret<F32>().Ret<F32>().Attr<int64_t>("NL").Attr<int64_t>("KS"));
