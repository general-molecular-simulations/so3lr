// Shared by the attention kernels (attn_pair7.cu, attn_bwd7.cu): the model's widths, the TF32
// x3 tensor-core product, the degree entries' layout, the real spherical harmonics and their
// gradients, and the pipelines' barriers and ldmatrix fragment loads.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// the model's widths (so3lr-tauri-m by default; -DVF=256 -DVNH=8 -DVM=48 -DVSH=64 -DVDEGS=0x44332211
// for so3lr-tauri-l): features, heads (one per degree entry), equivariant components, the
// spherical filters' hidden units, the degree of each entry
#ifndef VF
#define VF 128
#endif
#ifndef VNH
#define VNH 4
#endif
#ifndef VM
#define VM 24
#endif
#ifndef VSH
#define VSH 32
#endif
#ifndef VDEGS
#define VDEGS 0x4321                      // the degree of entry e in bits 4 e .. 4 e + 3 (entry 0 lowest)
#endif
constexpr int F = VF;
constexpr int SH = VSH;
constexpr int M = VM;
constexpr int NH = VNH;
constexpr int HC = NH;                    // heads per CTA (32 columns each): every head of the model
constexpr int FC = HC * 32;               // the CTA's filter columns
constexpr int MC = M / 8;                 // equivariant components per producer thread
__host__ __device__ constexpr int deg_of(int e) { return (VDEGS >> (4 * e)) & 15; }
static_assert(NH % HC == 0 && F == NH * 32 && M % 8 == 0 && SH % 32 == 0, "one head of 32 features per degree entry");
#ifndef RANK
#define RANK 48
#endif
constexpr int R = RANK;                   // the radial tables' rank
constexpr int KD = SH + R;                // the filter product's inner dimension: [hz | coeffs]
// the spherical filters' first layer of both channels (a row of weights per degree entry, then
// the bias), in both kernels' shared memory
using FilterW1 = float[2][NH + 1][SH];
// the entries' first equivariant component and harmonic offset (d^2 - 1), as the engine's ys
__host__ __device__ constexpr int ev_start(int e) {
  int st = 0;
  for (int i = 0; i < e; ++i) st += 2 * deg_of(i) + 1;
  return st;
}
__host__ __device__ constexpr int y_off(int e) { return deg_of(e) * deg_of(e) - 1; }

// TF32 operand bits: the float rounded to its top 19 bits (IADD + LOP3; cvt.rna.tf32
// lowers to a rounding sequence). Split as hi = tf32(x), lo = tf32(x - hi): x - hi is exact, so
// x is represented to ~2^-20, as the 3xTF32 products need.
__device__ __forceinline__ uint32_t tf32(float x) { return (__float_as_uint(x) + 0x1000u) & 0xFFFFE000u; }   // nearest (ties away), unbiased

// the low part as raw float bits: the tensor cores read only its top 19 bits (truncation of a
// quantity of either sign, so no bias), saving the rounding of every lo operand
__device__ __forceinline__ uint32_t tlo(float x) { return __float_as_uint(x); }

__device__ __forceinline__ void mma(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// degree entry of equivariant column m, and the real harmonic it goes with: compare-select
// chains over the entries with their starts folded at compile time (an entry index known
// only at run time would put ev_start's loop into every use)
template <int E> __device__ __forceinline__ int entry_step(int m, int e) {
  if constexpr (E < NH) return entry_step<E + 1>(m, m >= ev_start(E) ? E : e);
  else return e;
}
template <int E> __device__ __forceinline__ int yidx_step(int m, int y) {
  if constexpr (E < NH) return yidx_step<E + 1>(m, m >= ev_start(E) ? y_off(E) + (m - ev_start(E)) : y);
  else return y;
}
__device__ __forceinline__ int entry_of(int m) { return entry_step<0>(m, 0); }
__device__ __forceinline__ int yidx(int m) { return yidx_step<0>(m, 0); }

// the real spherical harmonics' constants (degrees 1-4)
namespace yc {
constexpr float c1 = 0.4886025119029199f, a2 = 1.0925484305920792f, b2 = 0.31539156525252005f, c2 = 0.5462742152960396f, d0 = 0.5900435899266435f, d1 = 2.890611442640554f, d2 = 0.4570457994644658f, d3 = 0.3731763325901154f, d5 = 1.445305721320277f, e0 = 2.5033429417967046f, e1 = 1.7701307697799304f, e2 = 0.9461746957575601f, e3 = 0.6690465435572892f, e4 = 0.10578554691520431f, e6 = 0.47308734787878004f, e8 = 0.6258357354491761f;
}

// real spherical harmonics, component m (0..23), as flash's _ylm_components
__device__ __forceinline__ float ylm(int m, float x, float y, float z) {
  using namespace yc;
  switch (m) {
    case 0: return c1 * y;
    case 1: return c1 * z;
    case 2: return c1 * x;
    case 3: return a2 * x * y;
    case 4: return a2 * y * z;
    case 5: return b2 * (3.f * z * z - 1.f);
    case 6: return a2 * x * z;
    case 7: return c2 * (x * x - y * y);
    case 8: return d0 * y * (3.f * x * x - y * y);
    case 9: return d1 * x * y * z;
    case 10: return d2 * y * (5.f * z * z - 1.f);
    case 11: return d3 * (5.f * z * z * z - 3.f * z);
    case 12: return d2 * x * (5.f * z * z - 1.f);
    case 13: return d5 * (x * x - y * y) * z;
    case 14: return d0 * x * (x * x - 3.f * y * y);
    case 15: return e0 * x * y * (x * x - y * y);
    case 16: return e1 * y * (3.f * x * x - y * y) * z;
    case 17: return e2 * x * y * (7.f * z * z - 1.f);
    case 18: return e3 * y * (7.f * z * z * z - 3.f * z);
    case 19: return e4 * (35.f * z * z * z * z - 30.f * z * z + 3.f);
    case 20: return e3 * x * (7.f * z * z * z - 3.f * z);
    case 21: return e6 * (x * x - y * y) * (7.f * z * z - 1.f);
    case 22: return e1 * x * (x * x - 3.f * y * y) * z;
    default: return e8 * (x * x * (x * x - 3.f * y * y) - y * y * (3.f * x * x - y * y));
  }
}

// d Y_m / d(x, y, z), as flash's _ylm_vjp per column
__device__ __forceinline__ void dylm(int m, float x, float y, float z, float& gx, float& gy, float& gz) {
  using namespace yc;
  const float q5 = 5.f * z * z - 1.f, q7 = 7.f * z * z - 1.f, c3 = 7.f * z * z * z - 3.f * z,
              dc3 = 21.f * z * z - 3.f;
  gx = gy = gz = 0.f;
  switch (m) {
    case 0: gy = c1; break;
    case 1: gz = c1; break;
    case 2: gx = c1; break;
    case 3: gx = a2 * y; gy = a2 * x; break;
    case 4: gy = a2 * z; gz = a2 * y; break;
    case 5: gz = 6.f * b2 * z; break;
    case 6: gx = a2 * z; gz = a2 * x; break;
    case 7: gx = 2.f * c2 * x; gy = -2.f * c2 * y; break;
    case 8: gx = 6.f * d0 * x * y; gy = d0 * (3.f * x * x - 3.f * y * y); break;
    case 9: gx = d1 * y * z; gy = d1 * x * z; gz = d1 * x * y; break;
    case 10: gy = d2 * q5; gz = 10.f * d2 * y * z; break;
    case 11: gz = d3 * (15.f * z * z - 3.f); break;
    case 12: gx = d2 * q5; gz = 10.f * d2 * x * z; break;
    case 13: gx = 2.f * d5 * x * z; gy = -2.f * d5 * y * z; gz = d5 * (x * x - y * y); break;
    case 14: gx = d0 * (3.f * x * x - 3.f * y * y); gy = -6.f * d0 * x * y; break;
    case 15: gx = e0 * (3.f * x * x * y - y * y * y); gy = e0 * (x * x * x - 3.f * x * y * y); break;
    case 16: gx = 6.f * e1 * x * y * z; gy = e1 * (3.f * x * x * z - 3.f * y * y * z);
             gz = e1 * (3.f * x * x * y - y * y * y); break;
    case 17: gx = e2 * y * q7; gy = e2 * x * q7; gz = 14.f * e2 * x * y * z; break;
    case 18: gy = e3 * c3; gz = e3 * y * dc3; break;
    case 19: gz = e4 * (140.f * z * z * z - 60.f * z); break;
    case 20: gx = e3 * c3; gz = e3 * x * dc3; break;
    case 21: gx = 2.f * e6 * x * q7; gy = -2.f * e6 * y * q7; gz = 14.f * e6 * z * (x * x - y * y); break;
    case 22: gx = e1 * (3.f * x * x * z - 3.f * y * y * z); gy = -6.f * e1 * x * y * z;
             gz = e1 * (x * x * x - 3.f * x * y * y); break;
    default: gx = e8 * (4.f * x * x * x - 12.f * x * y * y); gy = e8 * (4.f * y * y * y - 12.f * x * x * y); break;
  }
}

// the pipelines' named barriers (producer / consumer warps) and the persistent tile loop
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ int tile_of(int i) { return i * gridDim.x + blockIdx.x; }

// a 16 x 8 fp32 tile (rows r0.., columns c0..) as the m16k8 A fragment: lanes 0-7 address rows
// 0-7 at column c0, 8-15 rows 0-7 at c0 + 4, 16-23 rows 8-15 at c0, 24-31 rows 8-15 at c0 + 4
__device__ __forceinline__ void ldm_a(uint32_t a[4], const float* base, int stride, int lane) {
  const float* p = base + ((lane & 7) + ((lane >> 4) << 3)) * stride + ((lane >> 3) & 1) * 4;
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(p);
  uint32_t r0, r1, r2, r3;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
  a[0] = r0; a[1] = r2; a[2] = r1; a[3] = r3;        // (A[g][tig], A[g+8][tig], A[g][tig+4], A[g+8][tig+4])
}
// two 8 x 8 fp32 tiles of a transposed weight matrix Bt[n][k] (n rows): for n-tiles nt, nt + 1
// the (b0, b1) = (B[tig][g], B[tig+4][g]) fragments of k-step kt
__device__ __forceinline__ void ldm_b2(uint32_t b[2][2], const float* bt, int stride, int lane) {
  const float* p = bt + ((lane & 7) + ((lane >> 4) << 3)) * stride + ((lane >> 3) & 1) * 4;
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(b[0][0]), "=r"(b[0][1]), "=r"(b[1][0]), "=r"(b[1][1]) : "r"(addr));
}
