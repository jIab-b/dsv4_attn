#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <torch/extension.h>

#include "params.h"
#include "sm100/kernel.h"

namespace dsv4::hca {

constexpr int kHcaDecodeTileTokens = 64;

struct HcaDecodeScheduleEntry {
    int logical_start;
    int logical_end;
    int compressed_start;
    int compressed_end;
    int swa_start;
    int swa_end;
};

struct HcaDecodeSchedule {
    int B;
    int M_cur;
    int swa_len;
    int tile_tokens;
    int total_kv;
    int partitions_per_batch;
    bool run_compression;
    std::vector<HcaDecodeScheduleEntry> entries;

    dim3 grid(int head_blocks = 1) const {
        return dim3(partitions_per_batch, B, head_blocks);
    }
};

inline int hca_ceil_div(int x, int y) {
    return x / y + (x % y != 0);
}

inline HcaDecodeSchedule build_hca_decode_schedule(
    int64_t B,
    int64_t M_cur,
    int64_t swa_len,
    int64_t partial_count,
    int64_t tile_tokens = kHcaDecodeTileTokens
) {
    TORCH_CHECK(B >= 0, "B must be non-negative");
    TORCH_CHECK(M_cur >= 0, "M_cur must be non-negative");
    TORCH_CHECK(swa_len >= 0 && swa_len <= 128, "swa_len must be in [0, 128]");
    TORCH_CHECK(partial_count >= 0 && partial_count < 128,
                "partial_count must be in [0, 127]");
    TORCH_CHECK(tile_tokens == kHcaDecodeTileTokens,
                "decode schedule currently assumes 64-token KV tiles");
    TORCH_CHECK(B <= std::numeric_limits<int>::max() &&
                M_cur <= std::numeric_limits<int>::max(),
                "decode schedule dimensions exceed int range");

    TORCH_CHECK(M_cur + swa_len <= std::numeric_limits<int>::max(),
                "total KV length exceeds int range");
    const int total_kv = static_cast<int>(M_cur + swa_len);

    HcaDecodeSchedule schedule{};
    schedule.B = static_cast<int>(B);
    schedule.M_cur = static_cast<int>(M_cur);
    schedule.swa_len = static_cast<int>(swa_len);
    schedule.tile_tokens = static_cast<int>(tile_tokens);
    schedule.total_kv = total_kv;
    schedule.partitions_per_batch =
        total_kv == 0 ? 0 : hca_ceil_div(total_kv, schedule.tile_tokens);
    schedule.run_compression = (partial_count == 127);
    schedule.entries.reserve(schedule.partitions_per_batch);

    for (int p = 0; p < schedule.partitions_per_batch; ++p) {
        const int logical_start = p * schedule.tile_tokens;
        const int logical_end = std::min(logical_start + schedule.tile_tokens,
                                         schedule.total_kv);

        HcaDecodeScheduleEntry entry{};
        entry.logical_start = logical_start;
        entry.logical_end = logical_end;
        entry.compressed_start = std::min(logical_start, schedule.M_cur);
        entry.compressed_end = std::min(logical_end, schedule.M_cur);
        entry.swa_start = std::max(0, logical_start - schedule.M_cur);
        entry.swa_end = std::min(schedule.swa_len,
                                 std::max(0, logical_end - schedule.M_cur));
        schedule.entries.push_back(entry);
    }

    return schedule;
}

inline HcaDecodeSchedule build_hca_decode_schedule(
    const HcaParams& p,
    int64_t tile_tokens = kHcaDecodeTileTokens
) {
    return build_hca_decode_schedule(
        p.B, p.M_cur, p.swa_len, p.partial_count, tile_tokens);
}

inline at::Tensor hca_decode_schedule_cpu(
    int64_t B,
    int64_t M_cur,
    int64_t swa_len,
    int64_t partial_count,
    int64_t tile_tokens = kHcaDecodeTileTokens
) {
    HcaDecodeSchedule schedule = build_hca_decode_schedule(
        B, M_cur, swa_len, partial_count, tile_tokens);

    at::Tensor out = at::empty(
        {schedule.partitions_per_batch, 6},
        at::TensorOptions().dtype(at::kInt).device(at::kCPU));
    auto* rows = out.data_ptr<int32_t>();
    for (int i = 0; i < schedule.partitions_per_batch; ++i) {
        const HcaDecodeScheduleEntry& e = schedule.entries[i];
        rows[i * 6 + 0] = e.logical_start;
        rows[i * 6 + 1] = e.logical_end;
        rows[i * 6 + 2] = e.compressed_start;
        rows[i * 6 + 3] = e.compressed_end;
        rows[i * 6 + 4] = e.swa_start;
        rows[i * 6 + 5] = e.swa_end;
    }
    return out;
}

inline at::Tensor hca_compress_reduce_fwd(
    const at::Tensor& C,
    const at::Tensor& Z,
    const at::Tensor& bias,
    int64_t m_prime
) {
    TORCH_CHECK(C.is_cuda() && Z.is_cuda() && bias.is_cuda());
    TORCH_CHECK(C.scalar_type() == at::kBFloat16 && Z.scalar_type() == at::kBFloat16
                && bias.scalar_type() == at::kBFloat16);
    TORCH_CHECK(C.dim() == 3 && C.sizes() == Z.sizes());
    TORCH_CHECK(C.is_contiguous() && Z.is_contiguous());
    TORCH_CHECK(bias.dim() == 2 && bias.size(0) == m_prime && bias.size(1) == C.size(2));

    const int64_t Bs = C.size(0);
    const int64_t n  = C.size(1);
    const int64_t c_out = C.size(2);
    TORCH_CHECK(n % m_prime == 0);
    const int64_t n_blocks = n / m_prime;

    at::Tensor out = at::empty({Bs, n_blocks, c_out}, C.options());

    HcaCompressReduceParams p{};
    p.B           = static_cast<int>(Bs);
    p.n           = static_cast<int>(n);
    p.n_blocks    = static_cast<int>(n_blocks);
    p.c_out       = static_cast<int>(c_out);
    p.m_prime     = static_cast<int>(m_prime);
    p.C           = reinterpret_cast<const __nv_bfloat16*>(C.data_ptr());
    p.Z           = reinterpret_cast<const __nv_bfloat16*>(Z.data_ptr());
    p.bias        = reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr());
    p.out         = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.stride_C_b      = static_cast<int>(n * c_out);
    p.stride_C_n      = static_cast<int>(c_out);
    p.stride_out_b    = static_cast<int>(n_blocks * c_out);
    p.stride_out_blk  = static_cast<int>(c_out);

    TORCH_CHECK(m_prime == 128);
    constexpr int THREADS = 256;
    dim3 grid(p.n_blocks, p.B);
    if (n_blocks * Bs >= 74) {
        sm100::hca_compress_kernel<128, THREADS><<<grid, THREADS>>>(p);
    } else {
        constexpr int K_SPLITS = 8;
        dim3 grid_sk(p.n_blocks, p.B, K_SPLITS);
        sm100::hca_compress_splitk_kernel<128, THREADS, K_SPLITS><<<grid_sk, THREADS>>>(p);
    }
    return out;
}

}
