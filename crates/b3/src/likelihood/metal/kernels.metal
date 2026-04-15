#include <metal_stdlib>
using namespace metal;

constant uint NUM_PATTERNS [[function_constant(0)]];
constant uint NUM_LEAVES [[function_constant(1)]];
constant uint SCALE_LN [[function_constant(2)]];
constant float SCALE_THRESHOLD [[function_constant(3)]];
constant float SCALE_MULT [[function_constant(4)]];

inline uint id(uint edge, uint pattern) {
  return edge * NUM_PATTERNS + pattern;
}

inline uint tid(uint node, uint sub) { return node * 4 + sub; }

kernel void propose(device const uchar* leaves [[buffer(0)]],
                    device float4* projections [[buffer(1)]],
                    device uchar* scales [[buffer(2)]],
                    device uint* scale_sums [[buffer(3)]],
                    device const uint* nodes [[buffer(4)]],
                    device const uint* children [[buffer(5)]],
                    device const float4* transitions [[buffer(6)]],

                    constant uint& num_updated_nodes [[buffer(7)]],
                    constant uint& leaves_end [[buffer(8)]],
                    constant uint& internals_start [[buffer(9)]],

                    uint global_id [[thread_position_in_grid]],
                    uint tile [[quadgroup_index_in_threadgroup]],
                    uint sub [[thread_index_in_quadgroup]]) {
  uint pattern = global_id / 4;
  if (pattern >= NUM_PATTERNS) {
    return;
  }

  for (uint i = 0; i < leaves_end; i++) {
    uchar leaf = leaves[id(nodes[i], pattern)];
    float projection = 0.0f;
    float4 t_row = transitions[tid(i, sub)];
    if (leaf & 0b0001) {
      projection += t_row.x;
    }
    if (leaf & 0b0010) {
      projection += t_row.y;
    }
    if (leaf & 0b0100) {
      projection += t_row.z;
    }
    if (leaf & 0b1000) {
      projection += t_row.w;
    }

    float4 assembled_projection =
        float4(quad_broadcast(projection, 0), quad_broadcast(projection, 1),
               quad_broadcast(projection, 2), quad_broadcast(projection, 3));
    if (sub == 0) {
      projections[id(nodes[i], pattern)] = assembled_projection;
    }
  }

  uint scale_sum = scale_sums[pattern];
  for (uint i = internals_start; i < num_updated_nodes; i++) {
    uint left = children[(i - internals_start) * 2];
    uint right = children[(i - internals_start) * 2 + 1];
    uint current = nodes[i];
    uint scale_id = id(current, pattern);
    uint old_scale = scales[scale_id];

    float sub_likelihood = projections[id(left, pattern)][sub] *
                           projections[id(right, pattern)][sub];

    bool should_scale = quad_all(sub_likelihood < SCALE_THRESHOLD);
    if (should_scale) {
      sub_likelihood *= SCALE_MULT;
    }

    if (sub == 0 && should_scale != old_scale) {
      scales[scale_id] = should_scale;
      if (old_scale == 0) {
        scale_sum += SCALE_LN;
      } else {
        scale_sum -= SCALE_LN;
      }
    }

    float4 likelihood = float4(
        quad_broadcast(sub_likelihood, 0), quad_broadcast(sub_likelihood, 1),
        quad_broadcast(sub_likelihood, 2), quad_broadcast(sub_likelihood, 3));

    float projection = dot(transitions[tid(i, sub)], likelihood);

    float4 assembled_final =
        float4(quad_broadcast(projection, 0), quad_broadcast(projection, 1),
               quad_broadcast(projection, 2), quad_broadcast(projection, 3));
    if (sub == 0) {
      projections[id(current, pattern)] = assembled_final;
    }
  }

  if (sub == 0) {
    scale_sums[pattern] = scale_sum;
  }
}

kernel void update_likelihoods(device const float4* projections [[buffer(0)]],
                               device float* likelihoods [[buffer(1)]],
                               device uchar* scales [[buffer(2)]],
                               device uint* scale_sums [[buffer(3)]],

                               constant uint& root [[buffer(4)]],
                               constant uint& left_child [[buffer(5)]],
                               constant uint& right_child [[buffer(6)]],
                               constant float4& frequencies [[buffer(7)]],

                               uint pattern [[thread_position_in_grid]]) {
  if (pattern >= NUM_PATTERNS) {
    return;
  }
  float4 likelihood = (projections[id(left_child, pattern)] *
                       projections[id(right_child, pattern)]) *
                      frequencies;

  float sum = likelihood.x + likelihood.y + likelihood.z + likelihood.w;
  likelihoods[pattern] = log(sum);

  if (scales[id(root, pattern)]) {
    scales[id(root, pattern)] = 0;
    scale_sums[pattern] -= SCALE_LN;
  }
}

kernel void copy_projections(device const float4* projections_src [[buffer(0)]],
                             device float4* projections_dst [[buffer(1)]],

                             device const uchar* scales_src [[buffer(2)]],
                             device uchar* scales_dst [[buffer(3)]],

                             device const uint* nodes [[buffer(4)]],

                             uint3 global_id [[thread_position_in_grid]]) {
  uint pattern = global_id.x;
  if (pattern >= NUM_PATTERNS) {
    return;
  }
  uint node = global_id.y;
  uint proj_id = id(nodes[node], pattern);
  projections_dst[proj_id] = projections_src[proj_id];
  scales_dst[proj_id] = scales_src[proj_id];
}