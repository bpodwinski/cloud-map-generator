// Entry points of the df64 self-test (appended after df64.wgsl). Inputs and outputs are arrays of vec2<f32>.
// They run under whatever DF64_FMA / DF64_BARRIER the host sets, so the probe measures each configuration.

@group(0) @binding(0) var<storage, read> pair_in: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> pair_out: array<vec2<f32>>;

@compute @workgroup_size(64)
fn probe_two_sum(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&pair_in)) {
        let p = pair_in[id.x];
        pair_out[id.x] = df64_two_sum(p.x, p.y);
    }
}

// The product variant chosen by DF64_FMA.
@compute @workgroup_size(64)
fn probe_two_prod(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&pair_in)) {
        let p = pair_in[id.x];
        pair_out[id.x] = df64_two_prod(p.x, p.y);
    }
}
