// Entry points of the df64 accuracy tests (appended after df64.wgsl): c = op(a, b) on arrays of df64.

@group(0) @binding(0) var<storage, read> a_in: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> b_in: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> c_out: array<vec2<f32>>;

@compute @workgroup_size(64)
fn op_add(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&a_in)) { c_out[id.x] = df64_add(a_in[id.x], b_in[id.x]); }
}

@compute @workgroup_size(64)
fn op_sub(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&a_in)) { c_out[id.x] = df64_sub(a_in[id.x], b_in[id.x]); }
}

@compute @workgroup_size(64)
fn op_mul(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&a_in)) { c_out[id.x] = df64_mul(a_in[id.x], b_in[id.x]); }
}

@compute @workgroup_size(64)
fn op_div(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&a_in)) { c_out[id.x] = df64_div(a_in[id.x], b_in[id.x]); }
}

@compute @workgroup_size(64)
fn op_sqrt(@builtin(global_invocation_id) id: vec3<u32>) {
    _ = arrayLength(&b_in); // keeps binding 1 in the auto layout
    if (id.x < arrayLength(&a_in)) { c_out[id.x] = df64_sqrt(a_in[id.x]); }
}

@compute @workgroup_size(64)
fn op_floor(@builtin(global_invocation_id) id: vec3<u32>) {
    _ = arrayLength(&b_in); // keeps binding 1 in the auto layout
    if (id.x < arrayLength(&a_in)) { c_out[id.x] = df64_floor(a_in[id.x]); }
}

@compute @workgroup_size(64)
fn op_fract(@builtin(global_invocation_id) id: vec3<u32>) {
    _ = arrayLength(&b_in); // keeps binding 1 in the auto layout
    if (id.x < arrayLength(&a_in)) { c_out[id.x] = df64_fract(a_in[id.x]); }
}

@compute @workgroup_size(64)
fn op_mul_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&a_in)) { c_out[id.x] = df64_mul_f32(a_in[id.x], b_in[id.x].x); }
}
