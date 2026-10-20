# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cooperative VBD elasticity derived from the Kernel Factory pilot."""

import warp as wp

from ...geometry import ParticleFlags
from ...utils.mesh import MeshAdjacencyData
from .particle_vbd_kernels import (
    evaluate_dihedral_angle_based_bending_force_hessian,
    evaluate_neo_hookean_membrane_force_hessian,
    evaluate_volumetric_neo_hookean_force_and_hessian,
)

wp.set_module_options({"enable_backward": False, "block_dim": 96})

ForceHessian9 = wp.types.vector(length=9, dtype=wp.float32)


@wp.kernel
def solve_elasticity_cooperative(
    dt: float,
    particle_ids_in_color: wp.array[wp.int32],
    pos_prev: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    mass: wp.array[float],
    inertia: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    tri_poses: wp.array[wp.mat22],
    tri_materials: wp.array2d[float],
    tri_areas: wp.array[float],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_angles: wp.array[float],
    edge_rest_length: wp.array[float],
    edge_bending_properties: wp.array2d[float],
    tet_indices: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    tet_materials: wp.array2d[float],
    particle_adjacency: MeshAdjacencyData,
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
    particle_displacements: wp.array[wp.vec3],
):
    """Reduce elastic force and symmetric Hessian across three warps per particle."""
    thread = wp.tid()
    vertex_slot = thread // 96
    lane = thread % 96
    # A separate warp traverses each material adjacency; no extra setup buffers.
    adjacency_lane = lane % 32
    material_role = lane // 32
    particle_index = particle_ids_in_color[vertex_slot]
    particle_mass = mass[particle_index]
    if not particle_flags[particle_index] & ParticleFlags.ACTIVE or particle_mass == 0.0:
        if lane == 0:
            particle_displacements[particle_index] = wp.vec3(0.0)
        return
    local_force = wp.vec3(0.0)
    local_hessian = wp.mat33(0.0)
    if material_role == 0 and tri_indices.shape[0] > 0:
        face_begin = particle_adjacency.v_adj_tris_offsets[particle_index]
        face_end = particle_adjacency.v_adj_tris_offsets[particle_index + 1]
        for offset in range(face_begin + 2 * adjacency_lane, face_end, 64):
            face_index = particle_adjacency.v_adj_tris[offset]
            vertex_order = particle_adjacency.v_adj_tris[offset + 1]
            if tri_materials[face_index, 0] > 0.0 or tri_materials[face_index, 1] > 0.0:
                force, hessian = evaluate_neo_hookean_membrane_force_hessian(
                    face_index,
                    vertex_order,
                    pos,
                    pos_prev,
                    tri_indices,
                    tri_poses[face_index],
                    tri_areas[face_index],
                    tri_materials[face_index, 0],
                    tri_materials[face_index, 1],
                    tri_materials[face_index, 2],
                    dt,
                )
                local_force += force
                local_hessian += hessian
    if material_role == 1 and edge_indices.shape[0] > 0:
        edge_begin = particle_adjacency.v_adj_edges_offsets[particle_index]
        edge_end = particle_adjacency.v_adj_edges_offsets[particle_index + 1]
        for offset in range(edge_begin + 2 * adjacency_lane, edge_end, 64):
            edge_index = particle_adjacency.v_adj_edges[offset]
            vertex_order = particle_adjacency.v_adj_edges[offset + 1]
            if edge_bending_properties[edge_index, 0] > 0.0:
                force, hessian = evaluate_dihedral_angle_based_bending_force_hessian(
                    edge_index,
                    vertex_order,
                    pos,
                    pos_prev,
                    edge_indices,
                    edge_rest_angles,
                    edge_rest_length,
                    edge_bending_properties[edge_index, 0],
                    edge_bending_properties[edge_index, 1],
                    dt,
                )
                local_force += force
                local_hessian += hessian
    if material_role == 2 and tet_indices.shape[0] > 0:
        tet_begin = particle_adjacency.v_adj_tets_offsets[particle_index]
        tet_end = particle_adjacency.v_adj_tets_offsets[particle_index + 1]
        for offset in range(tet_begin + 2 * adjacency_lane, tet_end, 64):
            tet_index = particle_adjacency.v_adj_tets[offset]
            vertex_order = particle_adjacency.v_adj_tets[offset + 1]
            if tet_materials[tet_index, 0] > 0.0 or tet_materials[tet_index, 1] > 0.0:
                force, hessian = evaluate_volumetric_neo_hookean_force_and_hessian(
                    tet_index,
                    vertex_order,
                    pos_prev,
                    pos,
                    tet_indices,
                    tet_poses[tet_index],
                    tet_materials[tet_index, 0],
                    tet_materials[tet_index, 1],
                    tet_materials[tet_index, 2],
                    dt,
                )
                local_force += force
                local_hessian += hessian
    local_packed = ForceHessian9(
        local_force[0],
        local_force[1],
        local_force[2],
        local_hessian[0, 0],
        local_hessian[0, 1],
        local_hessian[0, 2],
        local_hessian[1, 1],
        local_hessian[1, 2],
        local_hessian[2, 2],
    )
    packed_sum = wp.tile_sum(wp.tile(local_packed, preserve_type=True))[0]
    force_sum = wp.vec3(packed_sum[0], packed_sum[1], packed_sum[2])
    hessian_sum = wp.mat33(
        packed_sum[3],
        packed_sum[4],
        packed_sum[5],
        packed_sum[4],
        packed_sum[6],
        packed_sum[7],
        packed_sum[5],
        packed_sum[7],
        packed_sum[8],
    )
    if lane == 0:
        inertia_scale = particle_mass / (dt * dt)
        force = particle_mass * (inertia[particle_index] - pos[particle_index]) * (1.0 / (dt * dt))
        hessian = inertia_scale * wp.identity(n=3, dtype=float)
        force += force_sum
        hessian += hessian_sum
        force += particle_forces[particle_index]
        hessian += particle_hessians[particle_index]
        if wp.abs(wp.determinant(hessian)) > 1.0e-8:
            particle_displacements[particle_index] = (
                particle_displacements[particle_index] + wp.inverse(hessian) * force
            )
