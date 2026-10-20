# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Track full examples used to qualify the campaign-derived Warp kernels."""

import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented

import newton.examples
from newton.examples.mpm.example_mpm_multi_material import Example as MPMExample
from newton.examples.softbody.example_softbody_franka import Example as ElasticityExample
from newton.viewer import ViewerNull


class WarpKernelExamples:
    """Measure complete simulation steps, including solver setup and host dispatch."""

    params = ["elasticity", "mpm"]
    param_names = ["example"]
    number = 1
    repeat = 5
    rounds = 2

    def setup(self, example):
        if not wp.get_device().is_cuda:
            raise SkipNotImplemented
        example_class = ElasticityExample if example == "elasticity" else MPMExample
        parser = example_class.create_parser() if example == "mpm" else newton.examples.create_parser()
        self.example = example_class(ViewerNull(num_frames=1000), parser.parse_args([]))
        for _ in range(60):
            self.example.step()
        wp.synchronize_device()

    def time_step(self, example):
        for _ in range(20):
            self.example.step()
        wp.synchronize_device()
