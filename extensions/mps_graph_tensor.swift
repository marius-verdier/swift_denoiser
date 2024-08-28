//
//  mps_graph_tensor.swift
//  Denoiser Swift / Extensions
//
//  Created by Marius Verdier on 28/07/2024.
//

import Foundation
import MetalPerformanceShadersGraph

extension MPSGraphTensor {
    var dim: Int? { shape?.count }
}
