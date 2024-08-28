//
//  mps_graph.swift
//  Denoiser Swift / Extensions
//
//  Created by Marius Verdier on 28/07/2024.
//

import Foundation

import MetalPerformanceShadersGraph
import Accelerate

extension MPSGraph {
    
    public var invSqrt2: MPSGraphTensor {
        constant(1/sqrt(2), dataType: .float32)
    }
    
    private func sinc(_ x: Float) -> Float {
        if x == 0 {
            return 1
        } else {
            return Darwin.sin(x)/x
        }
    }
    
    public func sincKernel(_ zeros: Int = 56) -> MPSGraphTensor {
        var kernelArray: [Float]
        let windowSize = 4 * zeros + 1
        var window = [Float](repeating: 0.0, count: windowSize)
        
        vDSP_hann_window(&window, vDSP_Length(windowSize), Int32(vDSP_HANN_NORM))
        
        let winodd = stride(from: 1, to: windowSize, by: 2).map { window[Int($0)] }
        

        let t = stride(from: -Float(zeros) + 0.5, to: Float(zeros) + 0.5, by: 1).map { $0 * .pi }
        
        let sincValues = t.map(sinc)
        
        kernelArray = zip(sincValues, winodd).map(*).map { Float($0) }
        
        let data = Data(bytes: kernelArray, count: kernelArray.count * MemoryLayout<Float>.size)
        return constant(data, shape: [1,1,kernelArray.count as NSNumber], dataType: .float32)
    }
    
    public func upsample(_ tensor: MPSGraphTensor, zeros: Int = 56, name: String? = nil) -> MPSGraphTensor {
        var x = tensor
        let kernel = sincKernel(zeros)
        let dims = tensor.shape!.dims3D
        
        var x_r = reshape(x, shape: [-1, 1, dims.2], name: name)

        var out = convolution1D(source: x_r, weights: kernel, biases: constant(0, dataType: .float32), descriptor: MPSGraphConvolution1dOpDescriptor(stride: 1, dilationRate: 1, paddingLeft: zeros, paddingRight: zeros, paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW))
        out = sliceTensor(out, starts: [0,0,1], ends: out.shape!, strides: [1,1,1], name: nil)
        
        out = reshape(out, shape: tensor.shape!, name: name)
        var y = stack([x, out], axis: -1, name: name)

        y = reshape(y, shape: [dims.0, dims.1, -1], name: name)
        
        return y
    }
    
    public func downsample(_ tensor: MPSGraphTensor, zeros: Int = 56, name: String? = nil) -> MPSGraphTensor {
        var x = tensor
        let kernel = sincKernel(zeros)
        
        if x.shape!.dims3D.2.intValue % 2 != 0 {
            x = padTensor(x, with: .constant, leftPadding: [NSNumber](repeating: 0, count: x.shape!.count), rightPadding: [NSNumber](repeating: 0, count: x.shape!.count - 1) + [1 as NSNumber], constantValue: 0, name: nil)
        }
        
        var x_even = sliceTensor(x, starts: [0,0,0], ends: x.shape!, strides: [1,1,2], name: nil)
        var x_odd = sliceTensor(x, starts: [0,0,1], ends: x.shape!, strides: [1,1,2], name: nil)
        
        let dims = x_odd.shape!.dims3D
        let conv_inp = reshape(x_odd, shape: [-1, 1, dims.2], name: nil)
        
        var conv = convolution1D(source: conv_inp, weights: kernel, biases: constant(0, dataType: .float32), descriptor: MPSGraphConvolution1dOpDescriptor(stride: 1, dilationRate: 1, paddingLeft: zeros, paddingRight: zeros, paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW))
        
        var end_size = conv.shape!
        
        end_size[conv.shape!.count - 1] = conv.shape!.last!.intValue - 1 as NSNumber

        conv = sliceTensor(conv,
                           starts: [NSNumber](repeating: 0, count: x.shape!.count),
                           ends: end_size,
                           strides: [NSNumber](repeating: 1, count: x.shape!.count),
                           name: nil)
        
        conv = reshape(conv, shape: [dims.0,dims.1,dims.2], name: nil)
        var out = addition(x_even, conv, name: nil)
        
        out = reshape(out, shape: [dims.0, dims.1, -1], name: nil)
        let half = constant(0.5, shape: out.shape!, dataType: .float32)
        
        out = multiplication(out, half, name: nil)
        
        return out
    }
    
    public func std(of tensor: MPSGraphTensor, axes: [NSNumber], name: String? = nil) -> MPSGraphTensor {
        let varianceTensor = variance(of: tensor, axes: axes, name: name != nil ? "\(name!).var" : nil)
        return squareRoot(with: varianceTensor, name: name)
    }
    
    public func arange(start: Int = 0, end: Int, step: Int = 1) -> MPSGraphTensor {
        let length = end-start
        guard length > 0 else { fatalError("End value can't be less or equal to start value") }
        
        let elements = length / step
        let numbers = (0..<elements).map { Float($0 * step + start) }
        let data = Data(bytes: numbers, count: numbers.count * MemoryLayout<Float>.size)
        
        return constant(data, shape: [elements as NSNumber], dataType: .float32)
    }
    
    public func eye(size: Int) -> MPSGraphTensor {
        let matrix = (0..<size).flatMap { i in (0..<size).map { j in Float(i == j ? 1 : 0) } }
        let data = Data(bytes: matrix, count: matrix.count * MemoryLayout<Float>.size)
        
        return constant(data, shape: [size as NSNumber, size as NSNumber], dataType: .float32)
    }
    
    public func GLU(of tensor: MPSGraphTensor, dim: Int, name: String? = nil) -> MPSGraphTensor {
        var x = split(tensor, numSplits: 2, axis: dim, name: nil)
        
        let x1 = x[0]
        let x2 = sigmoid(with: x[1], name: nil)
        
        let mul = multiplication(x1, x2, name: name)
        
        return mul
    }
    
    public func convolution1D(source: MPSGraphTensor, weights: MPSGraphTensor, biases: MPSGraphTensor, descriptor: MPSGraphConvolution1dOpDescriptor, name: String? = nil) -> MPSGraphTensor {
        guard source.shape?.count == 3 else { fatalError("Source needs to be dimension of 4") }
        guard weights.shape?.count == 3 else { fatalError("Weights need to be dimension of 4") }

        var source = expandDims(source, axes: [2], name: name)
        let weights = expandDims(weights, axes: [2], name: name)

        let conv2DDescriptor = descriptor.descriptor2D
        source = convolution2D(source, weights: weights, descriptor: conv2DDescriptor, name: name)
        source = squeeze(source, axes: [2], name: name)

        return addition(source, biases, name: name)
    }

    public func convolutionTranspose1D(source: MPSGraphTensor, weights: MPSGraphTensor, biases: MPSGraphTensor, outputShape: [NSNumber], descriptor: MPSGraphConvolution1dOpDescriptor, name: String? = nil) -> MPSGraphTensor {
        
        guard source.shape?.count == 3 else { fatalError("Source needs to be dimension of 4") }
        guard weights.shape?.count == 3 else { fatalError("Weights need to be dimension of 4") }

        var source = expandDims(source, axes: [2], name: name)
           // source = transpose(source, permutation: [0, 3, 2, 1], name: name)
        // NCHW - HWIO

        var weights = expandDims(weights, axes: [2], name: name)
        
             weights = transpose(weights, permutation: [2, 3, 1, 0], name: name)
        //OIHW -> HWIO

        var outputShape = outputShape
        outputShape.insert(1, at: 2)

        let conv2DDescriptor = descriptor.descriptor2D
        source = convolutionTranspose2D(source, weights: weights, outputShape: outputShape, descriptor: conv2DDescriptor, name: name)
        source = squeeze(source, axes: [2], name: name)

        return addition(source, biases, name: name)
    }
    
}