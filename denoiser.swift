//
//  denoiser.swift
//  Denoiser Swift
//
//  Created by Marius Verdier on 28/07/2024.
//

import Foundation

import AVFoundation

import MetalPerformanceShadersGraph
import Accelerate

open class MPSGraphConvolution1dOpDescriptor {

    init(stride: Int, dilationRate: Int, paddingLeft: Int, paddingRight: Int, paddingStyle: MPSGraphPaddingStyle, dataLayout: MPSGraphTensorNamedDataLayout, weightsLayout: MPSGraphTensorNamedDataLayout) {
        self.stride = stride
        self.dilationRate = dilationRate
        self.paddingLeft = paddingLeft
        self.paddingRight = paddingRight
        self.paddingStyle = paddingStyle
        self.dataLayout = dataLayout
        self.weightsLayout = weightsLayout
    }

    open var descriptor2D: MPSGraphConvolution2DOpDescriptor {
        MPSGraphConvolution2DOpDescriptor(strideInX: stride, strideInY: 1, dilationRateInX: dilationRate, dilationRateInY: 1, groups: 1, paddingLeft: paddingLeft, paddingRight: paddingRight, paddingTop: 0, paddingBottom: 0, paddingStyle: paddingStyle, dataLayout: dataLayout, weightsLayout: weightsLayout)!
    }

    open var stride: Int
    open var dilationRate: Int

    open var paddingLeft: Int
    open var paddingRight: Int
    open var paddingStyle: MPSGraphPaddingStyle

    open var dataLayout: MPSGraphTensorNamedDataLayout
    open var weightsLayout: MPSGraphTensorNamedDataLayout
}

struct DenoiserSequentialEncoderDescriptor {
    internal init (chin: Int, hidden: Int, kernel_size: Int, stride: Int, ch_scale: Int = 2) {
        self.chin = chin
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.stride = stride
        self.ch_scale = ch_scale
    }
    
    var chin: Int = 0
    var hidden: Int = 0
    var kernel_size: Int = 0
    var stride: Int = 0
    var ch_scale: Int = 0
}

struct DenoiserSequentialDecoderDescriptor {
    internal init (chout: Int, hidden: Int, kernel_size: Int, stride: Int, ch_scale: Int = 2, first_layer: Bool = false) {
        self.chout = chout
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.stride = stride
        self.ch_scale = ch_scale
        self.first_layer = first_layer
    }
    
    var chout: Int = 0
    var hidden: Int = 0
    var kernel_size: Int = 0
    var stride: Int = 0
    var ch_scale: Int = 0
    var first_layer: Bool = false
}

public struct ProcessedSignal {
    let signal: [Float]
}

public class DenoiserGraph: MPSGraph {
    
    private let commandQueue: MTLCommandQueue
    private let device: MPSGraphDevice
    private let frameCount: Int
    private let channelCount: Int
    private let kernel_size: Int
    private let stride: Int
    private let resample: Int
    private let depth: Int
    private let ch_schale: Int = 2
    private let growth: Int = 2
    
    private var chin: Int
    private var chout: Int
    private var hidden: Int
    
    
    private var inputSignal: MPSGraphTensor!
    private var outputSignal: MPSGraphTensor!
    
    private lazy var epsilon: MPSGraphTensor = {
        constant(1e-5, dataType: .float32)
    }()
    
    init(device: MPSGraphDevice, queue: MTLCommandQueue, frameCount: Int, channelCount: Int, kernel_size: Int = 8, stride: Int = 4, depth: Int = 5, resample: Int = 4, chin: Int = 1, chout: Int = 1, hidden: Int = 64) {
        self.device = device
        self.commandQueue = queue
        self.frameCount = frameCount
        self.channelCount = channelCount
        self.kernel_size = kernel_size
        self.stride = stride
        self.resample = resample
        self.depth = depth
        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        
        super.init()
        
        self.epsilon = constant(1e-5, dataType: .float32)
        self.inputSignal = placeholder(shape: [1, channelCount as NSNumber, frameCount as NSNumber], name: "inputSignal")
        self.buildGraph()
    }
    
    private func validLength(_ length: Int) -> Int {
        var length = Double(length) * Double(resample)
        for _ in 0..<depth {
            length = Darwin.ceil((length - Double(kernel_size)) / Double(stride)) + 1
            length = max(length, 1)
        }
        for _ in 0..<depth {
            length = (length - 1) * Double(stride) + Double(kernel_size)
        }
        return Int(Darwin.ceil(length / Double(resample)))
    }


    /**
        Load weights from given bin file, exported from PyTorch
        - Parameters:
            - file: The name of the file
            - shape: The shape of the tensor
            - name: The name of the tensor
        - Returns: The tensor
    **/
    private func fileLoad(_ file: String, shape: [NSNumber], name: String? = nil) -> MPSGraphTensor {
        let modelDataURL = Bundle.main.url(forResource: "\(file)", withExtension: "bin")
        let modelData = try! Data(contentsOf: modelDataURL!)
        return constant(modelData, shape: shape, dataType: .float32)
    }
    
    private func convolutionTranspose1DLoad(source: MPSGraphTensor, chIn: Int, chOut: Int, padding: (Int, Int) = (0, 0), kernelSize: Int, stride: Int, dilation: Int = 1, path: String, name: String? = nil) -> MPSGraphTensor {
        
        let convWeights = fileLoad("\(path).weight", shape: [chIn as NSNumber, chOut as NSNumber, kernelSize as NSNumber], name: nil)
        let convBiases = fileLoad("\(path).bias", shape: [1, chOut as NSNumber, 1], name: nil)

        let (B, _, L) = source.shape!.dims3D
        let Lout = (L.intValue - 1) * stride - padding.0 - padding.1 + dilation * (kernelSize - 1) + 1
        
        let outputShape = [B, chOut as NSNumber, Lout as NSNumber]
        return convolutionTranspose1D(source: source, weights: convWeights, biases: convBiases, outputShape: outputShape, descriptor:  MPSGraphConvolution1dOpDescriptor(stride: stride, dilationRate: dilation, paddingLeft: padding.0, paddingRight: padding.1, paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .HWIO))
    }
    
    private func convolution1DLoad(source: MPSGraphTensor, chIn: Int, chOut: Int, padding: (Int, Int) = (0,0), kernel_size: Int, stride: Int, dilation: Int = 1, path: String, name: String? = nil) -> MPSGraphTensor {
        var x = source

        let convWeights = fileLoad("\(path).weight", shape: [chOut as NSNumber, chIn as NSNumber, kernel_size as NSNumber], name: nil)
        let convBias = fileLoad("\(path).bias", shape: [1, chOut as NSNumber, 1], name: nil)

        return convolution1D(source: x, weights: convWeights, biases: convBias, descriptor: MPSGraphConvolution1dOpDescriptor(stride: stride, dilationRate: dilation, paddingLeft: padding.0, paddingRight: padding.1, paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW))
    }
    
    private func LSTMLoad(source: MPSGraphTensor, inputSize: Int, hiddenSize: Int, path: String, name: String? = nil) -> MPSGraphTensor {

        let lstmKeys = [
            "weight_ih_l0": [4 * hiddenSize, inputSize],
            "weight_hh_l0": [4 * hiddenSize, inputSize],
            "bias_ih_l0": [4 * hiddenSize],
            "bias_hh_l0": [4 * hiddenSize],
            "weight_ih_l1": [4 * hiddenSize, inputSize],
            "weight_hh_l1": [4 * hiddenSize, inputSize],
            "bias_ih_l1": [4 * hiddenSize],
            "bias_hh_l1": [4 * hiddenSize],
        ]

        let lstmData = lstmKeys.reduce(into: [String: MPSGraphTensor]()) {
            
            $0[$1.key] = fileLoad("\(path).\($1.key)", shape: $1.value.nsArray)
        }

        func getData(index: Int) -> (MPSGraphTensor, MPSGraphTensor, MPSGraphTensor) {
            let inputWeights = lstmData["weight_ih_l\(index)"]!
            let hiddenWeights = lstmData["weight_hh_l\(index)"]!
            
            let inputBias = lstmData["bias_ih_l\(index)"]!
            let hiddenBias = lstmData["bias_hh_l\(index)"]!
            let bias = addition(inputBias, hiddenBias, name: nil)

            return (inputWeights, hiddenWeights, bias)
        }

        let lstmData0 = getData(index: 0)
        let lstmData1 = getData(index: 1)

        let descriptor = MPSGraphLSTMDescriptor()
        descriptor.bidirectional = false
        descriptor.cellGateActivation = .tanh
        descriptor.activation = .tanh
        descriptor.forgetGateActivation = .sigmoid
        descriptor.outputGateActivation = .sigmoid
        descriptor.inputGateActivation = .sigmoid

        var result = LSTM(source, recurrentWeight: lstmData0.1, inputWeight: lstmData0.0, bias: lstmData0.2, initState: nil, initCell: nil, mask: nil, peephole: nil, descriptor: descriptor, name: nil)[0]

        result = LSTM(result, recurrentWeight: lstmData1.1, inputWeight: lstmData1.0, bias: lstmData1.2, initState: nil, initCell: nil, mask: nil, peephole: nil, descriptor: descriptor, name: name)[0]

        return result
    }
    
    private func encodingLayer(input: MPSGraphTensor, descriptor: DenoiserSequentialEncoderDescriptor, path: String, name: String?  = nil) -> MPSGraphTensor {
        var x = input
        
        x = convolution1DLoad(source: x, chIn: descriptor.chin, chOut: descriptor.hidden, kernel_size: descriptor.kernel_size, stride: descriptor.stride, path: "\(path).0")
        
        x = reLU(with: x, name: nil)
        
        x = convolution1DLoad(source: x, chIn: descriptor.hidden, chOut: descriptor.hidden * descriptor.ch_scale, kernel_size: 1, stride: 1, path: "\(path).2")
        
        x = GLU(of: x, dim: 1)
        
        return x
    }
    
    private func decodingLayer(input: MPSGraphTensor, descriptor: DenoiserSequentialDecoderDescriptor, path: String, name: String?  = nil) -> MPSGraphTensor {
        var x = input
        
        x = convolution1DLoad(source: x, chIn: descriptor.hidden, chOut: descriptor.ch_scale * descriptor.hidden, kernel_size: 1, stride: 1, path: "\(path).0")
        
        x = GLU(of: x, dim: 1)
        
        x = convolutionTranspose1DLoad(source: x, chIn: descriptor.hidden, chOut: descriptor.chout, kernelSize: descriptor.kernel_size, stride: descriptor.stride, path: "\(path).2")
        
        if !descriptor.first_layer {
            x = reLU(with: x, name: nil)
        }
        
        return x
    }
    
    private func buildGraph(max_hidden: Int = 10000) {
        
        var x = inputSignal!
        
        if x.dim == 2 {
            x = expandDims(x, axis: 1, name: "x.unsqueeze")
        }
        
        let mono = mean(of: x, axes: [1], name: "x.mean")
        var std_v = std(of: mono, axes: [-1], name: "x.std")
        std_v = addition(std_v, epsilon, name: "x.stdNZ")
        
        x = division(inputSignal, std_v, name: "x.normalize")
        let leftPad = [NSNumber](repeating: 0, count: x.shape!.count)
        let rightPad = [NSNumber](repeating: 0, count: x.shape!.count - 1) + [validLength(frameCount) - frameCount as NSNumber]
        x = padTensor(x, with: .constant, leftPadding: leftPad, rightPadding: rightPad, constantValue: 0, name: "x.pad")
        
        if resample == 2 {
            x = upsample(x)
        } else if resample == 4 {
            x = upsample(upsample(x))
        }
        
        var encoderDescriptor: [DenoiserSequentialEncoderDescriptor] = []
        var decoderDescriptor: [DenoiserSequentialDecoderDescriptor] = []
        var skip: [MPSGraphTensor] = []
        
        for i in 0..<depth {
            let descriptorE = DenoiserSequentialEncoderDescriptor(chin: chin, hidden: hidden, kernel_size: kernel_size, stride: stride, ch_scale: ch_schale)
            
            encoderDescriptor.append(descriptorE)
            
            let descriptorD = DenoiserSequentialDecoderDescriptor(chout: chout, hidden: hidden, kernel_size: kernel_size, stride: stride, first_layer: i > 0)
            
            decoderDescriptor.insert(descriptorD, at: 0)
            
            chout = hidden
            chin = hidden
            hidden = min(Int(growth*hidden), max_hidden)
        }
        
        for i in 0..<depth {
            let descriptor = encoderDescriptor[i]
            x = encodingLayer(input: x, descriptor: descriptor, path: "encoder.\(i)")
            skip.append(x)
            
        }
        
        x = transpose(x, permutation: [2,0,1], name: nil)
        
        x = LSTMLoad(source: x, inputSize: chin, hiddenSize: chin, path: "lstm.lstm")
        
        x = transpose(x, permutation: [1,2,0], name: nil)
        
        
        for i in 0..<depth {
            let descriptor = decoderDescriptor[i]
            var sk = skip.popLast()!
            let shape_x = x.shape!
            let shape_sk = sk.shape!
            var end_size = shape_sk
            end_size[shape_sk.count - 1] = shape_x.last!
            sk = sliceTensor(sk,
                             starts: [NSNumber](repeating: 0, count: shape_sk.count),
                             ends: end_size,
                             strides: [NSNumber](repeating: 1, count: shape_sk.count),
                             name: nil)
            x = addition(x, sk, name: nil)
            x = decodingLayer(input: x, descriptor: descriptor, path: "decoder.\(i)")
        }
        
        if resample == 2 {
            x = downsample(x)
        } else if resample == 4 {
            x = downsample(downsample(x))
        }
        
        let shape_x = x.shape!
        var end_size = shape_x
        end_size[shape_x.count - 1] = frameCount as NSNumber
        x = sliceTensor(x,
                         starts: [NSNumber](repeating: 0, count: shape_x.count),
                         ends: end_size,
                         strides: [NSNumber](repeating: 1, count: shape_x.count),
                         name: nil)
        
        outputSignal = x
    }

    func dumpTensor(tensorData: MPSGraphTensorData) -> [Float] {
        let shape = tensorData.shape.map{$0.intValue}
        var resultData = [Float](repeating: 0, count: shape.reduce(1, *))
        tensorData.mpsndarray().readBytes(&resultData, strideBytes: nil)
        
        return resultData
    }
    
    public func run(data: Data) -> ProcessedSignal? {
        let signal = data
        let inputSignalData = MPSGraphTensorData(device: device, data: signal, shape: inputSignal.shape!, dataType: .float32)
        let result = run(with: commandQueue, feeds: [inputSignal: inputSignalData], targetTensors: [outputSignal], targetOperations: nil)[outputSignal]!
        
        return ProcessedSignal(signal: dumpTensor(tensorData: result))
    }
    
}

open class Denoiser : NSObject {
    private let chin: Int
    private let chout: Int
    private let growth: Int
    
    private var graph: MPSGraph = .init()
    private var device: MTLDevice
    
    private var commandQueue: MTLCommandQueue
    private var shouldSync: Bool
    
    private let graphDevice: MPSGraphDevice
    
    public init(chin: Int = 1, chout: Int = 1, growth: Int = 2) {
        self.chin = chin
        self.chout = chout
        self.growth = growth
        
        self.device = MTLCreateSystemDefaultDevice()!
        self.graphDevice = MPSGraphDevice(mtlDevice: device)
        self.commandQueue = device.makeCommandQueue()!
        self.shouldSync = !device.hasUnifiedMemory
    }
    
    private func toPCMBuffer(referenceBuffer: AVAudioPCMBuffer, data: NSData) -> AVAudioPCMBuffer? {
        let audioFormat = referenceBuffer.format  // given NSData audio format
        guard let PCMBuffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: UInt32(data.length) / audioFormat.streamDescription.pointee.mBytesPerFrame) else {
            return nil
        }
        PCMBuffer.frameLength = PCMBuffer.frameCapacity
        let channels = UnsafeBufferPointer(start: PCMBuffer.floatChannelData, count: Int(PCMBuffer.format.channelCount))
        data.getBytes(UnsafeMutableRawPointer(channels[0]) , length: data.length)
        
        return PCMBuffer
    }
    
    public func analyze(input: Data) -> [Float] {
        let denoiserGraph = DenoiserGraph(device: graphDevice, queue: commandQueue, frameCount: input.count, channelCount: 1)
        let result = denoiserGraph.run(data: input)!.signal
        
        return result
    }
    
}
