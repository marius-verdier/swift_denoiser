//
//  avaudio_pcm_buffer.swift
//  Denoiser Swift / Extensions
//
//  Created by Marius Verdier on 28/07/2024.
//

import Foundation
import AVFoundation

extension AVAudioPCMBuffer {
    
    func data() -> Data {
        let channelCount = 1
        let channels = UnsafeBufferPointer(start: self.floatChannelData, count: channelCount)
        let ch0Data = NSData(bytes: channels[0], length: Int(self.frameCapacity * self.format.streamDescription.pointee.mBytesPerFrame))
        
        return ch0Data as Data
    }
    
}