//
//  array.swift
//  Denoiser Swift / Extensions
//
//  Created by Marius Verdier on 28/07/2024.
//

import Foundation

extension Array where Element: NSNumber {
    var dims4D: (NSNumber, NSNumber, NSNumber, NSNumber) {
        guard count == 4 else { fatalError("Array does not have a valid 4D shape") }
        return (self[0], self[1], self[2], self[3])
    }
    
    var dims3D: (NSNumber, NSNumber, NSNumber) {
        guard count == 3 else { fatalError("Array does not have a valid 3D shape") }
        return (self[0], self[1], self[2])
    }
}

extension Array where Element == Int {
    var nsArray: [NSNumber] {
        map {$0 as NSNumber}
    }
}
