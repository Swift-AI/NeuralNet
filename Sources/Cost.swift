//
//  Cost.swift
//  Swift-AI
//
//  Created by Collin Hundley on 4/3/17.
//
//

import Foundation


public extension NeuralNet {
    
    /// The cost function for calculating error on a single set of validation data.
    public enum CostFunction {
        /// Mean squared error function.
        /// 1/2 * ∑[i]( (a[i] - t[i])^2 )
        case meanSquared
        /// Cross entropy error function.
        /// ∑[i]( [t[i] * ln(a[i]) + (1 − t[i]) * ln(1 − a[i])] )
        case crossEntropy
        /// Custom error function.
        case custom(cost: ((_ real: [Float], _ target: [Float]) -> Float), derivative: ((_ real: Float, _ target: Float) -> Float))
        
        
        // MARK: Initialization
        
        /// Attempts to create a `CostFunction` from a `String`.
        /// This is used to effectively give each function a raw string,
        /// while bypassing the restriction that Swift enums with associated values cannot have raw values.
        static func from(_ string: String) -> CostFunction? {
            switch string {
            case "meanSquared":
                return CostFunction.meanSquared
            case "crossEntropy":
                return CostFunction.crossEntropy
            default:
                return nil
            }
        }
        
        /// Returns the raw string value of the `CostFunction`.
        /// This is used to effectively give each function a raw string,
        /// while bypassing the restriction that Swift enums with associated values cannot have raw values.
        func stringValue() -> String {
            switch self {
            case .meanSquared:
                return "meanSquared"
            case .crossEntropy:
                return "crossEntropy"
            case .custom:
                return "custom"
            }
        }
        
        
        // MARK: Cost
        
        /// The calculated cost (error) for the given set of real and target outputs.
        ///
        /// - Parameters:
        ///   - real: The set of real outputs from the neural network from the most recent inference cycle.
        ///   - target: The set of target outputs (labels) for the neural network from the most recent inference cycle.
        /// - Returns: A `Float` quantifying the network's error.
        public func cost(real: [Float], target: [Float]) -> Float {
            switch self {
            case .meanSquared:
                // 1/2 * ∑[i]( (a[i] - t[i])^2 )
                let sum = zip(real, target).reduce(0) { (sum, pair) -> Float in
                    return (pair.0 - pair.1) * (pair.0 - pair.1)
                }
                return sum / 2
            case .crossEntropy:
                // ∑[i]( [t[i] * ln(a[i]) + (1 − t[i]) * ln(1 − a[i])] )
                return -zip(real, target).reduce(0) { (sum, pair) -> Float in
                    let temp = pair.1 * log(pair.0)
                    return sum + temp + (1 - pair.1) * log(1 - pair.0)
                }
            case .custom(let cost, _):
                return cost(real, target)
            }
        }
        
        
        // MARK: Derivative
        
        /// The derivative of the cost function, with respect to the output of the neural network.
        ///
        /// - Parameters:
        ///   - real: The neural network's real output from a particular node.
        ///   - target: The target output (label) for the node corresponding to `real`.
        /// - Returns: The gradient of the cost function with respect to the network's output.
        public func derivative(real: Float, target: Float) -> Float {
            switch self {
            case .meanSquared:
                return real - target
            case .crossEntropy:
                return (real - target) / ((1 - real) * real)
            case .custom(_, let derivative):
                return derivative(real, target)
            }
        }
        
    }
    
}
