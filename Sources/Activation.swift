//
//  Activation.swift
//  Swift-AI
//
//  Created by Collin Hundley on 4/3/17.
//
//

import Foundation


public extension NeuralNet {
    
    public enum ActivationFunction {
        /// Identity activation function.
        /// `f(x) = x`
        /// `f'(x) = 1`
        case identity
        /// Rectified linear activation function (ReLU).
        /// `f(x) = max(0, x)`
        /// `f'(x) = f(x) == 0 ? 0 : 1`
        case rectifiedLinear
        /// Sigmoid activation function.
        /// `f(x) = 1 / (1 + exp(-x))`
        /// `f'(x) = f(x) * (1 - f(x))`
        case sigmoid
        /// Hyperbolic tangent activation function.
        /// `f(x) = tanh(x)`
        /// `f'(x) = 1 - (f(x) * f(x))`
        case hyperbolicTangent
        /// Custom activation function.
        /// An activation function and derivative function must be provided.
        /// The derivative must accept the value `f(x)`.
        case custom(activation: (_ x: Float) -> Float, derivative: (_ y: Float) -> Float)
        
        
        // MARK: Initialization
        
        /// Attempts to create an `ActivationFunction` from a `String`.
        /// This is used to effectively give each function a raw string,
        /// while bypassing the restriction that Swift enums with associated values cannot have raw values.
        static func from(_ string: String) -> ActivationFunction? {
            switch string {
            case "identity":
                return ActivationFunction.identity
            case "ReLU":
                return ActivationFunction.rectifiedLinear
            case "sigmoid":
                return ActivationFunction.sigmoid
            case "hyperbolicTangent":
                return ActivationFunction.hyperbolicTangent
            default:
                return nil
            }
        }
        
        /// Returns the raw string value of the `ActivationFunction`.
        /// This is used to effectively give each function a raw string,
        /// while bypassing the restriction that Swift enums with associated values cannot have raw values.
        func stringValue() -> String {
            switch self {
            case .identity:
                return "identity"
            case .rectifiedLinear:
                return "ReLU"
            case .sigmoid:
                return "sigmoid"
            case .hyperbolicTangent:
                return "hyperbolicTangent"
            case .custom(_, _):
                return "custom"
            }
        }
        
        
        // MARK: Activation
        
        /// The activation function.
        ///
        /// - Parameter x: The x value at any point along the function.
        /// - Returns: The y value at this point.
        public func activation(_ x: Float) -> Float {
            switch self {
            case .identity:
                return x
            case .rectifiedLinear:
                return max(0, x)
            case .sigmoid:
                return 1 / (1 + exp(-x))
            case .hyperbolicTangent:
                return tanh(x)
            case .custom(let activation, _):
                return activation(x)
            }
        }
        
        
        // MARK: Derivative
        
        /// The derivative of the activation function.
        ///
        /// - Parameter y: The y value at any point along the function.
        /// - Returns: The function's derivative at this point.
        public func derivative(_ y: Float) -> Float {
            switch self {
            case .identity:
                return 1
            case .rectifiedLinear:
                return y == 0 ? 0 : 1
            case .sigmoid:
                return y * (1 - y)
            case .hyperbolicTangent:
                return 1 - (y * y)
            case .custom(_, let derivative):
                return derivative(y)
            }
        }
        
    }
    
}
