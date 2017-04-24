//
//  Activation.swift
//  Swift-AI
//
//  Created by Collin Hundley on 4/3/17.
//
//

import Foundation
import Accelerate


public extension NeuralNet {
    
    public enum ActivationFunction {
        
        // MARK: Hidden Activation Functions
        // These functions may be used for all hidden layers of a `NeuralNet`.
        
        
        // --------------------------------
        // IMPORTANT NOTE:
        //
        // All hidden activations include the activation function and its derivative.
        //
        // The derivative accepts a vector of each node's OUTPUT, and computes the gradient with respect to the node's INPUT.
        // Note that no error/cost function is applied here.
        // The derivative is used by `NeuralNet` to propagate the error from the succeeding layer.
        // --------------------------------
        
        
        public enum Hidden {
            /// Linear activation function.
            /// `f(x) = x`
            /// `f'(x) = 1`
            case linear
            /// Rectified linear activation function (ReLU).
            /// `f(x) = max(0, x)`
            /// `f'(x) = f(x) == 0 ? 0 : 1`
            case rectifiedLinear
            /// Hyperbolic tangent activation function.
            /// `f(x) = tanh(x)`
            /// `f'(x) = 1 - (f(x) * f(x))`
            case hyperbolicTangent
            /// Sigmoid activation function.
            /// `f(x) = 1 / (1 + exp(-x))`
            /// `f'(x) = f(x) * (1 - f(x))`
            case sigmoid
            /// Custom activation function.
            /// An activation function and derivative function must be provided.
            /// The derivative must accept a vector `y` of the network's output,
            /// and store the activation's derivative with respect to each node's INPUT in the `result` parameter.
            case custom(activation: (_ x: [Float], _ result: inout [Float], _ rows: Int, _ cols: Int) -> Void,
                derivative: (_ y: [Float], _ result: inout [Float], _ rows: Int, _ cols: Int) -> Void)
            
            
            // MARK: Initialization
            
            /// Attempts to create an `ActivationFunction.Hidden` from a `String`.
            /// This is used to effectively give each function a raw string,
            /// while bypassing the restriction that Swift enums with associated values cannot have raw values.
            static func from(_ string: String) -> ActivationFunction.Hidden? {
                switch string {
                case "linear":
                    return ActivationFunction.Hidden.linear
                case "ReLU":
                    return ActivationFunction.Hidden.rectifiedLinear
                case "hyperbolicTangent":
                    return ActivationFunction.Hidden.hyperbolicTangent
                case "sigmoid":
                    return ActivationFunction.Hidden.sigmoid
                default:
                    return nil
                }
            }
            
            /// Returns the raw string value of the `ActivationFunction.Hidden`.
            /// This is used to effectively give each function a raw string,
            /// while bypassing the restriction that Swift enums with associated values cannot have raw values.
            func stringValue() -> String {
                switch self {
                case .linear:
                    return "linear"
                case .rectifiedLinear:
                    return "ReLU"
                case .hyperbolicTangent:
                    return "hyperbolicTangent"
                case .sigmoid:
                    return "sigmoid"
                case .custom:
                    return "custom"
                }
            }
            
            
            // MARK: Activation
            
            /// The activation function.
            ///
            /// - Parameters:
            ///   - x: A vector of `x` values corresponding to each node's INPUT.
            ///   - result: An array in which the activated inputs `f(x)` should be stored.
            ///   - rows: The number of rows in the matrix (the number of sets in the batch).
            ///   - cols: The number of columns in the matrix (the number of inputs in each input set).
            public func computeActivation(_ x: [Float], result: inout [Float], rows: Int, cols: Int) {
                switch self {
                case .linear:
                    result = x
                case .rectifiedLinear:
                    for i in 0..<(rows * cols) {
                        result[i] = max(0, x[i])
                    }
                case .hyperbolicTangent:
                    result = x.map{tanh($0)}
                case .sigmoid:
                    result = x.map{1 / (1 + exp(-$0))}
                case .custom(let activation, _):
                    activation(x, &result, rows, cols)
                }
            }
            
            
            // MARK: Derivative
            
            /// The derivative of the activation function.
            ///
            /// - Parameters:
            ///   - y: The `f(x)` values correspnding to each node's OUTPUT.
            ///   - result: An array in which to store the activation function's derivative with respect to each node's INPUT.
            ///   - rows: The number of rows in the matrix (the number of sets in the batch).
            ///   - cols: The number of columns in the matrix (the number of inputs in each input set).
            public func calculateDerivative(_ y: [Float], result: inout [Float], rows: Int, cols: Int) {
                switch self {
                case .linear:
                    for i in 0..<(rows * cols) {
                        result[i] = 1
                    }
                case .rectifiedLinear:
                    for i in 0..<(rows * cols) {
                        result[i] = y[i] == 0 ? 0 : 1
                    }
                case .hyperbolicTangent:
                    for i in 0..<(rows * cols) {
                        result[i] = 1 - (y[i] * y[i])
                    }
                case .sigmoid:
                    for i in 0..<(rows * cols) {
                        result[i] = y[i] * (1 - y[i])
                    }
                case .custom(_, let derivative):
                    derivative(y, &result, rows, cols)
                }
            }
            
        }
        
        
        // MARK: Output Activation Functions
        // These functions may be used for the output layer of a `NeuralNet`.
        
        
        // --------------------------------
        // IMPORTANT NOTE:
        //
        // All output activations include the activation function, a cost function, and a 'gradient' calculator.
        // An appropriate cost function has been chosen to correpond to each activation.
        // If a different cost function is desired, a 'custom' activation may be created.
        //
        // The gradient calculator computes the error gradient for each output node with respect to the node's INPUT.
        // This gradient is equal to the product of the activation derivative and the cost derivative.
        // However, many activation/cost combos can be reduced mathematically, which is why these two steps
        // are combined into a single 'gradient' function (as opposed to computing the derivative of activation and cost separately).
        // --------------------------------
        
        
        public enum Output {
            /// Linear activation function.
            /// `f(x) = x`
            case linear
            /// Rectified linear activation function (ReLU).
            /// `f(x) = max(0, x)`
            case rectifiedLinear
            /// Hyperbolic tangent activation function.
            /// `f(x) = tanh(x)`
            case hyperbolicTangent
            /// Sigmoid activation function with mean-squared loss.
            /// `f(x) = 1 / (1 + exp(-x))`
            /// `gradient = -real[i] * (1 - real[i]) * (target[i] - real[i])`
            case sigmoid
            /// Softmax activation with cross entropy loss.
            /// `f(x) = exp(x) / ∑[i]( exp(x[i]) )`
            /// `gradient = real[i] - target[i]`
            case softmax
            /// Custom activation function.
            /// An activation function and gradient calculator must be provided.
            /// The gradient calculator should accept the network's real output and target output (labels),
            /// along with the number of rows (sets in the batch) and columns (network outputs) in the matrix,
            /// and store the computed error gradient (with respect to each node's INPUT) in the `result` parameter.
            case custom(activation: (_ x: [Float], _ result: inout [Float], _ rows: Int, _ cols: Int) -> Void,
                gradient: (_ real: [Float], _ target: [Float], _ result: inout [Float], _ rows: Int, _ cols: Int) -> Void)
            
            
            // MARK: Initialization
            
            /// Attempts to create an `ActivationFunction.Output` from a `String`.
            /// This is used to effectively give each function a raw string,
            /// while bypassing the restriction that Swift enums with associated values cannot have raw values.
            static func from(_ string: String) -> ActivationFunction.Output? {
                switch string {
                case "linear":
                    return ActivationFunction.Output.linear
                case "ReLU":
                    return ActivationFunction.Output.rectifiedLinear
                case "hyperbolicTangent":
                    return ActivationFunction.Output.hyperbolicTangent
                case "sigmoid":
                    return ActivationFunction.Output.sigmoid
                case "softmax":
                    return ActivationFunction.Output.softmax
                default:
                    return nil
                }
            }
            
            /// Returns the raw string value of the `ActivationFunction.Hidden`.
            /// This is used to effectively give each function a raw string,
            /// while bypassing the restriction that Swift enums with associated values cannot have raw values.
            func stringValue() -> String {
                switch self {
                case .linear:
                    return "linear"
                case .rectifiedLinear:
                    return "ReLU"
                case .hyperbolicTangent:
                    return "hyperbolicTangent"
                case .sigmoid:
                    return "sigmoid"
                case .softmax:
                    return "softmax"
                case .custom:
                    return "custom"
                }
            }
            
            
            // MARK: Activation
            
            /// The activation function.
            ///
            /// - Parameters:
            ///   - x: A vector of `x` values corresponding to each node's INPUT.
            ///   - result: An array in which to store the computed activations for each node.
            ///   - rows: The number of rows in the matrix (number of output sets in the batch).
            ///   - cols: The number of columns in the matrix (number of outputs per set).
            public func computeActivation(_ x: [Float], result: inout [Float], rows: Int, cols: Int) {
                switch self {
                case .linear:
                    result = x
                case .rectifiedLinear:
                    result = x.map{max(0, $0)}
                case .hyperbolicTangent:
                    result = x.map{tanh($0)}
                case .sigmoid:
                    result = x.map{1 / (1 + exp(-$0))}
                case .softmax:
                    for row in 0..<rows {
                        var sum: Float = 0
                        for col in 0..<cols {
                            let idx = row * cols + col
                            sum += exp(x[idx])
                        }
                        for col in 0..<cols {
                            let idx = row * cols + col
                            result[idx] = exp(x[idx]) / sum
                        }
                    }
                case .custom(let activation, _):
                    activation(x, &result, rows, cols)
                }
            }
            
            
            // MARK: Gradient
            
            /// Computes the total error gradient for each output node, with respect to the node's INPUT.
            ///
            /// - Parameters:
            ///   - real: The network's real output. Each element is the final output of one output node.
            ///   - target: The network's target output (labels).
            ///   - result: An array in which to store the computed gradients.
            ///   - rows: The number of rows in the matrix (number of output sets in the batch).
            ///   - cols: The number of columns in the matrix (number of outputs per set).
            public func calculateErrorGradient(real: [Float], target: [Float], result: inout [Float], rows: Int, cols: Int) {
                switch self {
                case .linear:
                    // TODO
                    break
                case .rectifiedLinear:
                    // TODO
                    break
                case .hyperbolicTangent:
                    // TODO
                    break
                case .sigmoid:
                    result = zip(real, target).map{(-$0 * (1 - $0) * ($1 - $0))}
                case .softmax:
                    vDSP_vsub(target, 1,
                              real, 1,
                              &result, 1,
                              vDSP_Length(real.count))
                case .custom(_, let gradient):
                    gradient(real, target, &result, rows, cols)
                }
            }
            
        }
        
    }
    
}
