//
//  Error.swift
//  NeuralNet-MNIST
//
//  Created by Collin Hundley on 4/22/17.
//
//

import Foundation


public extension NeuralNet {
    
    /// Functions for calculating error on a validation set.
    public enum ErrorFunction {
        /// Mean squared error function.
        /// 1 / n * ∑( (a[i] - t[i]) * (a[i] - t[i]) )
        case meanSquared
        /// Cross entropy error function.
        /// -1 / n * ∑( [t[i] * ln(a[i]) + (1 − t[i]) * ln(1 − a[i])] )
        case crossEntropy
        /// Percentage error for classification problems.
        /// This should generally be used for networks with Softmax activation and one-hot encoded labels.
        /// This function compares the index of the maximum value in each output set (row) and corresponding validation set.
        /// If the index matches, the output is considered "correct".
        /// Returns a value between `0.0` and `1.0`: `incorrect / numSets`.
        case percentage
        /// Custom error function.
        /// `rows` is the number of rows in the matrix (total number of output sets across all batches).
        /// `cols` is the number of columns in the matrix (the network's output size).
        case custom(error: ((_ real: [Float], _ target: [Float], _ rows: Int, _ cols: Int) -> Float))
        
        
        // MARK: Error calculations
        
        /// The calculated error for the given set of real and target outputs.
        ///
        /// - Parameters:
        ///   - real: The set of real outputs from the neural network from the most recent inference cycle.
        ///   - target: The set of target outputs (labels) for the neural network from the most recent inference cycle.
        /// - Returns: A `Float` quantifying the network's error.
        
        
        /// The calculated error for the given set of real and target values.
        ///
        /// - Parameters:
        ///   - real: A serialized array of all outputs from an entire dataset.
        ///   - target: A serialized array of all labels from an entire data set.
        ///   - rows: The number of rows in the matrix (total number of output sets across all batches).
        ///   - cols: The number of columns in the matrix (the network's output size).
        /// - Returns: The error value for the dataset.
        ///     This may take various forms (sum, average, percentage) and depends on the function chosen.
        public func computeError(real: [Float], target: [Float], rows: Int, cols: Int) -> Float {
            switch self {
            case .meanSquared:
                // 1 / n * ∑( (a[i] - t[i]) * (a[i] - t[i]) )
                let sum = zip(real, target).reduce(0) { (sum, pair) -> Float in
                    return (pair.0 - pair.1) * (pair.0 - pair.1)
                }
                return sum / Float(real.count)
            case .crossEntropy:
                // -1 / n * ∑( [t[i] * ln(a[i]) + (1 − t[i]) * ln(1 − a[i])] )
                return -zip(real, target).reduce(0) { (sum, pair) -> Float in
                    let temp = pair.1 * log(pair.0)
                    return sum + temp + (1 - pair.1) * log(1 - pair.0)
                    } / Float(real.count)
            case .percentage:
                // incorrect / batchSize
                var incorrect = 0
                // Iterate through each set of outputs/labels
                for row in 0..<rows {
                    let start = row * cols
                    let realSet = real[start..<(start + cols)]
                    let labelSet = target[start..<(start + cols)]
                    // Compare max values of outputs and labels
                    if realSet.index(of: realSet.max()!) != labelSet.index(of: labelSet.max()!) {
                        // Incorrect answer; increment counter
                        incorrect += 1
                    }
                }
                return Float(incorrect) / Float(rows)
            case .custom(let error):
                return error(real, target, rows, cols)
            }
        }
        
    }
    
}
