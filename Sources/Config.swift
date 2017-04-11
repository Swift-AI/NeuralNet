//
//  Config.swift
//  Swift-AI
//
//  Created by Collin Hundley on 4/3/17.
//
//

import Foundation


public extension NeuralNet {
    
    /// Basic configuration settings for `NeuralNet`.
    public struct Configuration {
        
        /// Possible `Configuration` errors.
        public enum Error: Swift.Error {
            case initialize(String)
        }
        
        /// The activation function to apply to hidden nodes during inference.
        let hiddenActivation: ActivationFunction
        /// The activation function to apply to the output layer during inference.
        let outputActivation: ActivationFunction
        /// The cost function to use for backpropagation.
        let cost: CostFunction
        /// The learning rate to apply during training.
        let learningRate: Float
        /// The momentum factor to apply during training.
        let momentumFactor: Float
        
        public init(hiddenActivation: ActivationFunction, outputActivation: ActivationFunction, cost: CostFunction,
                    learningRate: Float, momentum: Float) throws {
            // Ensure valid parameters
            guard learningRate >= 0 && momentum >= 0 else {
                throw Error.initialize("Learning rate and momentum must be positive.")
            }
            
            // Initialize properties
            self.hiddenActivation = hiddenActivation
            self.outputActivation = outputActivation
            self.cost = cost
            self.learningRate = learningRate
            self.momentumFactor = momentum
        }
        
    }
    
}
