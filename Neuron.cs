﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkVisualizer
{
    class Neuron
    {
        private List<double> weights { get; set; }
        private double bias { get; set; }
        private List<double> inputs { get; set; }
        private double output { get; set; }
        private List<double> weightsDerivatives { get; set; }
        private double biasDerivative { get; set; }
        private string activation { get; set; }
        private double z { get; set; }

        public Neuron(string activation)
        {
            // initialize activation function, weights, weights derivatives, bias, bias derivative, and inputs
            this.activation = activation;

            this.inputs = new List<double>();

            this.weights = new List<double>();
            this.weightsDerivatives = new List<double>();

            this.bias = 0.0;
            this.biasDerivative = 0.0;
        }

        // Calculates weighed sum (Z) of inputs, weights, and bias
        // Z = w1 * x1 + w2 * x2 + ... + wN * xN + b
        public void CalculateWeightedSum()
        {
            this.Z = 0.0;

            for (int i = 0; i < this.inputs.Count; i++)
            {
                this.Z += this.inputs[i] * this.weights[i];
            }

            this.Z += this.bias;
        }

        // Uses the proper activation function given the 
        // neuron's activation function type
        public double ActivationFunction()
        {
            if (this.activation == "sigmoid")
            {
                return this.Sigmoid();
            }
            else if (this.activation == "relu")
            {
                return this.ReLU();
            }
            else
            {
                return this.Tanh();
            }
            
        }

        // Returns the derivative of the proper activation function given the
        // neuron's activation function type
        public double ActivationDerivative()
        {
            if (this.activation == "sigmoid")
            {
                return this.SigmoidDerivative();
            }
            else if (this.activation == "relu")
            {
                return this.ReLUDerivative();
            }
            else
            {
                return this.TanhDerivative();
            }
        }

        // Tanh activation: (e^Z - e^-Z) / (e^Z + e^-Z)
        public double Tanh()
        {
            return Math.Tanh(this.Z);
        }

        // Tanh derivative: 1 - Tanh()^2
        public double TanhDerivative()
        {
            double tanhZ = this.Tanh();
            return 1 - tanhZ * tanhZ;
        }


        // ReLU activation: Z if Z > 0, alpha*Z if Z <= 0
        public double ReLU()
        {
            double alpha = 0.01;
            return this.Z > 0 ? this.Z : alpha * this.Z;
        }

        // ReLU derivative: 1 if Z > 0, alpha if Z <= 0
        public double ReLUDerivative()
        {
            double alpha = 0.01;
            return this.Z > 0 ? 1 : alpha;
        }

        // Sigmoid activation: 1 / (1 + e^-Z)
        public double Sigmoid()
        {
            if (this.Z >= 0)
            {
                return 1.0 / (1.0 + Math.Exp(-this.Z));
            }
            else
            {
                double expZ = Math.Exp(this.Z);
                return expZ / (1.0 + expZ);
            }
        }

        // Sigmoid derivative: Sigmoid() * (1 - Sigmoid())
        public double SigmoidDerivative()
        {
            return this.Sigmoid() * (1 - this.Sigmoid());
        }

        public double Z
        {
            get { return z; }
            set { z = value; }
        } 

        public List<double> Weights
        {
            get { return weights; }
            set { weights = value; }
        }

        public double Bias
        {
            get { return bias; }
            set { bias = value; }
        }

        public List<double> Inputs
        {
            get { return inputs; }
            set { inputs = value; }
        }

        public double Output
        {
            get { return output; }
            set { output = value; }
        }

        public List<double> WeightsDerivatives
        {
            get { return weightsDerivatives; }
            set { weightsDerivatives = value; }
        }

        public double BiasDerivative
        {
            get { return biasDerivative; }
            set { biasDerivative = value; }
        }
    }
}
