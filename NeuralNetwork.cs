using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;
using MathNet.Numerics.Distributions;

namespace NeuralNetworkVisualizer
{
    internal class NeuralNetwork
    {
        private List<Layer> layers { get; set; }
        private List<List<double>> data { get; set; }
        private List<double> labels { get; set; }
        public NeuralNetwork(int[] numNeurons, string activation)
        {
            this.layers = new List<Layer>();
            this.data = new List<List<double>>();
            this.labels = new List<double>();
            // Hidden layers
            for (int i = 0; i < numNeurons.Length - 1; i++)
            {
                this.layers.Add(new Layer(numNeurons[i], activation));
            }
            // Output layer
            this.layers.Add(new Layer(numNeurons[numNeurons.Length - 1], activation));

            this.InitializeWeightsInputs();
        }

        public void LoadXORData()
        {
            int dataSize = 1000;
            int numFeatures = 2;

            // Initialize a random number generator
            Random rng = new Random(0);

            // Generate random data
            double[,] X = new double[dataSize, numFeatures];
            for (int i = 0; i < dataSize; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    X[i, j] = Normal.Sample(rng, 0, 1);
                }
            }

            // Move positive/negative data to 1 or -1 and add some noise
            for (int i = 0; i < dataSize; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    X[i, j] = Math.Sign(X[i, j]) * 2 - 1 + 0.1 * Normal.Sample(rng, 0, 1);
                }
            }

            // Compare X0 and X1 sign to determine XOR
            double[] y = new double[dataSize];
            for (int i = 0; i < dataSize; i++)
            {
                y[i] = (X[i, 0] > 0) ^ (X[i, 1] > 0) ? 1 : 0;
            }

            // Multiply X by 3
            for (int i = 0; i < dataSize; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    X[i, j] *= 3;
                }
            }

            // Convert 2D array X to List<List<double>>
            for (int i = 0; i < dataSize; i++)
            {
                List<double> row = new List<double>();
                for (int j = 0; j < numFeatures; j++)
                {
                    row.Add(X[i, j]);
                }
                data.Add(row);
            }

            // Store y in labels
            labels.AddRange(y);
        }


        private void InitializeWeightsInputs()
        {
            // For each layer excluding the input layer
            for (int currLayer = 1; currLayer < this.layers.Count; currLayer++)
            {
                // For each neuron in the current layer
                foreach (Neuron currNeuron in this.layers[currLayer].Neurons)
                {
                    // Generate random weights for each neuron in the previous layer
                    foreach (Neuron _ in this.layers[currLayer - 1].Neurons)
                    {
                        currNeuron.Weights.Add(new Random().NextDouble());
                    }
                    // Generate random bias
                    currNeuron.Bias = new Random().NextDouble();
                    // Set inputs to the outputs of the previous layer
                    currNeuron.Inputs = this.layers[currLayer - 1].Neurons.Select(neuron => neuron.Output).ToList();
                }
            }
        }

        public void Forward()
        {
            // send inputs into the first layer
            foreach (Neuron neuron in this.layers[1].Neurons)
            {
                neuron.Inputs = this.layers[0].Neurons.Select(neuron => neuron.Output).ToList();
            }

            // Propagate through each layer excluding the input layer
            for (int currLayer = 1; currLayer < this.layers.Count; currLayer++)
            {
                foreach (Neuron currNeuron in this.layers[currLayer].Neurons)
                {
                    // Take inputs as the outputs of the previous layer
                    currNeuron.Inputs = this.layers[currLayer - 1].Neurons.Select(neuron => neuron.Output).ToList();
                    // Calculated weighted sum (z)
                    currNeuron.CalculateWeightedSum();
                    // Apply activation function to z to find output
                    currNeuron.Output = currNeuron.ActivationFunction();
                }
            }
        }

        public void Backprop(double error)
        {
            // Iterate through each layer backwards excluding the input layer
            for (int currLayer = this.layers.Count - 1; currLayer > 0; currLayer--)
            {
                // Iterate through each neuron in the current layer
                for (int currNeuron = 0; currNeuron < this.layers[currLayer].Neurons.Count; currNeuron++)
                {
                    // error = (target - output)
                    double gradient = -error;
                    // Calculate bias derivative
                    for (int layer = this.layers.Count - 1; layer > currLayer; layer--)
                    {
                        // Multiply by weights and sigmoid derivatives of outputs along the way
                        // Always takes the path of the first neuron in the layer
                        gradient *= this.layers[layer].Neurons[0].ActivationDerivative();
                        // Ensure that the weight connecting currNeuron is multiplied on the final iteration, otherwise 
                        // multiply the weight connecting the first neuron in the layer
                        if (layer == currLayer + 1)
                        {
                            gradient *= this.layers[layer].Neurons[0].Weights[currNeuron];
                        }
                        else
                        {
                            gradient *= this.layers[layer].Neurons[0].Weights[0];
                        }
                    }
                    // Multiply by the sigmoid derivative of currNeuron
                    gradient *= this.layers[currLayer].Neurons[currNeuron].ActivationDerivative();
                    // set the bias derivative
                    this.layers[currLayer].Neurons[currNeuron].BiasDerivative = gradient;
                    // Calculate weight derivatives
                    // For each weight, multiply the gradient by the output of the neuron it connects to
                    for (int currWeight = 0; currWeight < this.layers[currLayer].Neurons[currNeuron].Weights.Count; currWeight++)
                    {
                        this.layers[currLayer].Neurons[currNeuron].WeightsDerivatives.Add(gradient * this.layers[currLayer].Neurons[currNeuron].Inputs[currWeight]);
                    }
                }
            }
        }

        public void UpdateWeights()
        {
            // Learning rate
            double alpha = 0.05;
            // Iterate through each layer excluding the input layer
            for (int currLayer = 1; currLayer < this.layers.Count; currLayer++)
            {
                // Iterate through each neuron in the current layer
                for (int currNeuron = 0; currNeuron < this.layers[currLayer].Neurons.Count; currNeuron++)
                {
                    // Update bias
                    this.layers[currLayer].Neurons[currNeuron].Bias -= alpha * this.layers[currLayer].Neurons[currNeuron].BiasDerivative;
                    // Update weights
                    for (int currWeight = 0; currWeight < this.layers[currLayer].Neurons[currNeuron].Weights.Count; currWeight++)
                    {
                        this.layers[currLayer].Neurons[currNeuron].Weights[currWeight] -= alpha * this.layers[currLayer].Neurons[currNeuron].WeightsDerivatives[currWeight];
                    }
                }
            }
            // Reset derivatives to prepare for next iteration
            foreach (Layer layer in this.layers)
            {
                foreach (Neuron neuron in layer.Neurons)
                {
                    neuron.BiasDerivative = 0;
                    neuron.WeightsDerivatives = new List<double>();
                }
            }
        }
        
        public void Train()
        {
            int epochs = 1000;
            // Train for a number of epochs
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // Iterate through the first 75% of the data for training
                for (int i = 0; i < (int)Math.Round(0.75 * this.data.Count); i++)
                {
                    // Send inputs into the input layer
                    // Allows for more than 2 input features
                    for (int inputNeuron = 0; inputNeuron < this.layers[0].Neurons.Count; inputNeuron++)
                    {
                        this.layers[0].Neurons[inputNeuron].Output = this.data[i][inputNeuron];
                    }
                    // Forward prop
                    this.Forward();
                    // Calculate error (target - output)
                    double error = this.labels[i] - this.layers[this.layers.Count - 1].Neurons[0].Output;
                    // Backpropagation
                    this.Backprop(error);
                    // Update weights
                    this.UpdateWeights();
                }
            }
        }

        public double Test()
        {
            int correct = 0;
            // Iterate through the last 25% of the data for testing
            for (int i = (int)Math.Round(0.75 * this.data.Count); i < this.data.Count; i++)
            {
                // Send inputs into the input layer
                // Allows for more than 2 input features
                for (int inputNeuron = 0; inputNeuron < this.layers[0].Neurons.Count; inputNeuron++)
                {
                    this.layers[0].Neurons[inputNeuron].Output = this.data[i][inputNeuron];
                }
                // Forward prop
                this.Forward();
                // Calculate error (target - output)
                double error = this.labels[i] - this.layers[this.layers.Count - 1].Neurons[0].Output;
                // If the error is less than 0.5, the prediction is correct
                if (Math.Abs(error) < 0.5)
                {
                    correct++;
                }
            }
            // Print accuracy
            return (double)correct / (this.data.Count - (int)Math.Round(0.75 * this.data.Count));
        }

    }
}
