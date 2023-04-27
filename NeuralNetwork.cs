﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;
using MathNet.Numerics.Distributions;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Axes;
using OxyPlot.Wpf;

namespace NeuralNetworkVisualizer
{
    public static class ArrayBuilder
    {
        public static double[] CreateVector(double start, double end, int numElements)
        {
            if (numElements <= 1)
            {
                throw new ArgumentException("The number of elements must be greater than 1.");
            }

            double[] vector = new double[numElements];
            double step = (end - start) / (numElements - 1);

            for (int i = 0; i < numElements; i++)
            {
                vector[i] = start + i * step;
            }

            return vector;
        }
    }

    class NeuralNetwork
    {
        public event EventHandler PlotUpdated;

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

        private OxyPalette CreateTransparentViridisPalette(int steps, byte alpha)
        {
            var viridisPalette = OxyPalettes.Plasma(steps);
            var colors = new OxyColor[steps];

            for (int i = 0; i < steps; i++)
            {
                OxyColor originalColor = viridisPalette.Colors[i];
                colors[i] = OxyColor.FromArgb(alpha, originalColor.R, originalColor.G, originalColor.B);
            }

            return new OxyPalette(colors);
        }


        public PlotModel PlotData()
        {
            var plotModel = new PlotModel();

            // Create scatter plot data series for each class
            var class0 = new ScatterSeries { MarkerType = MarkerType.Circle, Title = "Class 0", MarkerSize = 2, MarkerFill = OxyColors.Blue };
            var class1 = new ScatterSeries { MarkerType = MarkerType.Circle, Title = "Class 1", MarkerSize = 2, MarkerFill = OxyColors.Red };

            // Separate the data points by class
            for (int i = 0; i < data.Count; i++)
            {
                if (labels[i] == 0)
                {
                    class0.Points.Add(new ScatterPoint(data[i][0], data[i][1], double.NaN, 255));
                }
                else
                {
                    class1.Points.Add(new ScatterPoint(data[i][0], data[i][1], double.NaN, 255));
                }
            }

            // Generate a grid of points
            int gridSize = 100;
            double[] x1Values = ArrayBuilder.CreateVector(-15, 15, gridSize);
            double[] x2Values = ArrayBuilder.CreateVector(-15, 15, gridSize);

            // Predict the class probabilities for each point in the grid
            double[,] gridPredictions = new double[gridSize, gridSize];
            for (int i = 0; i < gridSize; i++)
            {
                for (int j = 0; j < gridSize; j++)
                {
                    double[] input = new double[] { x1Values[i], x2Values[j] };
                    double prediction = Predict(input);
                    gridPredictions[i, j] = prediction;
                }
            }

            var colorAxis = new LinearColorAxis
            {
                Position = AxisPosition.Right,
                Palette = CreateTransparentViridisPalette(500, 128), // Change the second parameter to set the desired alpha value
                Title = "Probability",
                Key = "ColorAxisKey"
            };

            // Add heatmap
            var heatmapSeries = new HeatMapSeries
            {
                X0 = -15,
                X1 = 15,
                Y0 = -15,
                Y1 = 15,
                Interpolate = false,
                RenderMethod = HeatMapRenderMethod.Bitmap,
                Data = gridPredictions,
                ColorAxisKey = "ColorAxisKey"
            };

            plotModel.Series.Add(heatmapSeries);

            plotModel.Series.Add(class0);
            plotModel.Series.Add(class1);

            plotModel.Axes.Add(new LinearColorAxis { Position = AxisPosition.Right, Palette = OxyPalettes.Jet(200) });
            plotModel.Axes.Add(colorAxis);

            // Add a legend
            plotModel.IsLegendVisible = true;

            // Set plot boundaries
            var xAxis = new LinearAxis() { Position = AxisPosition.Bottom };
            var yAxis = new LinearAxis() { Position = AxisPosition.Left };

            yAxis.AbsoluteMinimum = -10;
            yAxis.AbsoluteMaximum = 10;
            xAxis.AbsoluteMinimum = -10;
            xAxis.AbsoluteMaximum = 10;

            yAxis.Minimum = -10;
            yAxis.Maximum = 10;
            xAxis.Minimum = -10;
            xAxis.Maximum = 10;

            // Disable zooming
            xAxis.IsZoomEnabled = false;
            yAxis.IsZoomEnabled = false;

            plotModel.Axes.Add(yAxis);
            plotModel.Axes.Add(xAxis);

            return plotModel;
        }



        public double Predict(double[] input)
        {
            // Set input values
            for (int i = 0; i < layers[0].Neurons.Count; i++)
            {
                layers[0].Neurons[i].Output = input[i];
            }

            // Forward pass
            Forward();

            // Return the output of the last neuron
            return layers[layers.Count - 1].Neurons[0].Output;
        }

        public void LoadCircleData()
        {
            int numSamples = 1000;
            double radiusInner = 2.0;
            double radiusOuter = 4.0;

            Random rng = new Random();

            for (int i = 0; i < numSamples; i++)
            {
                // Randomly choose a class (inner circle or outer ring)
                int target = rng.Next(0, 2);

                // Generate a random angle
                double angle = 2 * Math.PI * rng.NextDouble();

                // Choose a random distance from the center based on the class
                double distance;
                if (target == 0)  // Inner circle
                {
                    distance = radiusInner * Math.Sqrt(rng.NextDouble());
                }
                else  // Outer ring
                {
                    distance = radiusOuter + 1.5 * rng.NextDouble();
                }

                // Calculate x and y coordinates based on the angle and distance
                double x1 = distance * Math.Cos(angle);
                double x2 = distance * Math.Sin(angle);

                // Add the data point to the input and target lists
                data.Add(new List<double> { x1, x2 });
                labels.Add(target);
            }
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
                    X[i, j] = Math.Sign(X[i, j]) * 2 - 1 + Normal.Sample(rng, 0, 1);
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
                        currNeuron.Weights.Add(2 * new Random().NextDouble() - 1);
                    }
                    // Generate random bias
                    currNeuron.Bias = 2 * new Random().NextDouble() - 1;
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
            double alpha = 0.1;
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

                if (epoch % 5 == 0)
                {
                    PlotUpdated?.Invoke(this, EventArgs.Empty);
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