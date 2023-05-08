using MathNet.Numerics.Distributions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Markup;

namespace NeuralNetworkVisualizer
{
    class NNDataLoader
    {
        private NeuralNetwork nn;

        public NeuralNetwork NN { get { return nn; } set { nn = value; } }
        public NNDataLoader(NeuralNetwork nn)
        {
            this.nn = nn;
        }
        /*
         * Data Generation
         */
        public void LoadMoonData(int numSamples = 1000, double noise = 0.1)
        {
            Random rng = new Random();

            for (int i = 0; i < numSamples; i++)
            {
                // Randomly choose a class (upper moon or lower moon)
                int target = rng.Next(0, 2);

                // Generate a random angle
                double angle = Math.PI * rng.NextDouble();

                // Calculate x and y coordinates based on the angle and class
                double x1, x2;

                if (target == 0)  // Upper moon
                {
                    x1 = Math.Cos(angle);
                    x2 = Math.Sin(angle);
                }
                else  // Lower moon
                {
                    x1 = 1.0 + Math.Cos(angle);
                    x2 = -Math.Sin(angle) + 0.75;
                }

                // Add noise to the data
                x1 += rng.NextDouble() * noise - noise / 2;
                x2 += rng.NextDouble() * noise - noise / 2;

                // Enlarge and center the data
                x1 *= 5;
                x1 -= 2.5;
                x2 *= 5;
                x2 -= 1;

                // Add the data point to the input and target lists
                nn.Data.Add(new List<double> { x1, x2 });
                nn.Labels.Add(target);
            }
        }

        public void LoadSpiralData(int numSamples = 1000, double noise = 0.1)
        {
            nn.Data.Clear();
            nn.Labels.Clear();

            Random rng = new Random();

            for (int i = 0; i < numSamples / 2; i++)
            {
                for (int spiralClass = 0; spiralClass < 2; spiralClass++)
                {
                    double r = (double)i / (numSamples / 2) * 2;
                    double t = 1.75 * r * Math.PI + (2 * Math.PI * spiralClass);
                    if (spiralClass == 1) { t += Math.PI; }
                    double x1 = r * Math.Cos(t) + noise * rng.NextDouble() * 2 - 1;
                    double x2 = r * Math.Sin(t) + noise * rng.NextDouble() * 2 - 1;

                    nn.Data.Add(new List<double> { x1 * 4 + 3.5, x2 * 4 + 4 });
                    nn.Labels.Add(spiralClass);
                }
            }
        }


        public void LoadCircleData(int numSamples = 1000)
        {
            double radiusInner = 4;
            double radiusOuter = 8;

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
                nn.Data.Add(new List<double> { x1, x2 });
                nn.Labels.Add(target);
            }
        }

        public void LoadXORData(int numSamples = 1000)
        {

            int numFeatures = 2;

            // Initialize a random number generator
            Random rng = new Random(0);

            // Generate random data
            double[,] X = new double[numSamples, numFeatures];
            for (int i = 0; i < numSamples; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    X[i, j] = Normal.Sample(rng, 0, 1);
                }
            }

            // Move positive/negative data to 1 or -1 and add some noise
            for (int i = 0; i < numSamples; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    X[i, j] = Math.Sign(X[i, j]) * 2 - 1 + Normal.Sample(rng, 0, 1);
                }
            }

            // Compare X0 and X1 sign to determine XOR
            double[] y = new double[numSamples];
            for (int i = 0; i < numSamples; i++)
            {
                y[i] = (X[i, 0] > 0) ^ (X[i, 1] > 0) ? 1 : 0;
            }

            // Multiply X by 3
            for (int i = 0; i < numSamples; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    X[i, j] *= 3;
                }
            }

            // Convert 2D array X to List<List<double>>
            for (int i = 0; i < numSamples; i++)
            {
                List<double> row = new List<double>();
                for (int j = 0; j < numFeatures; j++)
                {
                    row.Add(X[i, j]);
                }
                nn.Data.Add(row);
            }

            // Store y in labels
            nn.Labels.AddRange(y);
        }
    }
}
