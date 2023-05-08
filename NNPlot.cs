using OxyPlot.Axes;
using OxyPlot.Legends;
using OxyPlot.Series;
using OxyPlot;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Markup;

namespace NeuralNetworkVisualizer
{
    class NNPlot
    {
        private NeuralNetwork nn;

        public NeuralNetwork NN { get { return nn; } set { nn = value; } }

        public NNPlot(NeuralNetwork nn)
        {
            this.nn = nn;
        }

        private OxyPalette CreateTransparentPalette(int steps, byte alpha)
        {
            var numberOfColors = 256; // Number of colors in the palette

            // Define the start and end colors of the palette
            var startColor = OxyColor.FromArgb(150, 241, 161, 97); // Yellowish-orange color
            var endColor = OxyColor.FromArgb(150, 178, 72, 74); // Reddish-pink color

            var customPalette = OxyPalette.Interpolate(numberOfColors, startColor, endColor);
            return customPalette;
        }

        public OxyPlot.PlotModel LossAccuracyPlotData()
        {
            var plotModel = new OxyPlot.PlotModel { PlotAreaBackground = OxyColor.FromArgb(255, 51, 52, 57) };

            var accuracyLineSeries = new OxyPlot.Series.LineSeries
            {
                LineStyle = OxyPlot.LineStyle.Solid,
                Color = OxyColor.FromArgb(255, 103, 199, 188),
                StrokeThickness = 3,
                Title = "Accuracy"
            };

            for (int i = 0; i < nn.AccuracyHistory.Count; i++)
            {
                accuracyLineSeries.Points.Add(new OxyPlot.DataPoint(i, nn.AccuracyHistory[i]));
            }

            plotModel.Series.Add(accuracyLineSeries);

            var lossLineSeries = new OxyPlot.Series.LineSeries
            {
                LineStyle = OxyPlot.LineStyle.Solid,
                Color = OxyColor.FromArgb(255, 120, 68, 81),
                StrokeThickness = 3,
                Title = "Loss"
            };

            for (int i = 0; i < nn.LossHistory.Count; i++)
            {
                lossLineSeries.Points.Add(new OxyPlot.DataPoint(i, nn.LossHistory[i]));
            }

            plotModel.Series.Add(lossLineSeries);

            plotModel.Legends.Add(new Legend()
            {
                LegendPosition = LegendPosition.RightBottom,
                LegendTextColor = OxyColor.FromArgb(255, 251, 212, 125)
            });

            plotModel.IsLegendVisible = true;

            // Add and configure y-axis
            var yAxis = new OxyPlot.Axes.LinearAxis
            {
                Position = OxyPlot.Axes.AxisPosition.Left,
                IsZoomEnabled = false,
                MajorGridlineStyle = LineStyle.Solid,
                MinorGridlineStyle = LineStyle.Dot,
                MajorGridlineColor = OxyColor.FromArgb(255, 54, 57, 62),
                MinorGridlineColor = OxyColor.FromArgb(255, 54, 57, 62),
                MajorGridlineThickness = 1,
                MinorGridlineThickness = 1,
                Minimum = 0,
                Maximum = 1,
                TicklineColor = OxyColor.FromArgb(255, 251, 212, 125),
                TextColor = OxyColor.FromArgb(255, 251, 212, 125),
                TitleColor = OxyColor.FromArgb(255, 251, 212, 125)

            };
            plotModel.Axes.Add(yAxis);

            // Add x-axis
            var xAxis = new OxyPlot.Axes.LinearAxis
            {
                Title = "Epoch",
                Position = OxyPlot.Axes.AxisPosition.Bottom,
                IsZoomEnabled = false,
                MajorGridlineStyle = LineStyle.Solid,
                MinorGridlineStyle = LineStyle.Dot,
                MajorGridlineColor = OxyColor.FromArgb(255, 54, 57, 62),
                MinorGridlineColor = OxyColor.FromArgb(255, 54, 57, 62),
                MajorGridlineThickness = 1,
                MinorGridlineThickness = 1,
                TicklineColor = OxyColor.FromArgb(255, 251, 212, 125),
                TextColor = OxyColor.FromArgb(255, 251, 212, 125),
                TitleColor = OxyColor.FromArgb(255, 251, 212, 125)
            };
            plotModel.Axes.Add(xAxis);

            return plotModel;
        }

        public PlotModel PlotData()
        {
            var plotModel = new PlotModel { PlotAreaBackground = OxyColor.FromArgb(255, 51, 52, 57) };

            // Create scatter plot data series for each class
            var class0 = new ScatterSeries { MarkerType = MarkerType.Circle, Title = "Class 0", MarkerSize = 2, MarkerFill = OxyColor.FromArgb(200, 252, 212, 125) };
            var class1 = new ScatterSeries { MarkerType = MarkerType.Circle, Title = "Class 1", MarkerSize = 2, MarkerFill = OxyColor.FromArgb(255, 200, 68, 81) };

            // Separate the data points by class
            for (int i = 0; i < nn.Data.Count; i++)
            {
                if (nn.Labels[i] == 0)
                {
                    class0.Points.Add(new ScatterPoint(nn.Data[i][0], nn.Data[i][1], double.NaN, 255));
                }
                else
                {
                    class1.Points.Add(new ScatterPoint(nn.Data[i][0], nn.Data[i][1], double.NaN, 255));
                }
            }

            // Generate a grid of points
            int gridSize = 100;
            double[] x1Values = CreateVector(-15, 15, gridSize);
            double[] x2Values = CreateVector(-15, 15, gridSize);

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
                Palette = CreateTransparentPalette(500, 128), // Change the second parameter to set the desired alpha value
                Title = "Probability",
                Key = "ColorAxisKey",
                IsAxisVisible = false
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

            plotModel.Axes.Add(new LinearColorAxis { Position = AxisPosition.Right, IsAxisVisible = false });
            plotModel.Axes.Add(colorAxis);

            // Add a legend
            plotModel.IsLegendVisible = true;

            // Set plot boundaries
            var xAxis = new LinearAxis()
            {
                Position = AxisPosition.Bottom,
                Title = "X1",
                TicklineColor = OxyColor.FromArgb(255, 251, 212, 125),
                TextColor = OxyColor.FromArgb(255, 251, 212, 125),
                TitleColor = OxyColor.FromArgb(255, 251, 212, 125)
            };
            var yAxis = new LinearAxis()
            {
                Position = AxisPosition.Left,
                Title = "X2",
                TicklineColor = OxyColor.FromArgb(255, 251, 212, 125),
                TextColor = OxyColor.FromArgb(255, 251, 212, 125),
                TitleColor = OxyColor.FromArgb(255, 251, 212, 125)
            };

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
            for (int i = 0; i < nn.Layers[0].Neurons.Count; i++)
            {
                nn.Layers[0].Neurons[i].Output = input[i];
            }

            // Forward pass
            nn.Forward();

            // Return the output of the last neuron
            return nn.Layers[nn.Layers.Count - 1].Neurons[0].Output;
        }

        public double[] CreateVector(double start, double end, int numElements)
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
}
