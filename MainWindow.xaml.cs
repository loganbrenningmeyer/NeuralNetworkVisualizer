﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace NeuralNetworkVisualizer
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private NeuralNetwork nn;
        public MainWindow()
        {

            InitializeComponent();

        }

        private void Nn_PlotUpdated(object? sender, EventArgs e, NeuralNetwork nn)
        {
            // Update the plot view
            Dispatcher.Invoke(() =>
            {
                plotView.Model = nn.PlotData();
                plotView.InvalidatePlot(true);
            });
        }

        private async Task TrainAsync()
        {
            await Task.Run(() => nn.Train());

            // Update the UI on the main thread after training has finished
            Dispatcher.Invoke(() =>
            {
                Accuracy.Text = "Training...";
                Accuracy.Text = nn.Test().ToString();
                plotView.Model = nn.PlotData();
            });
        }

        private async void PlayButton_Click(object sender, RoutedEventArgs e)
        {
            plotView.DataContext = null;

            int[] numNeurons = new int[] { 2, 5, 7, 3, 2, 1 };
            string activation = "sigmoid";
            nn = new NeuralNetwork(numNeurons, activation);

            // Subscribe to plot update event
            nn.PlotUpdated += (sender, e) => Nn_PlotUpdated(sender, e, nn);

            nn.LoadCircleData();

            await TrainAsync();
        }
    }
}
