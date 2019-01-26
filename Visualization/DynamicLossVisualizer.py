"""
The purpose of this class is to have the ability to visualize our current training + validation losses dynamically
on a graph so that it is easier to see whether there is any underfitting / overfitting occuring in the current model
we're training. This is done by passing the losses during the training and validation phases to the function
update_visualization_loss, which will update the graph dynamically.
"""

import matplotlib.pyplot as plt
plt.ion()


class DynamicLossVisualizer:

    def __init__(self, epochs):

        # Minimum value for x is 0 while the max is the number of epochs * the length of the data generator
        self.x_min = 0
        self.x_max = epochs
        self.y_min = 0
        self.y_max = 2

        self.figure = None
        self.axis = None

        # Lines for the training + validation data
        self.training_lines = None
        self.validation_lines = None

        # The x axis for both the training + validation data
        self.x_axis_plot_points = []

        # The losses for the training and validation data
        self.y_axis_training_loss_plot_points = []
        self.y_axis_validation_loss_plot_points = []

        self.counter = 0
        self.on_launch()

    def on_launch(self):
        # Set up plot
        self.figure, self.axis = plt.subplots()
        self.figure.suptitle("Model Losses")
        self.training_lines, = self.axis.plot([], [], '-', color="#20891B", label="Training Loss")
        self.validation_lines, = self.axis.plot([], [], '-', color="#FFA500", label="Validation Loss")
        plt.legend()

        # Autoscale on unknown axis and known lims on the other
        self.axis.set_xlim(self.x_min, self.x_max)
        self.axis.set_ylim(self.y_min, self.y_max)
        self.axis.set_xlabel("Epoch")
        self.axis.set_ylabel("Loss")

    def update_visualization_loss(self, epoch_train_loss, epoch_validation_loss):
        # Update x-axis accordingly for both training + validation plots
        self.x_axis_plot_points.append(self.counter)
        self.training_lines.set_xdata(self.x_axis_plot_points)
        self.validation_lines.set_xdata(self.x_axis_plot_points)
        self.counter += 1

        # Update the losses accordingly for both training + validation plots
        self.y_axis_training_loss_plot_points.append(epoch_train_loss)
        self.training_lines.set_ydata(self.y_axis_training_loss_plot_points)

        self.y_axis_validation_loss_plot_points.append(epoch_validation_loss)
        self.validation_lines.set_ydata(self.y_axis_validation_loss_plot_points)

        self.axis.relim()
        self.axis.autoscale_view()

        # We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
