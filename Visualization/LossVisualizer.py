import matplotlib.pyplot as plt


class LossVisualizer:
    def __init__(self):
        self.counter = 0
        self.x_axis_plot_points = []
        self.y_axis_training_loss_plot_points = []
        self.y_axis_validation_loss_plot_points = []
        self.plt = plt

    def update_loss_visualization(self, new_data, training=False):
        self.x_axis_plot_points.append(self.counter)
        self.counter += 1
        if training:
            self.y_axis_training_loss_plot_points.append(new_data)
            self.plt.plot(self.x_axis_plot_points, self.y_axis_training_loss_plot_points)
        else:
            self.y_axis_validation_loss_plot_points.append(new_data)
            self.plt.plot(self.x_axis_plot_points, self.y_axis_validation_loss_plot_points)

        self.plt.draw()
