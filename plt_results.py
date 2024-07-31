import matplotlib.pyplot as plt


def plot_results(infile, outfile, title):
    file_path = infile

    x = []
    y = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into two parts and convert them to floats
            point = line.split()
            x.append(float(point[0]))
            y.append(float(point[1]))

    plt.figure()
    plt.plot(x)
    plt.plot(y)
    plt.xlabel('Epochs')
    plt.ylabel('HD, SD')
    plt.legend(['HD', 'SD'])
    plt.title(title)
    # plt.grid(True)
    # plt.show()
    plt.savefig(outfile)


if __name__ == '__main__':

    infile = "./exp_zoom_in/metrics/train.txt"
    outfile = "./exp_zoom_in/figures/training.png"
    title = 'Zoom in - Hard Dice, Soft Dice - Training'
    plot_results(infile, outfile, title)

    infile = "./exp_zoom_in/metrics/val.txt"
    outfile = "./exp_zoom_in/figures/validation.png"
    title = 'Zoom in - Hard Dice, Soft Dice - Validation'
    plot_results(infile, outfile, title)

    infile = "./exp_zoom_out/metrics/train.txt"
    outfile = "./exp_zoom_out/figures/training.png"
    title = 'Zoom out - Hard Dice, Soft Dice - Training'
    plot_results(infile, outfile, title)

    infile = "./exp_zoom_out/metrics/val.txt"
    outfile = "./exp_zoom_out/figures/validation.png"
    title = 'Zoom out - Hard Dice, Soft Dice - Validation'
    plot_results(infile, outfile, title)

