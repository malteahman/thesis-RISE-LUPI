import os
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from support_functions_logging import class_names

def scalars_to_dataframe(log_dir):
    # Initialize an accumulator to collect all scalar data
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={
        event_accumulator.SCALARS: 0,  # 0 means load all scalars
    })

    # Load the data
    ea.Reload()

    # Prepare DataFrame
    all_scalars_df = pd.DataFrame()

    # Iterate over all scalar tags and read the data
    for tag in ea.Tags()['scalars']:
        scalar_events = ea.Scalars(tag)
        df = pd.DataFrame([(e.step, e.value) for e in scalar_events], columns=['Step', tag])
        if all_scalars_df.empty:
            all_scalars_df = df
        else:
            all_scalars_df = pd.merge(all_scalars_df, df, on='Step', how='outer')

    return all_scalars_df


def plot_histogram(text_values, log_dir, title="Histogram of Text Values"):
    # Convert text values to numerical values
    bin_edges = list(range(len(text_values)))
    
    # Plotting the histogram
    plt.figure(figsize=(10, 8))
    plt.bar(bin_edges, text_values, width=1.0, align='edge')
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('mIoU')
    plt.xticks(bin_edges, [str(i) for i in bin_edges]) 
    plt.grid(True)
    plt.savefig(f'{log_dir}/{title}.png') 
    plt.close()  # Close the figure to free up memory

def plot_2_histograms(values_LUPI, values_wo, log_dir, title):
    n_classes = len(values_LUPI)

    # Setting up the positions for the bars
    index = np.arange(n_classes) *1.5  # Class indices
    bar_width = 0.35  # Width of the bars

    # Creating the bar plot
    plt.figure(figsize=(16, 8))

    # Plot bars for the first group
    plt.bar(index, values_LUPI, bar_width, label='Trained with privileged information')

    # Plot bars for the second group, offset by the width of the bars
    plt.bar(index + bar_width, values_wo, bar_width, label='Trained without privileged informaion')

    # Customize the plot
    plt.title('mIoU per class')
    plt.xlabel('Class')
    plt.ylabel('mIoU')
    plt.xticks(index + bar_width / 2, [str(i+1) for i in range(n_classes)])  # Update class labels and font size
    plt.legend()

    textstr = '\n'.join([f'{i+1}: {name}' for i, name in enumerate(class_names)])  # Example with a few class names
    plt.annotate(textstr, xy=(1.02, 0.5), xycoords='axes fraction', fontsize=10,
             bbox=dict(boxstyle="round, pad=0.4", fc="lightgray", ec="black", lw=1))

    plt.grid(True)
    plt.savefig(f'{log_dir}/{title}.png', bbox_inches='tight') 
    plt.close()  # Close the figure to free up memory

    
def handle_df(log_dir):    
    # Convert all scalar data to a pandas DataFrame
    scalar_data_df = scalars_to_dataframe(log_dir)

    # Display the first few rows of the DataFrame
    print(scalar_data_df.shape)

    #search for min loss to find the epoch to use on test etc
    min_val_loss_idx = scalar_data_df['val/loss'].idxmin()
    val_columns = [col for col in scalar_data_df.columns if col.startswith('val/')]
    filtered_df = scalar_data_df[val_columns]
    min_val_loss_row = filtered_df.loc[min_val_loss_idx]

    # Step 5: Print the values
    print(min_val_loss_row)
    print(f"Row with minimum 'val/loss': {min_val_loss_idx}")

# Directory where your TensorBoard logs are stored
model_folder = '2024-02-09_10-11-25'

log_dir = '../../logs/'+model_folder+'/tensorboard'

#extracted manually
#base_values_wo = [0.8038167, 0.6085486, 0.7509828, 0.636876, 0.81301343, 0.39043948, 0.64369786, 0.40655535, 0.7757048, 0.5586019, 0.6027413, 0.67422533, 0.5716616, 0.8249599, 0.002833728, 4.7015583e-06, 0.06651074, 0.27852723, 0.038257338]
#base_values_LUPI = [0.8353008, 0.6128503, 0.76629245, 0.65490055, 0.80666745, 0.41286272, 0.66755223, 0.41386184, 0.7902842, 0.5685992, 0.59474975, 0.6355176, 0.61501586, 0.74447453, 0.0058739297, 4.102077e-06, 0.085229434, 0.3405365, 0.024147043]
#base_values_wo_5 = [0.7710723, 0.5863532, 0.6997886, 0.5748548, 0.78606343, 0.24458766, 0.60252434, 0.32578334, 0.6547799, 0.53388226, 0.5384141, 0.6387581, 0.53271747, 0.80992764, 0.0, 2.0705726e-05, 0.00015649453, 0.41941866, 0.0011152776]
#base_values_LUPI_5 = [0.804736, 0.57583934, 0.71938103, 0.57596546, 0.8587578, 0.2658925, 0.6209361, 0.3633851, 0.66790587, 0.5566513, 0.53037965, 0.62785035, 0.5324492, 0.7470633, 0.0, 0.0, 0.0, 0.52929753, 0.023372378]

#plot_histogram(base_values_wo, log_dir, title="Histogram of mIoU per Class at Min Val Loss Epoch")

#title_5="Histogram of mIoU per class for model trained with and without privileged information, 5 percent of dataset"
#plot_2_histograms(base_values_LUPI_5, base_values_wo_5, log_dir, title="mIoU_hist_base_5")
#plot_2_histograms(base_values_LUPI, base_values_wo, log_dir, title="mIoU_hist_base_100")

ts_priv = [0.8101901, 0.6014085, 0.7213481, 0.65543926, 0.83594084, 0.30111408, 0.6527128, 0.44321617, 0.6686094, 0.571555, 0.6186719, 0.78826904, 0.4934956, 0.72667354, 0.0, 0.0, 0.0, 0.35661477, 0.019608168]
ts_RGB = [0.79133433, 0.5803743, 0.7112892, 0.5733606, 0.8691831, 0.20332421, 0.63076025, 0.4079901, 0.7006701, 0.53786135, 0.5799587, 0.5990952, 0.5235368, 0.7994979, 0.0, 0.0, 0.0, 0.4020761, 0.000113776]
ts_LUPI = [0.779031, 0.5842216, 0.70957416, 0.60938966, 0.7988855, 0.23122343, 0.5871178, 0.42568207, 0.63988674, 0.539346, 0.5763248, 0.722807, 0.5086796, 0.7831484, 0.0, 0.0, 0.0, 0.52737474, 0.01228389]
ts_mtd = [0.8157355, 0.5996452, 0.7307994, 0.6195038, 0.82030404, 0.3034038, 0.6166927, 0.41756344, 0.6780884, 0.55293083, 0.61684215, 0.7524185, 0.5074559, 0.73895687, 0.0, 0.0, 0.0, 0.43109792, 0.02756346]
ts_pretrained_priv = [0.81966066, 0.5974861, 0.73571515, 0.61413807, 0.8099132, 0.32010943, 0.6415876, 0.46791324, 0.7026203, 0.58453214, 0.6273629, 0.7572679, 0.5664096, 0.7783147, 0.0, 0.0, 0.0, 0.48326436, 0.0095482385]
ts_only_priv = [0.7720366, 0.5766848, 0.697344, 0.592332, 0.8263142, 0.1952029, 0.6335951, 0.39989135, 0.68660074, 0.5165202, 0.5802159, 0.7292443, 0.38922715, 0.73678607, 0.0, 0.0, 0.0, 0.6975686, 0.0035010437]

#handle_df(log_dir)

RGB = [0.7710723, 0.5863532, 0.6997886, 0.5748548, 0.78606343, 0.24458766, 0.60252434, 0.32578334, 0.6547799, 0.53388226, 0.5384141, 0.6387581, 0.53271747, 0.80992764, 0.0, 2.0705726e-05, 0.00015649453, 0.41941866, 0.0011152776]
NIR = [0.67802227, 0.4836962, 0.6281959, 0.4902823, 0.7607066, 0.21787532, 0.62364423, 0.36941162, 0.63041526, 0.45128694, 0.50003856, 0.52698326, 0.3825718, 0.5581291, 0.0, 0.0, 0.0, 0.20630305, 0.00046578672]
RGB_NIR = [0.78262985, 0.61278653, 0.7119289, 0.5469558, 0.79815185, 0.29979485, 0.6491624, 0.39602882, 0.7413466, 0.5468166, 0.568979, 0.7588472, 0.593324, 0.78694355, 0.0, 4.6397613e-06, 0.008820477, 0.40024787, 0.006339019]
log_dir = '../../logs/'
plot_2_histograms(NIR, RGB, log_dir, 'NIR vs RGB')
plot_2_histograms(RGB_NIR, RGB, log_dir, 'RGB+NIR vs RGB')


def plot_function(scalar_data_df):
    plots_dir = '../../'+model_folder+'/plots'
    columns_to_plot = scalar_data_df.columns[1:]
    for column in columns_to_plot:
        print(f'{plots_dir}{column}.png')
        plt.figure()
        plt.plot(scalar_data_df['step'], scalar_data_df[column], label=column)
        plt.xlabel('Step')
        plt.ylabel(column)
        plt.title(f'{column} over Steps')
        plt.legend()
        plt.savefig(f'{plots_dir}{column}.png')  # Save the figure with the column name
        plt.close()  # Close the figure to free up memory