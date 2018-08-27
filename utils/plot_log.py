import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_plot_log(base_dir, csv_file_path, index=None):
    """
    base_dir      : '/path/to/base_directory_for_saved_files'
    csv_file_path : '/pash/to/model_name_csv'
    index         : 'acc'  or  'loss'
    """
    file_name = os.path.basename(csv_file_path)
    model_name = file_name.replace('_csv.csv', '')
    df = pd.read_csv(csv_file_path)
    
    epoch    = df['epoch']
    acc      = df['acc']
    loss     = df['loss']
    val_acc  = df['val_acc']
    val_loss = df['val_loss']
    
    if index == 'acc':
        figure_file_name = model_name + '__acc_log.png'
        plt.figure()
        plt.plot(epoch, acc, label='train acc')# ,marker="o", ms=3)
        plt.plot(epoch, val_acc, label='test acc')#, marker="o", ms=3) 
        plt.title(model_name)
        plt.xlabel('epochs') 
        plt.ylabel('accracy') 
        plt.yticks(np.arange(0.0, 1.1, 0.2)) 
        plt.xlim([0, len(epoch)])
        plt.ylim([0.0, 1.0])
        plt.legend(loc='lower right')                     
        plt.grid()                                                 
        plt.savefig(os.path.join(base_dir, figure_file_name)) 
    elif index == 'loss':
        figure_file_name = model_name + '__loss_log.png'
        plt.figure()
        plt.plot(epoch, loss, label='train loss')# ,marker="o", ms=3)
        plt.plot(epoch, val_loss, label='test loss')#, marker="o", ms=3) 
        plt.title(model_name)
        plt.xlabel('epochs') 
        plt.ylabel('loss (crossentropy)') 
        plt.xlim([0, len(epoch)])
        plt.legend(loc='upper right')                     
        plt.grid()                                                 
        plt.savefig(os.path.join(base_dir, figure_file_name))
    elif index is None:
        raise ValueError("[Usage:] plot_log(base_dir, csv_file_path, index='acc' or 'loss')")
    else:
        raise ValueError('Wrong value for keyword argument "index". ')
