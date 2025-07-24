import numpy as np
import matplotlib.pyplot as plt
def store_x_and_y_data(x, y, x_name, y_name, name=''):
    header = f"{x_name}\t{y_name}"
    with open(f"{name}.dat", "w") as f:
        f.write(header + "\n")
        for i in range(len(x)):
            f.write(f"{x[i]}\t{y[i]}\n")

def makelogplot(x, y, xlabel, ylabel, xscale='lin', yscale='lin', show=True, save=False, name=''):
    ''' x, y = array,
    xlabel, ylabel, name = str,
    save = bool,
    returns a linear plot of the data.'''
    if len(x) == 1 or len(y) == 1:
        return
    cm_width = cm_height = 5 #cm
    fig = plt.figure(figsize=(cm_width/2.54, cm_height/2.54))
    plt.rcParams['font.family'] = 'Arial'
    ax = fig.add_subplot(1,1,1) 
    ax.plot(x,y,'o', color='k', markerfacecolor='salmon', markersize=7, alpha=0.7)      
    ax.set_xlabel(xlabel, size=7) 
    ax.set_ylabel(ylabel, size=7)
    def minmax_lin(data):
        min_data = min(data)-0.05*max(data)
        max_data = max(data)+0.05*max(data)
        min_data_asp, max_data_asp = min_data, max_data
        return min_data, max_data, min_data_asp, max_data_asp
    def minmax_log(data):
        min_data = 0.1*min(data)
        max_data = 10*max(data)
        min_data_asp, max_data_asp = np.log10(min_data), np.log10(max_data)
        return min_data, max_data, min_data_asp, max_data_asp       
    if xscale=='lin':              
        xmin, xmax, xmin_asp, xmax_asp = minmax_lin(x)
    elif xscale=='log': 
        xmin,xmax,xmin_asp,xmax_asp = minmax_log(x)
        ax.set_xscale('log')
    if yscale=='lin':
        ymin,ymax,ymin_asp,ymax_asp = minmax_lin(y)
    elif yscale=='log':
        ymin,ymax,ymin_asp,ymax_asp = minmax_log(y)
        ax.set_yscale('log')
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)  
    ax.set_aspect((xmax_asp-xmin_asp)/(ymax_asp-ymin_asp))              
    ax.tick_params(axis="both", labelsize=7, direction='out', which='both')            
    if save is True:
        fig.savefig(f'{name}.pdf')
    if show is True:
        plt.show()
    plt.close(fig)
    
def format_seconds(seconds):
    """Formats seconds into a string of the form 'hh:mm:ss'."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"
