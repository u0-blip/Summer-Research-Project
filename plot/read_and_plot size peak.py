# %% reading in data
import sys
import os
import random
import math
import cmath
import numpy as np
import pickle
import traceback
import logging
from matplotlib import cm
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from matplotlib import colors
import pandas as pd

# from scipy import signal
# b, a = signal.butter(2, 0.4, 'low', analog=False)

# parser = ArgumentParser(description='Process some integers.')
# parser.add_argument('-n', '--name', help='Give the name of the file to plot')
# args = parser.parse_args()
# f_name = args.name

# f_names = ['__hexagon_r_3.0_gap_3.0_xloc_0.0_fcen_0.8167_ff_0.5_particle_size_0.05_3.0_5distance_1.0_3.0_20',
# '__sphere_r_3.0_gap_3.0_xloc_0.0_fcen_0.8167_ff_0.5_particle_size_0.05_3.0_5distance_1.0_3.0_20',
# '__triangle_r_3.0_gap_3.0_xloc_0.0_fcen_0.8167_ff_0.5_particle_size_0.05_3.0_5distance_1.0_3.0_20',
# '__cube_r_3.0_gap_3.0_xloc_0.0_fcen_0.8167_ff_0.5_particle_size_0.05_3.0_5distance_1.0_3.0_20']

# datas = []
# for f_name in f_names:
#     dir = '/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3/'
#     output_file_name = dir + f_name + '.std'

#     with open(output_file_name, 'rb') as f:
#         all_data = pickle.load(f)

#     param_name = all_data[0]
#     axis = [np.linspace(*ele) for ele in all_data[1]]
#     datas.append(all_data[2])

# # %%
# plt.plot(axis[0], data[:, :, 0])

# %% linear regression + peak location

# from sklearn.linear_model import LinearRegression

# reg =[]
# x_pred = np.array([[0.5, 3]])

# import matplotlib.cm as cm

# colors = cm.rainbow(np.linspace(0, 1, 4))

# for i, data in enumerate(datas):
#     data_t = data[:, :, 0].transpose()
#     peaks =[]
#     for j in range(data_t.shape[1]):
#         peak, _ = signal.find_peaks(data_t[:, j], height=0)
#         if len(peak) >= 1:
#             peaks.append(peak[0])
#         else:
#             peaks.append(0)

    # if peaks[-1] == 0:
    #     peaks = peaks[:3]
    
    # x = np.array([axis[0][1:]])
    # x = x.transpose()
    # reg.append(LinearRegression().fit(x[:len(peaks)-1], axis[1][peaks][1:]))
    # y_pred = reg[i].predict(x_pred.transpose())
    # plt.scatter(x[:len(peaks)-1], axis[1][peaks][1:], c=colors[i])
    # plt.plot(np.squeeze(x_pred), y_pred, c=colors[i])




# shapes = ['hexagon', 'sphere', 'triangle', 'cube']
# # for i, s in enumerate(shapes):
# #     shapes[i] = s + ' y={:.2f}x + {:.2f}'.format(reg[i].coef_[0], reg[i].intercept_)

# for i in range(data_t.shape[1]):
#     plt.plot(axis[1], data_t[:, i])

# for i in range(data_t.shape[1]):
#     peak = peaks[i]
#     plt.plot(axis[1][peak], data_t[:, i][peak], "x")

# p_size = axis[0]
# plt.title('Cube', fontsize=25)
# plt.xlabel('Gap between particles', fontsize=20)
# plt.ylabel('EM field mean strength', fontsize=20)
# plt.legend(p_size, title = 'Particle size')

# ax = plt.gca()
# ax.tick_params(labelsize=18)

# plt.savefig('/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3' + '/hex_peak_size.png', dpi=400, bbox_inches='tight')



# %%





# %%

# %% plot the normal distribution
import scipy.stats as stats

class P:

    """
    Things that cause the distribution to differ:
    1. initial distribution mode shift
    2. characteristics of the shief
    """

    def __init__(self, beta, sigma, mu):
        self.beta = beta
        self.sigma = sigma
        self.mu = mu

        self.sep = np.array([0, 75, 125, 200])
        self.colors = ['red', 'blue', 'green']


    def get_ref_data(self):
        import scipy.interpolate as interpolate
        ref_paper_size1 = [
            0.0027722772277227747, 0.7196932960471618,
            0.0706930693069307, 0.7340541153351976,
            0.13029702970297027, 0.746362028569269,
            0.20514851485148514, 0.7620900914518931,
            0.27861386138613864, 0.7771324918751416,
            0.3396039603960396, 0.7921871362708788,
            0.41861386138613865, 0.8072240949285767,
            0.48514851485148514, 0.8243343662610534,
            0.5572277227722772, 0.8386911042249261,
            0.6251485148514851, 0.8544259693144886,
        ]
        ref_paper_size2 = [
            0.0027722772277227747, 0.6276322273448718,
            0.034653465346534656, 0.6214177310860856,
            0.0706930693069307, 0.6165731993046633,
            0.12059405940594062, 0.6213333837200513,
            0.16910891089108906, 0.6309040888821705,
            0.2023762376237623, 0.6384286901972638,
            0.24396039603960398, 0.6535023807724283,
            0.28693069306930696, 0.6672006651046782,
            0.3229702970297029, 0.6809057516438666,
            0.3576237623762376, 0.6980473131282592,
            0.38811881188118813, 0.7110708185322347,
            0.4144554455445544, 0.7240984052603732,
            0.4435643564356435, 0.7384973169072631,
            0.47405940594059404, 0.7508337994104752,
            0.4990099009900989, 0.7645497694807647,
            0.5156435643564355, 0.7748387876955632,
            0.5405940594059406, 0.787180711964326,
            0.5599999999999999, 0.8009021238001661,
            0.5849504950495049, 0.8139310709696922,
            0.614059405940594, 0.8324521200211623,
            0.6348514851485149, 0.8482332401179047,
            0.6514851485148514, 0.8640184415388101,
            0.668118811881188, 0.880490665860479,
            0.6847524752475247, 0.8962758672813844,
            0.6972277227722772, 0.9120651500264528,
        ]
        ref_paper_size1 = [ref_paper_size1[0::2], ref_paper_size1[1::2]]
        ref_paper_size2 = [ref_paper_size2[0::2], ref_paper_size2[1::2]]
        self.ref_size_f1 = interpolate.interp1d(ref_paper_size1[0], ref_paper_size1[1], kind='quadratic')
        self.ref_size_f2 = interpolate.interp1d(ref_paper_size2[0], ref_paper_size2[1], kind = 'quadratic')

    def get_sift_leakage(self):
        '''
        the leakage of the sift is defined as the following.
        1 corresponding to a perfect sift
        0 corresponding to a all through sift
        '''
        leakage = 1

    def cut_distribution(self):
        sep_trans = (self.sep - self.mu)/self.sigma

        start = stats.lognorm.ppf(0.001, self.beta) if stats.lognorm.ppf(0.001, self.beta) >=sep_trans[0] else sep_trans[0] 
        stop = stats.lognorm.ppf(0.999, self.beta) if stats.lognorm.ppf(0.999, self.beta) <=sep_trans[-1] else sep_trans[-1] 
        x_range = [
            [start, sep_trans[1]],
            [sep_trans[1], sep_trans[2]],
            [sep_trans[2], stop]
        ]
        return x_range

    def plot_distribution(self):
        plt.figure(figsize=(7, 6))
        for i in range(3):
            plt.plot(self.tran_x[i], self.pdf[i], self.colors[i])


        plt.plot(self.particle_size[0]/max(self.particle_size[0])*200, self.particle_size[1]/max(self.particle_size[1])*max(self.pdf[0]))

        plt.xlim((0, 200))

        # legends
        ranges = ['<75', '>75 <125', '>125']
        line = [r.ljust(13, ' ')+ 'σ/x: {:0.2f}'.format(ele) for r, ele in zip(ranges, self.sig_div_mean)]
        line += ['Data from rock dust particle size']
        plt.legend(line, loc='upper right', fontsize=15)

        # seperation lines        
        for i in range(len(self.sep)):
            plt.plot([self.sep[i]]*2, [0, max(self.pdf[1])*2.2], '--b')

        # plot labels
        plt.title('particle size distribution', fontsize=25)
        plt.xlabel('size (um)', fontsize=20)
        plt.ylabel('fraction', fontsize=20)

        # set tick sizes
        ax = plt.gca()
        ax.tick_params(labelsize=18)

        plt.savefig('/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3' + '/size distribution.png', dpi=400, bbox_inches='tight')
        plt.show()

    def get_distribution(self):
        x_range = self.cut_distribution()

        tran_x = []
        pdf = []
        rescale_val = []

        for i in range(3):
            x = np.linspace(*x_range[i] , 100)
            pdf.append(stats.lognorm.pdf(x, self.beta)/self.sigma)
            tran_x.append(self.sigma*x+self.mu)

            rescale_val.append(np.trapz(pdf[i], x=tran_x[i]))

        rescaled_pdf = []
        mean, std = [], []
        for i in range(3):
            rescaled_pdf.append(pdf[i]/rescale_val[i])
            mean.append(np.trapz(rescaled_pdf[i]*tran_x[i], x=tran_x[i]))
            std.append(np.sqrt(np.trapz(rescaled_pdf[i]*(tran_x[i]-mean[i])**2, x=tran_x[i])))
            
        self.tran_x = tran_x
        self.pdf = pdf
        mean, std = np.array(mean), np.array(std)
        self.sig_div_mean = std/mean
        return tran_x, pdf
    
    def get_ref_loc(self):
        self.x_range = np.linspace(0.01, 0.6, 100)
        self.fill1=self.ref_size_f1(self.sig_div_mean)
        self.fill2=self.ref_size_f2(self.sig_div_mean)
        self.ref_size1 = self.ref_size_f1(self.x_range)
        self.ref_size2 = self.ref_size_f2(self.x_range)
        return self.fill1, self.fill2

    def particle_size_data(self):
        particle_size = [
            0.3444569161249737, -0.00019417475728153444,
            0.39788559251775935, -0.00019417475728153444,
            0.4772193361742764, -0.0005825242718446588,
            0.5342291419898317, -0.00019417475728153444,
            0.6288095610370006, -0.00019417475728153444,
            0.7172935675936876, 0,
            0.7979647861575165, 0.000776699029126221,
            0.9102517578344823, 0.001553398058252442,
            1.0318500122869778, 0.0027184466019417597,
            1.1770485412376943, 0.0034951456310679807,
            1.3681711558877008, 0.004466019417475736,
            1.5220440460623745, 0.004660194174757298,
            1.747140264984919, 0.0048543689320388606,
            2.0181390095617795, 0.006601941747572823,
            2.148713937467825, 0.008543689320388362,
            2.4205323755408332, 0.014951456310679623,
            2.8669836077535114, 0.022524271844660215,
            3.4386266802399845, 0.030873786407767,
            4.098473155641941, 0.03223300970873788,
            4.824070615695076, 0.05398058252427186,
            5.785933997539965, 0.06834951456310681,
            6.853111362654022, 0.09203883495145632,
            8.271235002914736, 0.08718446601941748,
            9.554185402955705, 0.08213592233009709,
            11.387561384042767, 0.10524271844660195,
            13.830441832619094, 0.12563106796116505,
            16.27899598536411, 0.09398058252427186,
            19.40281230010154, 0.06524271844660195,
            22.837904049686582, 0.041941747572815546,
            27.391515589287877, 0.02388349514563108,
            32.2409347226627, 0.013786407766990305,
            38.18756248937668, 0.011262135922330108,
            45.231006527084105, 0.005825242718446616,
            53.57356736299296, 0.002330097087378663,
            66.30162102902106, 0.0011650485436893177,
            78.03972104556931, 0,
            99.03308278777223, 0.0001941747572815622,
            128.86527291554435, 0.0001941747572815622,
            171.94221092684694, 0.0003883495145631244,
            232.3135923661251, 0.0001941747572815622,
            292.9654851084261, 0.0005825242718446588,
        ]
        self.particle_size = [particle_size[0::2], particle_size[1::2]]

        cut_off_loc = 0
        for i, size in enumerate(self.particle_size[0]):
            if size > 50:
                cut_off_loc = i
                break
        self.particle_size[0] = np.array(self.particle_size[0][:cut_off_loc])
        self.particle_size[1] = np.array(self.particle_size[1][:cut_off_loc])
        
    def get_table_data(self):
        cell_text = []

        for i, sig in enumerate(self.sig_div_mean):
            mean = (self.fill1[i]+self.fill2[i])/2
            error_bound = np.abs(self.fill1[i]-self.fill2[i])/2
            cell_text.append([
                '{:0.4f}'.format(sig), 
                '{:0.4f}'.format(self.fill2[i]), 
                '{:0.4f}'.format(self.fill1[i]),
                '{:0.4f}±{:0.4f}'.format(mean, error_bound)
                ])
        cell_text = pd.DataFrame(cell_text)
        cell_text.columns=['σ/x', 'lower', 'upper', 'fill factor']
        cell_text.index = ['<75', '>75 <125', '>125']
        
        with pd.ExcelWriter('/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3' + '/size distribution.xlsx', engine='xlsxwriter') as writer:
            cell_text.to_excel(writer)


    def plot_ref_loc(self):
        colors = [
            'blue',
            'red'
        ]

        plt.figure(figsize=(7, 6))

        plt.plot(self.x_range, self.ref_size1, color=colors[0])
        plt.plot(self.x_range, self.ref_size2, color=colors[1])
        
        for i in range(3):
            plt.plot(self.sig_div_mean[i], self.fill1[i], '+', color=self.colors[i])
        for i in range(3):
            plt.plot(self.sig_div_mean[i], self.fill2[i], '*', color=self.colors[i])
        
        # plot labels
        plt.title('Particle size and fill factor', fontsize=25)
        plt.xlabel('σ/x', fontsize=20)
        plt.ylabel('Fill Factor', fontsize=20)

        # table and legend
        ranges = ['<75', '>75 <125', '>125']
        line = ['OS sand', 'MR sand']
        line += ['σ/x: {:0.2f}'.format(ele) for r, ele in zip(ranges, self.sig_div_mean)]
        
        plt.legend(line, loc='lower right', fontsize=15)

        # set tick sizes
        ax = plt.gca()
        ax.tick_params(labelsize=18)

        plt.savefig('/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3' + '/psize and fill factor.png', dpi=400, bbox_inches='tight')


    def __call__(self):
        self.get_distribution()
        self.plot_distribution()  

        self.get_ref_data()
        self.get_ref_loc()
        self.plot_ref_loc()

        

beta = 0.5
sigma = 60
mu = 0

ranges = ['<75', '>75 <125', '>125']


p = P(beta, sigma, mu)
p.particle_size_data()
p.get_distribution()
p.plot_distribution()
p.get_ref_data()
p.get_ref_loc()
p.plot_ref_loc()
fill1, fill2 = p.get_ref_loc()
p.get_table_data()
# print(fill1, fill2)



# %%
